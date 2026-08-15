"""
ppo_agent.py
============
PPO-clip agent with dynamic invalid-action masking (Sec. IV-B / IV-C).

Implements:
  - action selection with sequential dynamic masking (eq. 22) over the
    composed action space {c(t), d(t)}
  - Generalized Advantage Estimation                       (eq. 21, 24)
  - probability ratio                                       (eq. 23)
  - PPO clipped surrogate objective                         (eq. 25, 26)
  - critic (value function) loss                            (eq. 27)
  - Algorithm 1 update procedure (lines 12-17)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import ActorCritic, masked_categorical
from environment import LOCAL, EDGE, CLOUD, NOT_CACHE, CACHE


# --------------------------------------------------------------------------- #
# Rollout storage
# --------------------------------------------------------------------------- #
@dataclass
class Transition:
    state: np.ndarray
    cache_action: np.ndarray       # (K,)
    offload_action: np.ndarray     # (M,)
    cache_masks: np.ndarray        # (K, 2) masks used at sampling time (eq. 22)
    offload_masks: np.ndarray      # (M, 3) masks used at sampling time (eq. 22)
    logp: float                    # total_logp = cache_logp + offload_logp
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    def __init__(self):
        self.data: List[Transition] = []

    def add(self, transition: Transition):
        self.data.append(transition)

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)


# --------------------------------------------------------------------------- #
# PPO Agent
# --------------------------------------------------------------------------- #
class PPOAgent:
    def __init__(self, state_dim: int, M: int, K: int, ppo_cfg, sys_cfg, device="cpu"):
        self.M, self.K = M, K
        self.ppo_cfg = ppo_cfg
        self.sys_cfg = sys_cfg
        self.device = torch.device(device)

        self.net = ActorCritic(state_dim, M, K, ppo_cfg.hidden1, ppo_cfg.hidden2,
                                ppo_cfg.activation).to(self.device)
        # old policy used for rollout collection (line 6, Algorithm 1)
        self.old_net = ActorCritic(state_dim, M, K, ppo_cfg.hidden1, ppo_cfg.hidden2,
                                    ppo_cfg.activation).to(self.device)
        self.old_net.load_state_dict(self.net.state_dict())

        self.actor_optim = optim.Adam(self.net.actor.parameters(), lr=ppo_cfg.actor_lr)
        self.critic_optim = optim.Adam(self.net.critic.parameters(), lr=ppo_cfg.critic_lr)

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------ #
    # Action selection with dynamic masking (eq. 22)
    # ------------------------------------------------------------------ #
    def select_action(self, state_vec: np.ndarray, task_types: np.ndarray, env,
                       use_old_policy: bool = True, deterministic: bool = False):
        """
        Samples the composed action a_t = {c(t), d(t)} using the dynamic
        masking mechanism:
          1. Sample K caching subactions sequentially, masking "cache" when
             the remaining storage budget is insufficient (C2).
          2. Once c(t) is fixed, build the per-device offloading mask from
             c(t) and the requested service type tau_m(t) (C5), then sample
             the M offloading subactions.

        Returns cache_action (K,), offload_action (M,), total_logp (float),
        and value estimate V(s_t) (float).
        """
        net = self.old_net if use_old_policy else self.net
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            cache_logits, offload_logits, value = net(state_t)
        cache_logits = cache_logits.squeeze(0)     # (K, 2)
        offload_logits = offload_logits.squeeze(0)  # (M, 3)

        cfg = self.sys_cfg
        remaining_budget = cfg.mec_storage_capacity_mb

        cache_action = np.zeros(self.K, dtype=np.int64)
        cache_masks = np.zeros((self.K, 2), dtype=np.float32)
        cache_logp = 0.0
        for k in range(self.K):
            mask_np = env.get_cache_mask(remaining_budget)
            cache_masks[k] = mask_np
            mask = torch.as_tensor(mask_np, device=self.device)
            dist = masked_categorical(cache_logits[k], mask)
            a_k = dist.probs.argmax() if deterministic else dist.sample()
            cache_action[k] = int(a_k.item())
            cache_logp += float(dist.log_prob(a_k).item())
            if cache_action[k] == CACHE:
                remaining_budget -= cfg.service_storage_mb

        offload_action = np.zeros(self.M, dtype=np.int64)
        offload_masks = np.zeros((self.M, 3), dtype=np.float32)
        offload_logp = 0.0
        for m in range(self.M):
            mask_np = env.get_offload_mask(int(task_types[m]), cache_action)
            offload_masks[m] = mask_np
            mask = torch.as_tensor(mask_np, device=self.device)
            dist = masked_categorical(offload_logits[m], mask)
            a_m = dist.probs.argmax() if deterministic else dist.sample()
            offload_action[m] = int(a_m.item())
            offload_logp += float(dist.log_prob(a_m).item())

        total_logp = cache_logp + offload_logp
        return (cache_action, offload_action, cache_masks, offload_masks,
                total_logp, float(value.item()))

    # ------------------------------------------------------------------ #
    # Log-prob / entropy recomputation for PPO update (uses current net)
    # ------------------------------------------------------------------ #
    def evaluate_actions(self, states, cache_actions, offload_actions, cache_masks_batch, offload_masks_batch):
        """
        Recomputes log-probabilities, entropy, and value estimates for a
        batch of stored transitions under the CURRENT policy pi_theta.

        cache_masks_batch   : (batch, K, 2)  precomputed masks used when the
                               transitions were originally sampled
        offload_masks_batch : (batch, M, 3)
        """
        cache_logits, offload_logits, values = self.net(states)

        cache_masked = cache_logits + (1.0 - cache_masks_batch) * (-1.0e9)
        offload_masked = offload_logits + (1.0 - offload_masks_batch) * (-1.0e9)

        cache_dist = torch.distributions.Categorical(logits=cache_masked)
        offload_dist = torch.distributions.Categorical(logits=offload_masked)

        cache_logp = cache_dist.log_prob(cache_actions).sum(dim=1)      # sum over K subactions
        offload_logp = offload_dist.log_prob(offload_actions).sum(dim=1)  # sum over M subactions
        total_logp = cache_logp + offload_logp

        entropy = cache_dist.entropy().sum(dim=1) + offload_dist.entropy().sum(dim=1)

        return total_logp, entropy, values

    # ------------------------------------------------------------------ #
    # Generalized Advantage Estimation -- eq. (21), (24)
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_gae(rewards, values, dones, gamma, lam, last_value=0.0):
        """
        eq. (21): delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        eq. (24): A_t = delta_t + (gamma*lam)*delta_{t+1} + ... + (gamma*lam)^{T-t-1} * delta_{T-1}
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.asarray(values, dtype=np.float64)
        return advantages, returns

    # ------------------------------------------------------------------ #
    # Algorithm 1, lines 12-17: PPO update
    # ------------------------------------------------------------------ #
    def update(self, states, cache_actions, offload_actions, old_logp,
               advantages, returns, cache_masks, offload_masks):
        """
        Performs `epochs_per_update` epochs of minibatch PPO-clip updates.

        eq. (23): r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
        eq. (25): clip(r_t(theta), 1-eps, 1+eps)
        eq. (26): L^CLIP(theta) = E[min(r_t*A_t, clip(r_t,1-eps,1+eps)*A_t)] + beta*H
        eq. (27): L^VF(phi) = E[(V_phi(s_t) - V_target_t)^2]
        """
        cfg = self.ppo_cfg
        n = states.shape[0]

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        cache_actions_t = torch.as_tensor(cache_actions, dtype=torch.long, device=self.device)
        offload_actions_t = torch.as_tensor(offload_actions, dtype=torch.long, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        cache_masks_t = torch.as_tensor(cache_masks, dtype=torch.float32, device=self.device)
        offload_masks_t = torch.as_tensor(offload_masks, dtype=torch.float32, device=self.device)

        # normalize advantages (standard PPO practice, stabilizes training)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        idx = np.arange(n)
        actor_losses, critic_losses = [], []

        for _ in range(cfg.epochs_per_update):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb_idx = idx[start:start + cfg.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                logp, entropy, values = self.evaluate_actions(
                    states[mb_idx_t], cache_actions_t[mb_idx_t], offload_actions_t[mb_idx_t],
                    cache_masks_t[mb_idx_t], offload_masks_t[mb_idx_t],
                )

                ratio = torch.exp(logp - old_logp_t[mb_idx_t])  # eq. (23)

                adv = advantages_t[mb_idx_t]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv  # eq. (25)
                actor_loss = -torch.min(unclipped, clipped).mean() - cfg.entropy_coef * entropy.mean()  # eq. (26)

                critic_loss = ((values - returns_t[mb_idx_t]) ** 2).mean()  # eq. (27)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.net.actor.parameters(), cfg.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.net.critic.parameters(), cfg.max_grad_norm)
                self.critic_optim.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # line 18: theta_old <- theta, phi_old <- phi
        self.old_net.load_state_dict(self.net.state_dict())

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))
