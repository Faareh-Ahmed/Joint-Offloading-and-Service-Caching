"""
ddpg_agent.py
=============
"DDPG-based caching and offloading" baseline (Sec. V-A, item 4), following
the general approach of [21] and [24]: the deep deterministic policy
gradient algorithm is used to determine service caching and task-offloading
decisions.

DDPG is natively designed for continuous action spaces. To apply it to the
discrete, constrained action space of this problem (K binary caching
decisions + M ternary offloading decisions) we use the standard continuous
relaxation: the actor outputs continuous "preference" scores for every
discrete option, exploration noise (Ornstein-Uhlenbeck / Gaussian) is added
in this continuous space, and the discrete action executed in the
environment is obtained by taking the arg-max over the (masked) scores,
consistent with the "greedy over learned continuous scores" relaxation
commonly used to apply DDPG to discrete MEC offloading/caching problems in
the works this baseline is based on ([21], [24]).

This module intentionally mirrors the architecture (two hidden layers, 256
units) and general training hyperparameters used for the proposed approach
so that comparisons in Figs. 5 and 9 are architecture-matched.
"""

from __future__ import annotations
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import LOCAL, EDGE, CLOUD, CACHE


class DDPGActor(nn.Module):
    def __init__(self, state_dim, M, K, hidden1=256, hidden2=256):
        super().__init__()
        self.M, self.K = M, K
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.Tanh(),
            nn.Linear(hidden1, hidden2), nn.Tanh(),
        )
        self.cache_head = nn.Linear(hidden2, K * 2)
        self.offload_head = nn.Linear(hidden2, M * 3)

    def forward(self, state):
        feat = self.net(state)
        cache_scores = torch.tanh(self.cache_head(feat)).view(-1, self.K, 2)
        offload_scores = torch.tanh(self.offload_head(feat)).view(-1, self.M, 3)
        return cache_scores, offload_scores


class DDPGCritic(nn.Module):
    """Q(s, a) with the (continuous-relaxed) action concatenated to the state."""

    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden1), nn.Tanh(),
            nn.Linear(hidden1, hidden2), nn.Tanh(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, state, action_flat):
        x = torch.cat([state, action_flat], dim=-1)
        return self.net(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)


class DDPGAgent:
    def __init__(self, state_dim, M, K, sys_cfg, hidden1=256, hidden2=256,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.9, tau=0.01,
                 noise_std=0.2, device="cpu"):
        self.M, self.K, self.sys_cfg = M, K, sys_cfg
        self.gamma, self.tau, self.noise_std = gamma, tau, noise_std
        self.device = torch.device(device)
        self.action_dim = K * 2 + M * 3

        self.actor = DDPGActor(state_dim, M, K, hidden1, hidden2).to(self.device)
        self.actor_target = DDPGActor(state_dim, M, K, hidden1, hidden2).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = DDPGCritic(state_dim, self.action_dim, hidden1, hidden2).to(self.device)
        self.critic_target = DDPGCritic(state_dim, self.action_dim, hidden1, hidden2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer()

    def _flatten_scores(self, cache_scores, offload_scores):
        return torch.cat([cache_scores.reshape(cache_scores.shape[0], -1),
                           offload_scores.reshape(offload_scores.shape[0], -1)], dim=-1)

    def select_action(self, state_vec, task_types, env, explore=True):
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            cache_scores, offload_scores = self.actor(state_t)
        cache_scores = cache_scores.squeeze(0).cpu().numpy()
        offload_scores = offload_scores.squeeze(0).cpu().numpy()

        if explore:
            cache_scores = cache_scores + np.random.normal(0, self.noise_std, cache_scores.shape)
            offload_scores = offload_scores + np.random.normal(0, self.noise_std, offload_scores.shape)

        cfg = self.sys_cfg
        cache_action = np.zeros(self.K, dtype=np.int64)
        remaining_budget = cfg.mec_storage_capacity_mb
        for k in range(self.K):
            mask = env.get_cache_mask(remaining_budget)
            scores = np.where(mask > 0, cache_scores[k], -1e9)
            cache_action[k] = int(np.argmax(scores))
            if cache_action[k] == CACHE:
                remaining_budget -= cfg.service_storage_mb

        offload_action = np.zeros(self.M, dtype=np.int64)
        for m in range(self.M):
            mask = env.get_offload_mask(int(task_types[m]), cache_action)
            scores = np.where(mask > 0, offload_scores[m], -1e9)
            offload_action[m] = int(np.argmax(scores))

        action_flat = np.concatenate([cache_scores.reshape(-1), offload_scores.reshape(-1)])
        return cache_action, offload_action, action_flat

    def store(self, state, action_flat, reward, next_state, done):
        self.replay.push(state, action_flat, reward, next_state, float(done))

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return None, None
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_cache, next_offload = self.actor_target(next_states)
            next_action = self._flatten_scores(next_cache, next_offload)
            target_q = self.critic_target(next_states, next_action)
            y = rewards + self.gamma * (1 - dones) * target_q

        q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        cache_s, offload_s = self.actor(states)
        action_pred = self._flatten_scores(cache_s, offload_s)
        actor_loss = -self.critic(states, action_pred).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return float(actor_loss.item()), float(critic_loss.item())


def train_ddpg(sys_cfg, total_episodes=1000, steps_per_episode=None, seed=0, verbose=True,
                log_every=10):
    """Simple training loop for the DDPG baseline, mirroring Algorithm 1's outer structure."""
    from environment import MECEnvironment

    env = MECEnvironment(sys_cfg, seed=seed)
    steps_per_episode = steps_per_episode or sys_cfg.T
    agent = DDPGAgent(env.state_dim, sys_cfg.M, sys_cfg.K, sys_cfg)

    history = {"episode": [], "total_reward": []}
    state = env.reset()
    for ep in range(1, total_episodes + 1):
        env.reset()
        state = env._state
        ep_reward = 0.0
        for _ in range(steps_per_episode):
            state_vec = env.state_vector(state)
            task_types = state["tau"]
            cache_action, offload_action, action_flat = agent.select_action(
                state_vec, task_types, env, explore=True)
            next_state, reward, done, info = env.step(cache_action, offload_action)
            next_vec = env.state_vector(next_state) if next_state is not None else np.zeros_like(state_vec)
            agent.store(state_vec, action_flat, reward, next_vec, done)
            agent.update()
            ep_reward += reward
            state = next_state if next_state is not None else env.reset()

        history["episode"].append(ep)
        history["total_reward"].append(ep_reward)
        if verbose and ep % log_every == 0:
            print(f"[DDPG episode {ep:5d}] total_reward={ep_reward:9.2f}")

    return agent, history
