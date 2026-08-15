"""
train.py
========
Implements Algorithm 1 of the paper ("Proposed DRL-Based Approach") almost
line-for-line:

    1: Initialize actor/critic weights (orthogonal) and memory buffer D
    2: for iteration = 1 to L do
    3:   for time slot t = 0..N do
    4:     observe s_t
    5:     act with pi_theta_old(a_t|s_t) (with dynamic masking)
    6:     execute a_t, obtain r_t (eq. 17)
    7:     observe s_{t+1}
    8:     store (s_t, a_t, r_t, s_{t+1}) in D
    9:   for epoch = 1 to K do
    10:    shuffle D into minibatches
    11:    compute GAE (eq. 24)
    12:    update actor by maximizing eq. (26)
    13:    update critic by minimizing eq. (27)
    14:  theta_old <- theta, phi_old <- phi
    15:  clear D

Also supports a `use_mask=False, penalty=...` mode that reproduces the
"PPO-based caching and offloading" ablation baseline of Figs. 4-5, which
removes the action mask and instead applies a large penalty to invalid
actions.
"""

from __future__ import annotations
import numpy as np

from config import SystemConfig, PPOConfig
from environment import MECEnvironment, LOCAL, EDGE, CLOUD, CACHE
from ppo_agent import PPOAgent, Transition


def collect_rollout(env: MECEnvironment, agent: PPOAgent, rollout_len: int, use_mask: bool = True,
                     invalid_penalty: float = 50.0):
    """
    Executes `rollout_len` environment steps under the old policy and stores
    the resulting transitions in `agent.buffer` (Algorithm 1, lines 3-10).

    If use_mask is False, the offloading action is instead sampled from the
    *unmasked* distribution and a large penalty (`invalid_penalty`) is
    subtracted from the reward whenever an invalid edge-offload is chosen
    (task requests a service that is not cached); the action is then
    corrected to LOCAL execution so the environment's internal constraints
    still hold, mirroring the "large penalty" baseline described in
    Sec. IV-C / Sec. V-B.
    """
    state = env._state if env._state is not None else env.reset()
    for _ in range(rollout_len):
        state_vec = env.state_vector(state)
        task_types = state["tau"]

        if use_mask:
            (cache_action, offload_action, cache_masks, offload_masks,
             logp, value) = agent.select_action(state_vec, task_types, env, use_old_policy=True)
            penalty = 0.0
        else:
            (cache_action, offload_action, cache_masks, offload_masks,
             logp, value, penalty) = select_action_no_mask(agent, state_vec, task_types, env,
                                                             invalid_penalty)

        next_state, reward, done, info = env.step(cache_action, offload_action)
        reward -= penalty

        agent.buffer.add(Transition(
            state=state_vec, cache_action=cache_action, offload_action=offload_action,
            cache_masks=cache_masks, offload_masks=offload_masks,
            logp=logp, reward=reward, value=value, done=done,
        ))

        if done:
            state = env.reset()
        else:
            state = next_state

    # bootstrap value for the final state (0 if terminal)
    if done:
        last_value = 0.0
    else:
        import torch
        with torch.no_grad():
            sv = torch.as_tensor(env.state_vector(state), dtype=torch.float32,
                                  device=agent.device).unsqueeze(0)
            _, _, v = agent.old_net(sv)
            last_value = float(v.item())
    return last_value


def select_action_no_mask(agent: PPOAgent, state_vec, task_types, env, invalid_penalty):
    """Unmasked action sampling used by the 'PPO-based caching and offloading' baseline."""
    import torch
    from networks import masked_categorical

    net = agent.old_net
    state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        cache_logits, offload_logits, value = net(state_t)
    cache_logits = cache_logits.squeeze(0)
    offload_logits = offload_logits.squeeze(0)

    cfg = agent.sys_cfg
    all_valid_cache = np.ones((agent.K, 2), dtype=np.float32)  # only mask never applied here
    all_valid_offload = np.ones((agent.M, 3), dtype=np.float32)

    cache_action = np.zeros(agent.K, dtype=np.int64)
    remaining_budget = cfg.mec_storage_capacity_mb
    cache_logp = 0.0
    for k in range(agent.K):
        # Storage capacity (C2) is a hard physical constraint (can't store
        # more bits than exist), so it is still respected; only the
        # offloading<->caching coupling (C5) is left unmasked, matching the
        # paper's description that invalid *offloading* actions are the
        # ones handled by masking vs. penalty.
        mask_np = env.get_cache_mask(remaining_budget)
        all_valid_cache[k] = mask_np
        mask = torch.as_tensor(mask_np, device=agent.device)
        dist = masked_categorical(cache_logits[k], mask)
        a_k = dist.sample()
        cache_action[k] = int(a_k.item())
        cache_logp += float(dist.log_prob(a_k).item())
        if cache_action[k] == CACHE:
            remaining_budget -= cfg.service_storage_mb

    offload_action = np.zeros(agent.M, dtype=np.int64)
    offload_logp = 0.0
    penalty = 0.0
    for m in range(agent.M):
        from torch.distributions import Categorical
        dist = Categorical(logits=offload_logits[m])  # unmasked
        a_m = dist.sample()
        offload_action[m] = int(a_m.item())
        offload_logp += float(dist.log_prob(a_m).item())
        if offload_action[m] == EDGE and cache_action[task_types[m]] == 0:
            penalty += invalid_penalty
            offload_action[m] = LOCAL  # environment still requires a feasible action

    total_logp = cache_logp + offload_logp
    return cache_action, offload_action, all_valid_cache, all_valid_offload, total_logp, float(value.item()), penalty


def train(sys_cfg: SystemConfig, ppo_cfg: PPOConfig, use_mask: bool = True,
          invalid_penalty: float = 50.0, seed: int = 0, log_every: int = 1,
          verbose: bool = True):
    """
    Runs Algorithm 1 for `ppo_cfg.total_iterations` outer iterations.
    Returns the trained agent and a history dict with per-iteration total
    reward (used to reproduce the convergence plots of Figs. 3-5).
    """
    env = MECEnvironment(sys_cfg, seed=seed)
    state_dim = env.state_dim
    agent = PPOAgent(state_dim, sys_cfg.M, sys_cfg.K, ppo_cfg, sys_cfg)

    env.reset()
    history = {"iteration": [], "total_reward": [], "avg_cost": []}

    for it in range(1, ppo_cfg.total_iterations + 1):
        agent.buffer.clear()
        last_value = collect_rollout(env, agent, ppo_cfg.rollout_len, use_mask=use_mask,
                                      invalid_penalty=invalid_penalty)

        transitions = agent.buffer.data
        rewards = np.array([t.reward for t in transitions])
        values = np.array([t.value for t in transitions])
        dones = np.array([t.done for t in transitions])

        advantages, returns = agent.compute_gae(rewards, values, dones,
                                                  ppo_cfg.gamma, ppo_cfg.gae_lambda, last_value)

        states = np.stack([t.state for t in transitions])
        cache_actions = np.stack([t.cache_action for t in transitions])
        offload_actions = np.stack([t.offload_action for t in transitions])
        cache_masks = np.stack([t.cache_masks for t in transitions])
        offload_masks = np.stack([t.offload_masks for t in transitions])
        old_logp = np.array([t.logp for t in transitions])

        actor_loss, critic_loss = agent.update(states, cache_actions, offload_actions, old_logp,
                                                 advantages, returns, cache_masks, offload_masks)

        total_reward = float(np.sum(rewards))
        avg_cost = float(-np.mean(rewards))  # reward is negative total cost
        history["iteration"].append(it)
        history["total_reward"].append(total_reward)
        history["avg_cost"].append(avg_cost)

        if verbose and it % log_every == 0:
            print(f"[iter {it:5d}] total_reward={total_reward:9.2f}  "
                  f"actor_loss={actor_loss:8.4f}  critic_loss={critic_loss:8.4f}")

    return agent, history


if __name__ == "__main__":
    sys_cfg = SystemConfig()
    ppo_cfg = PPOConfig(total_iterations=50, rollout_len=64)  # small smoke-test run
    train(sys_cfg, ppo_cfg, verbose=True, log_every=5)
