"""
evaluate.py
===========
Utilities to evaluate the trained PPO agent (proposed approach) against the
heuristic baselines of baselines.py, and to reproduce paper-style plots:

  - Fig. 3-5 style convergence curves (total reward vs. iteration/episode)
  - Fig. 6-12 style average-cost-per-device bar/line comparisons
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import MECEnvironment
from baselines import ALL_BASELINES


def evaluate_agent(agent, sys_cfg, n_episodes=10, seed=1000):
    """Runs the trained PPO agent greedily and returns the average per-device cost."""
    env = MECEnvironment(sys_cfg, seed=seed)
    costs = []
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            state_vec = env.state_vector(state)
            task_types = state["tau"]
            cache_action, offload_action, _, _, _, _ = agent.select_action(
                state_vec, task_types, env, use_old_policy=False, deterministic=True)
            state, reward, done, info = env.step(cache_action, offload_action)
            costs.append(info["avg_cost"])
    return float(np.mean(costs))


def evaluate_baseline(scheme_cls, sys_cfg, n_episodes=10, seed=2000):
    env = MECEnvironment(sys_cfg, seed=seed)
    scheme = scheme_cls()
    scheme.reset(sys_cfg)
    costs = []
    for ep in range(n_episodes):
        state = env.reset()
        if hasattr(scheme, "reset"):
            scheme.reset(sys_cfg)
        done = False
        while not done:
            task_types = state["tau"]
            cache_action, offload_action = scheme.act(env, state, task_types)
            state, reward, done, info = env.step(cache_action, offload_action)
            costs.append(info["avg_cost"])
    return float(np.mean(costs))


def compare_all(agent, sys_cfg, n_episodes=10):
    """Returns {scheme_name: avg_cost} for the proposed approach and all baselines."""
    results = {"Proposed Approach": evaluate_agent(agent, sys_cfg, n_episodes)}
    for key, cls in ALL_BASELINES.items():
        results[cls.name] = evaluate_baseline(cls, sys_cfg, n_episodes)
    return results


def plot_convergence(history, title="Convergence Performance", save_path=None):
    """Reproduces the style of Figs. 3-5: total reward vs. number of episodes/iterations."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["iteration"], history["total_reward"], color="tab:blue", linewidth=1.0)
    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cost_comparison(results: dict, title="Average Cost Comparison", save_path=None):
    """Reproduces the style of Figs. 6-12: bar chart of average cost per device."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results.keys())
    values = [results[n] for n in names]
    ax.bar(names, values, color="tab:blue")
    ax.set_ylabel("Average Cost per Device")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def sweep_parameter(sys_cfg_factory, param_name, values, train_fn, ppo_cfg, n_eval_episodes=5):
    """
    Generic helper to reproduce sweeps such as Fig. 6 (task size), Fig. 7
    (CPU cycles), Fig. 8 (bandwidth), Fig. 9 (MEC capacity), Fig. 10 (lambda_t),
    Fig. 11 (storage capacity).

    sys_cfg_factory : callable(value) -> SystemConfig with `param_name` set
    train_fn         : callable(sys_cfg, ppo_cfg) -> (agent, history)
    Returns {value: {scheme_name: avg_cost}}
    """
    sweep_results = {}
    for v in values:
        cfg = sys_cfg_factory(v)
        agent, _ = train_fn(cfg, ppo_cfg, verbose=False)
        sweep_results[v] = compare_all(agent, cfg, n_episodes=n_eval_episodes)
    return sweep_results
