"""
generate_all_figures.py
========================
Generates the following figures from the research paper via live DRL training
runs and environment evaluations:

  Fig. 5  - Convergence comparison of DRL schemes
  Fig. 7  - Impact of computational workload (CPU cycles) on average cost
  Fig. 8  - Impact of channel bandwidth on average cost
  Fig. 9  - Impact of MEC computing capacity on average cost
  Fig. 11 - Impact of MEC storage capacity on average cost

Reference:
  Shang, C., Huang, Y., Sun, Y., & Guizani, M. (2024).
  "Joint Computation Offloading and Service Caching in Mobile Edge-Cloud
  Computing via Deep Reinforcement Learning."
  IEEE Internet of Things Journal, 11(24), 40331-40344.

Usage:
  python generate_all_figures.py [--iterations 500] [--rollout-len 128] [--M 5] [--K 8]
  python generate_all_figures.py --figures 5,7,8
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SystemConfig, PPOConfig
from environment import MECEnvironment
from train import train
from ddpg_agent import train_ddpg
from evaluate import evaluate_agent, compare_all

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['grid.color'] = '#d0d0d0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

FIGURES_DIR = "figures"


def ensure_fig_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


SCHEME_COLORS = {
    "Proposed Approach": "#1f77b4",
    "PPO-based (Penalty=50)": "#ff7f0e",
    "PPO-based": "#ff7f0e",
    "DDPG-based": "#2ca02c",
    "MP Caching and Offloading": "#d62728",
    "LRU Caching and Offloading": "#9467bd",
    "Cloud Computing": "#8c564b",
    "Local Computing": "#e377c2",
    "Random Caching and Offloading": "#7f7f7f"
}

SCHEME_MARKERS = {
    "Proposed Approach": "o",
    "PPO-based (Penalty=50)": "s",
    "PPO-based": "s",
    "DDPG-based": "^",
    "MP Caching and Offloading": "D",
    "LRU Caching and Offloading": "v",
    "Cloud Computing": "p",
    "Local Computing": "h",
    "Random Caching and Offloading": "x"
}


# ==============================================================================
# Fig 5: DRL Convergence Comparison (Proposed PPO vs Penalty PPO vs DDPG)
# ==============================================================================
def draw_fig5(sys_cfg, iterations, rollout_len, save_path="figures/fig5_drl_convergence.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ppo_cfg = PPOConfig(total_iterations=iterations, rollout_len=rollout_len)

    print(f"Training PPO (masked) for Fig 5 across {iterations} iterations...")
    agent_masked, hist_m = train(sys_cfg, ppo_cfg, use_mask=True, verbose=False)

    print(f"Training PPO (penalty=50) for Fig 5 across {iterations} iterations...")
    agent_p50, hist_p = train(sys_cfg, ppo_cfg, use_mask=False, invalid_penalty=50, verbose=False)

    print(f"Training DDPG for Fig 5 across {iterations} iterations...")
    agent_ddpg, hist_d = train_ddpg(sys_cfg, total_episodes=iterations,
                                     steps_per_episode=rollout_len, verbose=False)

    ax.plot(hist_m["iteration"], hist_m["total_reward"],
            label="Proposed Approach (PPO + Mask)", color="#1f77b4", lw=1.2)
    ax.plot(hist_p["iteration"], hist_p["total_reward"],
            label="PPO-based (Penalty = 50)", color="#ff7f0e", lw=1.0)
    ax.plot(hist_d["episode"], hist_d["total_reward"],
            label="DDPG-based Baseline", color="#2ca02c", lw=1.0)

    ax.set_xlabel("Number of Episodes", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Reward", fontsize=10, fontweight="bold")
    ax.set_title("Fig. 5: Convergence Performance Comparison of DRL Schemes",
                 fontsize=11, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")
    return agent_masked, agent_p50, agent_ddpg


# ==============================================================================
# Helper for Parameter Sweeps (Figs 7, 8, 9)
# ==============================================================================
def plot_sweep_figure(x_vals, data_dict, x_label, title, save_path,
                      exclude_schemes=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    if exclude_schemes is None:
        exclude_schemes = []

    for name, y_vals in data_dict.items():
        if name in exclude_schemes:
            continue
        color = SCHEME_COLORS.get(name, None)
        marker = SCHEME_MARKERS.get(name, "o")
        ax.plot(x_vals, y_vals, label=name, color=color, marker=marker,
                markersize=5, lw=1.2)

    ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Cost per Device", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def evaluate_trained_agents_on_sweep(agent_masked, agent_p50, agent_ddpg,
                                      config_factory, values, n_eval_episodes=5):
    data = {
        "Proposed Approach": [],
        "PPO-based (Penalty=50)": [],
        "DDPG-based": [],
        "MP Caching and Offloading": [],
        "LRU Caching and Offloading": [],
        "Cloud Computing": [],
        "Local Computing": [],
        "Random Caching and Offloading": []
    }

    for v in values:
        cfg = config_factory(v)

        cost_m = evaluate_agent(agent_masked, cfg, n_episodes=n_eval_episodes, use_mask=True)
        data["Proposed Approach"].append(cost_m)

        cost_p = evaluate_agent(agent_p50, cfg, n_episodes=n_eval_episodes, use_mask=False)
        data["PPO-based (Penalty=50)"].append(cost_p)

        costs_d = []
        env_d = MECEnvironment(cfg, seed=777)
        for _ in range(n_eval_episodes):
            s = env_d.reset()
            done = False
            while not done:
                s_vec = env_d.state_vector(s)
                c_a, o_a, _ = agent_ddpg.select_action(s_vec, s["tau"], env_d, explore=False)
                s, _, done, info = env_d.step(c_a, o_a)
                costs_d.append(info["avg_cost"])
        data["DDPG-based"].append(float(np.mean(costs_d)))

        b_res = compare_all(agent_masked, cfg, n_episodes=n_eval_episodes)
        for b_name in ["MP Caching and Offloading", "LRU Caching and Offloading",
                       "Cloud Computing", "Local Computing", "Random Caching and Offloading"]:
            if b_name in b_res:
                data[b_name].append(b_res[b_name])

    return data


# ==============================================================================
# Fig 7: Impact of Computational Workload (CPU Cycles) on Average Cost
# ==============================================================================
def draw_fig7(agent_m, agent_p, agent_d,
              save_path="figures/fig7_impact_of_cpu_cycles.png"):
    scale_factors = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    print("Evaluating sweep for Fig 7 (CPU Cycles)...")
    M, K = agent_m.sys_cfg.M, agent_m.sys_cfg.K
    data = evaluate_trained_agents_on_sweep(
        agent_m, agent_p, agent_d,
        lambda v: SystemConfig(cpu_scale_factor=v, M=M, K=K),
        scale_factors
    )
    plot_sweep_figure(
        scale_factors, data,
        "Computational Workload Scaling Factor (CPU Cycles)",
        "Fig. 7: Impact of Computational Workload on Average Cost",
        save_path,
        exclude_schemes=["Local Computing", "Cloud Computing"]
    )


# ==============================================================================
# Fig 8: Impact of Channel Bandwidth on Average Cost
# ==============================================================================
def draw_fig8(agent_m, agent_p, agent_d,
              save_path="figures/fig8_impact_of_bandwidth.png"):
    bandwidths = np.array([1, 2, 3, 4, 5, 6])
    print("Evaluating sweep for Fig 8 (Bandwidth)...")
    M, K = agent_m.sys_cfg.M, agent_m.sys_cfg.K
    data = evaluate_trained_agents_on_sweep(
        agent_m, agent_p, agent_d,
        lambda v: SystemConfig(bandwidth_mhz=float(v), M=M, K=K),
        bandwidths
    )
    plot_sweep_figure(
        bandwidths, data,
        "Channel Bandwidth (MHz)",
        "Fig. 8: Impact of Channel Bandwidth on Average Cost",
        save_path,
        exclude_schemes=["Local Computing", "Cloud Computing"]
    )


# ==============================================================================
# Fig 9: Impact of MEC Computing Capacity on Average Cost
# ==============================================================================
def draw_fig9(agent_m, agent_p, agent_d,
              save_path="figures/fig9_impact_of_mec_computing_capacity.png"):
    mec_capacities = np.array([8, 10, 12, 14, 16, 18])
    print("Evaluating sweep for Fig 9 (MEC Compute Capacity)...")
    M, K = agent_m.sys_cfg.M, agent_m.sys_cfg.K
    data = evaluate_trained_agents_on_sweep(
        agent_m, agent_p, agent_d,
        lambda v: SystemConfig(f_edge_hz=float(v) * 1e9, M=M, K=K),
        mec_capacities
    )
    plot_sweep_figure(
        mec_capacities, data,
        "MEC Computing Capacity (GHz)",
        "Fig. 9: Impact of MEC Computing Capacity on Average Cost",
        save_path,
        exclude_schemes=["Local Computing", "Cloud Computing"]
    )


# ==============================================================================
# Fig 11: Impact of MEC Storage Capacity on Average Cost
# ==============================================================================
def draw_fig11(iterations, rollout_len, M=5, K=8,
               save_path="figures/fig11_storage_capacity.png"):
    capacities = np.array([50, 75, 100, 125, 150, 175, 200])
    print("Evaluating sweep for Fig 11 (MEC Storage Capacity)...")
    data = {
        "Proposed Approach": [],
        "PPO-based (Penalty=50)": [],
        "DDPG-based": [],
        "MP Caching and Offloading": [],
        "LRU Caching and Offloading": [],
        "Cloud Computing": [],
        "Local Computing": [],
        "Random Caching and Offloading": []
    }

    for u in capacities:
        cfg = SystemConfig(mec_storage_capacity_mb=float(u), M=M, K=K)
        ppo_cfg = PPOConfig(total_iterations=iterations, rollout_len=rollout_len)
        agent_m, _ = train(cfg, ppo_cfg, use_mask=True, verbose=False)
        agent_p, _ = train(cfg, ppo_cfg, use_mask=False, invalid_penalty=50, verbose=False)
        agent_d, _ = train_ddpg(cfg, total_episodes=iterations,
                                 steps_per_episode=rollout_len, verbose=False)

        data["Proposed Approach"].append(
            evaluate_agent(agent_m, cfg, n_episodes=3, use_mask=True))
        data["PPO-based (Penalty=50)"].append(
            evaluate_agent(agent_p, cfg, n_episodes=3, use_mask=False))

        costs_d = []
        env = MECEnvironment(cfg, seed=999)
        for _ in range(3):
            s = env.reset()
            done = False
            while not done:
                s_vec = env.state_vector(s)
                c_a, o_a, _ = agent_d.select_action(s_vec, s["tau"], env, explore=False)
                s, _, done, info = env.step(c_a, o_a)
                costs_d.append(info["avg_cost"])
        data["DDPG-based"].append(float(np.mean(costs_d)))

        baselines_res = compare_all(agent_m, cfg, n_episodes=3)
        for b_name in ["MP Caching and Offloading", "LRU Caching and Offloading",
                       "Cloud Computing", "Local Computing", "Random Caching and Offloading"]:
            if b_name in baselines_res:
                data[b_name].append(baselines_res[b_name])

    plot_sweep_figure(
        capacities, data,
        "MEC Storage Capacity U (Mb)",
        "Fig. 11: Impact of MEC Storage Capacity on Average Cost",
        save_path,
        exclude_schemes=["Local Computing", "Cloud Computing"]
    )


# ==============================================================================
# Master Execution Function
# ==============================================================================
def generate_figures(iterations=50, rollout_len=64, M=5, K=8, selected_figs=None):
    ensure_fig_dir()
    sys_cfg = SystemConfig(M=M, K=K)

    all_supported = [5, 7, 8, 9, 11]
    if selected_figs is None:
        selected_figs = all_supported

    unsupported = [f for f in selected_figs if f not in all_supported]
    if unsupported:
        print(f"Warning: figures {unsupported} are not available. "
              f"Supported figures: {all_supported}")
        selected_figs = [f for f in selected_figs if f in all_supported]

    print(f"Generating figures {selected_figs} "
          f"(iterations={iterations}, rollout_len={rollout_len}, M={M}, K={K})")

    agent_m, agent_p, agent_d = None, None, None

    if 5 in selected_figs or any(f in selected_figs for f in [7, 8, 9]):
        agent_m, agent_p, agent_d = draw_fig5(
            sys_cfg, iterations=iterations, rollout_len=rollout_len)

    if 7 in selected_figs:
        draw_fig7(agent_m, agent_p, agent_d)
    if 8 in selected_figs:
        draw_fig8(agent_m, agent_p, agent_d)
    if 9 in selected_figs:
        draw_fig9(agent_m, agent_p, agent_d)
    if 11 in selected_figs:
        draw_fig11(iterations, rollout_len, M=M, K=K)

    print(f"\nDone. Figures saved to '{FIGURES_DIR}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figs 5, 7, 8, 9, 11 from real DRL training runs"
    )
    parser.add_argument("--iterations", type=int, default=50,
                        help="Training iterations (outer PPO loops)")
    parser.add_argument("--rollout-len", type=int, default=64,
                        help="Rollout steps per iteration")
    parser.add_argument("--M", type=int, default=5,
                        help="Number of mobile devices")
    parser.add_argument("--K", type=int, default=8,
                        help="Number of service types")
    parser.add_argument("--figures", type=str, default="",
                        help="Comma-separated figure numbers to generate "
                             "(e.g. 5,7,8). Default: all supported figures.")
    args = parser.parse_args()

    figs_to_run = None
    if args.figures.strip():
        figs_to_run = [int(x.strip()) for x in args.figures.split(",")
                       if x.strip().isdigit()]

    generate_figures(
        iterations=args.iterations,
        rollout_len=args.rollout_len,
        M=args.M,
        K=args.K,
        selected_figs=figs_to_run
    )
