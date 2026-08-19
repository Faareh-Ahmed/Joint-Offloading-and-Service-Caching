"""
main.py
=======
Command-line entry point for training and evaluation.

Examples
--------
Train the proposed approach (dynamic-masking PPO) and compare it against
all heuristic baselines:

    python main.py --iterations 3500 --rollout-len 200

Train the penalty-based PPO ablation baseline:

    python main.py --no-mask --penalty 50 --iterations 3500

Generate all supported figures (Figs 5, 7, 8, 9, 11):

    python main.py --all-figures --iterations 500 --rollout-len 128
"""

from __future__ import annotations
import argparse
import json

from config import SystemConfig, PPOConfig
from train import train
from evaluate import compare_all, plot_convergence, plot_cost_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Joint Computation Offloading and Service Caching via DRL"
    )
    parser.add_argument("--M", type=int, default=5,
                        help="number of mobile devices")
    parser.add_argument("--K", type=int, default=10,
                        help="number of service types")
    parser.add_argument("--iterations", type=int, default=200,
                        help="PPO outer iterations (L)")
    parser.add_argument("--rollout-len", type=int, default=128,
                        help="steps per iteration / episode")
    parser.add_argument("--no-mask", action="store_true",
                        help="disable dynamic masking and use penalty-based baseline instead")
    parser.add_argument("--penalty", type=float, default=50.0,
                        help="invalid-action penalty term (used only with --no-mask)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", type=str, default="results")
    parser.add_argument("--all-figures", action="store_true",
                        help="generate Figs 5, 7, 8, 9, 11 from real training runs")
    args = parser.parse_args()

    if args.all_figures:
        from generate_all_figures import generate_figures
        generate_figures(
            iterations=args.iterations,
            rollout_len=args.rollout_len,
            M=args.M,
            K=args.K
        )
        return

    sys_cfg = SystemConfig(M=args.M, K=args.K)
    ppo_cfg = PPOConfig(total_iterations=args.iterations, rollout_len=args.rollout_len)

    agent, history = train(
        sys_cfg, ppo_cfg,
        use_mask=not args.no_mask,
        invalid_penalty=args.penalty,
        seed=args.seed,
        log_every=max(1, args.iterations // 20)
    )

    plot_convergence(history, save_path=f"{args.out_prefix}_convergence.png")

    results = compare_all(agent, sys_cfg, n_episodes=args.eval_episodes)
    plot_cost_comparison(results, save_path=f"{args.out_prefix}_cost_comparison.png")

    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump({"history": history, "final_costs": results}, f, indent=2)

    print("\n=== Average cost per device (lower is better) ===")
    for name, cost in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:40s} {cost:8.4f}")


if __name__ == "__main__":
    main()
