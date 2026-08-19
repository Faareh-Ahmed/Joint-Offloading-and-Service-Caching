# Joint Computation Offloading and Service Caching via Deep Reinforcement Learning

Implementation of the following research paper:

> C. Shang, Y. Huang, Y. Sun, and M. Guizani, "Joint Computation Offloading
> and Service Caching in Mobile Edge–Cloud Computing via Deep Reinforcement
> Learning," *IEEE Internet of Things Journal*, vol. 11, no. 24,
> pp. 40331–40344, Dec. 2024.

The code implements Algorithm of the paper: a PPO agent with **dynamic action masking** that jointly optimizes computation offloading decisions and service caching in a three-tier mobile edge-cloud system.

---

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `config.py` | System and PPO hyperparameters (Table I & II of the paper) |
| `system_model.py` | Physical cost models: channel model, task generation, time/energy computation (eqs. 3–13, 28–29) |
| `environment.py` | MDP definition: state (eq. 15), action (eq. 16), reward (eq. 17), constraints C1–C5 (eq. 14) |
| `networks.py` | Actor/critic neural networks with masked-categorical distributions (eq. 22) |
| `ppo_agent.py` | PPO update: GAE (eqs. 21, 24), clipped objective (eqs. 25–26), critic loss (eq. 27) |
| `train.py` | Training loop implementing Algorithm 1 |
| `baselines.py` | Heuristic comparison schemes: Local, Cloud, Most-Popular, LRU, Random |
| `ddpg_agent.py` | DDPG baseline agent |
| `evaluate.py` | Evaluation utilities and convergence/cost plotting |
| `generate_all_figures.py` | Generates Figs 5, 7, 8, 9, 11 from live DRL training |
| `main.py` | CLI entry point |

---

## Running the Code

### 1. Generate paper figures

The following figures from the paper are supported:

| Figure | Description |
|--------|-------------|
| Fig. 5 | Convergence comparison: Proposed PPO vs Penalty PPO vs DDPG |
| Fig. 7 | Impact of computational workload (CPU cycles) on average cost |
| Fig. 8 | Impact of channel bandwidth on average cost |
| Fig. 9 | Impact of MEC computing capacity on average cost |
| Fig. 11 | Impact of MEC storage capacity on average cost |

Generate all supported figures:

```bash
python generate_all_figures.py --iterations 500 --rollout-len 128 --M 5 --K 8
```

Output figures are saved to the `figures/` directory.


## Key Implementation Notes

### Dynamic action masking (eq. 22)

The joint action `a_t = {c(t), d(t)}` is generated via action composition:
- **K binary caching sub-actions** `c_k(t) ∈ {0, 1}` — whether to cache service *k*
- **M ternary offloading sub-actions** `d_m(t) ∈ {local, edge, cloud}` — execution mode for device *m*

Caching decisions are sampled sequentially with a running storage budget, masking `cache=1` for any service that would exceed the remaining capacity (constraint C2). Once `c(t)` is fixed, offloading decisions mask `edge` whenever the device's requested service is not cached (constraint C5).

All masking uses the formula `z'_i = z_i + (1 − m_i) · C` (eq. 22, where *C* is a large negative constant).

### Reward function

`r_t = −Σ_m [λ_t T_m(t) + λ_e E_m(t)]` (eq. 17)

where `T_m(t)` and `E_m(t)` are the completion time and energy consumption for device *m*, and `λ_t = λ_e = 0.5` by default.

