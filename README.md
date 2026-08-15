# Joint Computation Offloading and Service Caching via DRL (Reproduction)

Python reproduction of:

> C. Shang, Y. Huang, Y. Sun, and M. Guizani, "Joint Computation Offloading
> and Service Caching in Mobile Edge–Cloud Computing via Deep Reinforcement
> Learning," *IEEE Internet of Things Journal*, vol. 11, no. 24,
> pp. 40331–40344, 15 Dec. 2024.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Proposed approach (PPO + dynamic action masking), small smoke-test run
python train.py

# Full CLI: train + evaluate against all baselines + save plots
python main.py --iterations 3500 --rollout-len 200 --M 5 --K 8
```

## File map

| File               | Paper section(s)     | Contents                                                                 |
|--------------------|-----------------------|---------------------------------------------------------------------------|
| `config.py`        | Sec. V-A / Table I, II | System & PPO hyperparameters                                              |
| `system_model.py`  | Sec. III               | eq. (3)-(13), (28)-(29): task generation, channel model, cost models      |
| `environment.py`   | Sec. IV-A               | MDP: state (15), action (16), reward (17), constraints C1-C5 (14)        |
| `networks.py`      | Sec. IV-C / Fig. 2      | Actor/critic nets, masked-categorical distributions (eq. 22)              |
| `ppo_agent.py`     | Sec. IV-B/C             | GAE (21,24), ratio (23), clipped objective (25,26), critic loss (27)      |
| `train.py`         | Algorithm 1             | Full training loop, line-by-line                                          |
| `baselines.py`     | Sec. V-A                | Local, Cloud, MP, LRU, Random schemes                                     |
| `ddpg_agent.py`    | Sec. V-A item 4         | DDPG baseline (continuous relaxation of the discrete action space)        |
| `evaluate.py`      | Sec. V-B/C              | Convergence & average-cost comparison plots (Figs. 3-12 style)            |
| `main.py`          | -                        | CLI entry point                                                            |

## Important implementation notes / assumptions

The PDF's **Table II** (exact experimental parameter values) is embedded as
an **image**, so its numeric contents could not be extracted from the text
supplied. Every parameter in `config.py` is annotated as either:

- **(explicit)** — value stated in the paper's running text (e.g. cell
  radius 100 m, service storage 50 Mb, MEC storage 100 Mb, Zipf α = 0.8,
  fc = 915 MHz, antenna gain A = 4.11, path-loss exponent E = 3, Rician LOS
  ratio 0.6, uplink/downlink correlation 0.7, cloud rate 20 Mb/s,
  λ_t = λ_e = 0.5, hidden layers = 256/256, actor/critic lr = 1e-4,
  clip ε = 0.1, γ = 0.9, GAE λ = 0.95, Tanh activation), or
- **(ASSUMED)** — a reasonable default filled in because the paper does not
  give a numeric value in the extractable text (e.g. transmit/idle/receive
  power, local/edge CPU frequencies, energy coefficient κ, task
  input/output-size and CPU-cycle ranges, noise power, entropy coefficient
  β, number of PPO epochs per update, minibatch size, number of training
  iterations/episodes).

If you have access to the original Table II image, simply edit the
corresponding fields in `config.py` — every other module consumes these
values symbolically and needs no further changes.

### Dynamic action masking (eq. 22)

The joint action `a_t = {c(t), d(t)}` is generated via **action
composition**: K independent binary "cache/not-cache" categorical
subactions and M independent ternary "local/edge/cloud" categorical
subactions, exactly as described in Sec. IV-C. Because the storage
constraint (C2) is a *joint* constraint across the K caching subactions,
caching decisions are sampled **sequentially** with a running storage
budget, masking `cache=1` for any service whose storage requirement would
exceed the remaining budget. Once `c(t)` is fixed, the M offloading
subactions are sampled with `edge` masked out whenever the requested
service's caching decision is 0, directly implementing constraint C5. All
masking uses the paper's exact masked-logit formula
`z'_i = z_i + (1 - m_i) * C` (eq. 22), applied per subaction.

### Reward / cost

`r_t = - Σ_m [λ_t T_m(t) + λ_e E_m(t)]` (eq. 17) is computed exactly, and
`Z_m(t) = λ_t T_m(t) + λ_e E_m(t)` (eq. 13) is tracked separately per step
so that `evaluate.py` can report the *average cost per device* metric used
throughout the paper's Figs. 6-12.

### PPO-based (no-mask, penalty) ablation

`train.py::select_action_no_mask` reproduces the "PPO-based caching and
offloading" baseline of Figs. 4-5: the *offloading* action is sampled from
unmasked logits, and a configurable penalty (`--penalty`, default 50, as
used in the paper's final experiments) is subtracted from the reward
whenever an invalid edge-offload is sampled; the executed action is then
corrected to `local` so the environment's hard physical constraints still
hold. Run it with:

```bash
python main.py --no-mask --penalty 50
```
