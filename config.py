"""
config.py
=========
Configuration parameters for:

Shang, C., Huang, Y., Sun, Y., & Guizani, M. (2024).
"Joint Computation Offloading and Service Caching in Mobile Edge-Cloud
Computing via Deep Reinforcement Learning."
IEEE Internet of Things Journal, 11(24), 40331-40344.

Every parameter below is annotated with (explicit) if its numeric value is
stated in the paper text, or (ASSUMED) if the paper only describes it
qualitatively / gives it in Table II, which is embedded as an image in the
supplied PDF and therefore could not be extracted verbatim. Where a value
is ASSUMED, a typical order-of-magnitude value used in the MEC literature
referenced by the paper is used instead. Change these freely to match the
authors' exact Table II if you obtain it.
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    # ---------------- Network topology ----------------
    M: int = 5                    # Table II: M = 5
    K: int = 10                   # Table II: K = 10
    cell_radius: float = 100.0    # m; devices randomly distributed within 100 m radius (explicit, Sec. V-A)

    # ---------------- Episode length ----------------
    T: int = 128                  # Table II: T = 128 time slots

    # ---------------- Service caching (eq. 1) ----------------
    service_storage_mb: float = 50.0        # u_k for every service k, Mb (explicit, Sec. V-A)
    mec_storage_capacity_mb: float = 100.0  # U, MEC server storage, Mb (explicit, Sec. V-A; swept in Fig. 11)

    # ---------------- Zipf task-popularity distribution (eq. 28) ----------------
    zipf_alpha: float = 0.8       # skewness factor alpha (explicit)

    # ---------------- Wireless channel model (eq. 29) ----------------
    bandwidth_mhz: float = 2.0        # Table II: W = 2 MHz
    carrier_freq_mhz: float = 915.0   # f_c (explicit)
    antenna_gain_A: float = 4.11      # A (explicit)
    path_loss_exponent_E: float = 3.0  # E (explicit)
    rician_los_ratio: float = 0.6     # LOS power = 0.6 * h_bar (explicit)
    channel_correlation: float = 0.7  # correlation between uplink/downlink fading (explicit)
    noise_power_w: float = 1e-13      # Table II: sigma^2 = 10^-13 W

    # ---------------- Transmission / device powers (Table II) ----------------
    p_tr_w: float = 0.5      # Table II: p_tr = 0.5 W
    p_bs_w: float = 5.0      # Table II: p_bs = 5 W
    p_idle_w: float = 0.01   # Table II: p_id = 0.01 W
    p_receive_w: float = 0.1  # Table II: p_re = 0.1 W

    # ---------------- Computing capacities (Table II) ----------------
    f_local_hz: float = 1.0e9     # Table II: f_local = 1 GHz
    f_edge_hz: float = 10.0e9     # Table II: f_edge = 10 GHz
    kappa_energy_coeff: float = 1e-27  # Table II: kappa = 10^-27

    # ---------------- Cloud backhaul (explicit) ----------------
    cloud_rate_mbps: float = 20.0   # R_c, MEC<->cloud link rate, Mb/s (explicit, Sec. V-A)

    # ---------------- Task characteristics (Table II) ----------------
    input_size_range_mb: tuple = (1.0, 5.0)     # Table II: i(t) = 1 ~ 5 MB
    output_size_range_mb: tuple = (1.0, 5.0)    # Table II: o(t) = 1 ~ 5 MB
    cpu_cycles_range: tuple = (1.0e9, 5.0e9)    # Table II: p(t) = 1 ~ 5 G cycles

    # ---------------- Cost function weights (eq. 13) ----------------
    lambda_t: float = 0.5   # time weight (explicit default, Sec. V-A: lambda_t = lambda_e = 0.5)
    lambda_e: float = 0.5   # energy weight (explicit default)

    # ---------------- Scaling factors used to reproduce Figs. 6-7 ----------------
    task_scale_factor: float = 1.0   # scales i_m(t), o_m(t) -- Fig. 6 sweeps 0.7-1.2
    cpu_scale_factor: float = 1.0    # scales phi_m(t) -- Fig. 7 sweeps 0.7-1.2

    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6

    def carrier_freq_hz(self) -> float:
        return self.carrier_freq_mhz * 1e6

    def cloud_rate_mbps_value(self) -> float:
        return self.cloud_rate_mbps


@dataclass
class PPOConfig:
    """Hyperparameters explicitly given in Sec. V-A unless marked ASSUMED."""
    hidden1: int = 256            # first hidden layer size (explicit)
    hidden2: int = 256            # second hidden layer size (explicit)
    actor_lr: float = 1e-4        # explicit (both actor & critic use 1e-4)
    critic_lr: float = 1e-4       # explicit
    clip_eps: float = 0.1         # epsilon, clipping coefficient (explicit)
    gamma: float = 0.9            # discount factor (explicit)
    gae_lambda: float = 0.95      # GAE lambda (explicit)
    entropy_coef: float = 0.01    # beta, entropy weight (ASSUMED; value not numerically given)
    epochs_per_update: int = 10   # K inner epochs in Algorithm 1 (ASSUMED)
    minibatch_size: int = 64      # ASSUMED
    rollout_len: int = 200        # N, number of steps collected per iteration (ASSUMED = T)
    total_iterations: int = 1000  # L, number of outer iterations (ASSUMED; paper trains to ~3500 episodes)
    activation: str = "tanh"      # Tanh activation function used throughout (explicit)
    max_grad_norm: float = 0.5    # gradient clipping (ASSUMED, standard PPO practice)
