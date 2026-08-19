"""
config.py
=========
Configuration parameters for the simulation of:

  Shang, C., Huang, Y., Sun, Y., & Guizani, M. (2024).
  "Joint Computation Offloading and Service Caching in Mobile Edge-Cloud
  Computing via Deep Reinforcement Learning."
  IEEE Internet of Things Journal, 11(24), 40331-40344.

Parameters annotated (explicit) are stated directly in the paper text;
those annotated (ASSUMED) use standard MEC literature values where the paper
does not provide a specific number.
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    # --- Network topology ---
    M: int = 5                    # Table II: M = 5 mobile devices
    K: int = 10                   # Table II: K = 10 service types
    cell_radius: float = 100.0    # m; cell radius (explicit, Sec. V-A)

    # --- Episode length ---
    T: int = 128                  # Table II: T = 128 time slots

    # --- Service caching ---
    service_storage_mb: float = 50.0        # u_k per service, Mb (explicit, Sec. V-A)
    mec_storage_capacity_mb: float = 100.0  # U, MEC storage budget, Mb (explicit, Sec. V-A)

    # --- Zipf task-popularity distribution (eq. 28) ---
    zipf_alpha: float = 0.8       # skewness factor alpha (explicit)

    # --- Wireless channel model (eq. 29) ---
    bandwidth_mhz: float = 2.0        # Table II: W = 2 MHz
    carrier_freq_mhz: float = 915.0   # f_c (explicit)
    antenna_gain_A: float = 4.11      # A (explicit)
    path_loss_exponent_E: float = 3.0  # E (explicit)
    rician_los_ratio: float = 0.6     # LOS power ratio (explicit)
    channel_correlation: float = 0.7  # uplink/downlink fading correlation (explicit)
    noise_power_w: float = 1e-13      # Table II: sigma^2 = 10^-13 W

    # --- Device power consumption (Table II) ---
    p_tr_w: float = 0.5      # transmit power p_tr = 0.5 W
    p_bs_w: float = 5.0      # base-station transmit power p_bs = 5 W
    p_idle_w: float = 0.01   # idle power p_id = 0.01 W
    p_receive_w: float = 0.1  # receive power p_re = 0.1 W

    # --- Computing capacities (Table II) ---
    f_local_hz: float = 1.0e9     # local CPU frequency = 1 GHz
    f_edge_hz: float = 10.0e9     # MEC CPU frequency = 10 GHz
    kappa_energy_coeff: float = 1e-27  # energy coefficient kappa = 10^-27

    # --- Cloud backhaul (explicit) ---
    cloud_rate_mbps: float = 20.0  # R_c, MEC<->cloud link rate = 20 Mb/s (explicit, Sec. V-A)

    # --- Task characteristics (Table II) ---
    input_size_range_mb: tuple = (1.0, 5.0)     # i(t) in [1, 5] MB
    output_size_range_mb: tuple = (1.0, 5.0)    # o(t) in [1, 5] MB
    cpu_cycles_range: tuple = (1.0e9, 5.0e9)    # phi(t) in [1, 5] G cycles

    # --- Cost function weights (eq. 13) ---
    lambda_t: float = 0.5   # time weight (explicit, Sec. V-A)
    lambda_e: float = 0.5   # energy weight (explicit, Sec. V-A)

    # --- Scaling factors for parameter sweeps (Figs 7 and 8) ---
    task_scale_factor: float = 1.0   # scales i_m(t), o_m(t)
    cpu_scale_factor: float = 1.0    # scales phi_m(t)

    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6

    def carrier_freq_hz(self) -> float:
        return self.carrier_freq_mhz * 1e6

    def cloud_rate_mbps_value(self) -> float:
        return self.cloud_rate_mbps


@dataclass
class PPOConfig:
    """PPO hyperparameters from Sec. V-A (explicit) or standard defaults (ASSUMED)."""
    hidden1: int = 256            # first hidden layer size (explicit)
    hidden2: int = 256            # second hidden layer size (explicit)
    actor_lr: float = 1e-4        # actor learning rate (explicit)
    critic_lr: float = 1e-4       # critic learning rate (explicit)
    clip_eps: float = 0.1         # PPO clipping coefficient (explicit)
    gamma: float = 0.9            # discount factor (explicit)
    gae_lambda: float = 0.95      # GAE lambda (explicit)
    entropy_coef: float = 0.01    # entropy regularization weight (ASSUMED)
    epochs_per_update: int = 10   # inner PPO epochs per update (ASSUMED)
    minibatch_size: int = 64      # minibatch size (ASSUMED)
    rollout_len: int = 200        # rollout steps per iteration (ASSUMED)
    total_iterations: int = 1000  # total outer iterations (ASSUMED)
    activation: str = "tanh"      # activation function (explicit)
    max_grad_norm: float = 0.5    # gradient clipping threshold (ASSUMED)
