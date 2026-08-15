"""
system_model.py
================
Implements the system model of Section III of the paper:

  - Zipf task-popularity distribution                       (eq. 28)
  - Free-space path-loss average channel gain                (eq. 29)
  - Correlated Rician small-scale fading for uplink/downlink
  - Shannon-Hartley uplink / downlink rates                  (eq. 5, 6)
  - Local / edge / cloud computing time and energy models    (eq. 3, 4, 7-13)

All equation numbers below refer to the equations of the paper.
"""

from __future__ import annotations
import numpy as np

SPEED_OF_LIGHT = 3e8  # m/s


# --------------------------------------------------------------------------- #
# Zipf popularity distribution (eq. 28)
# --------------------------------------------------------------------------- #
def zipf_probabilities(K: int, alpha: float) -> np.ndarray:
    """
    q_k = (1/k^alpha) / sum_{k=1}^{K} (1/k^alpha)   -- eq. (28)
    Returns an array of length K with q_k for k = 1..K.
    """
    ranks = np.arange(1, K + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


# --------------------------------------------------------------------------- #
# Wireless channel model
# --------------------------------------------------------------------------- #
def avg_channel_gain(distance_m: np.ndarray, fc_hz: float, A: float, E: float) -> np.ndarray:
    """
    Free-space path loss average channel gain -- eq. (29):
        h_bar = A * (3e8 / (4*pi*fc*d))^E
    """
    distance_m = np.maximum(distance_m, 1.0)  # avoid singularities at d=0
    return A * np.power(SPEED_OF_LIGHT / (4.0 * np.pi * fc_hz * distance_m), E)


def correlated_rician_gains(h_bar: np.ndarray, los_ratio: float, rho: float,
                             rng: np.random.Generator):
    """
    Draw one realization of correlated uplink/downlink Rician power gains
    for each device.

    The paper states (Sec. V-A):
      - the time-varying fading channel follows an i.i.d. Rician distribution,
      - the line-of-sight (LOS) link power is 0.6 * h_bar,
      - the uplink and downlink channels are correlated with coefficient 0.7.

    Implementation: a Rician fading amplitude is the magnitude of a complex
    Gaussian random variable with a non-zero mean (LOS component). The
    Rician K-factor implied by "LOS power = los_ratio * h_bar" (with total
    average power h_bar) is K = los_ratio / (1 - los_ratio). Uplink and
    downlink share a common random scattered component (weighted by rho)
    to induce the target correlation, and each additionally has an
    independent scattered component (weighted by sqrt(1-rho)).

    Returns
    -------
    h_u, h_d : np.ndarray
        Instantaneous uplink and downlink power gains for each device.
    """
    K_factor = los_ratio / max(1e-6, (1.0 - los_ratio))
    s = np.sqrt(K_factor / (K_factor + 1.0))              # LOS amplitude scale
    sigma = np.sqrt(1.0 / (2.0 * (K_factor + 1.0)))        # scattered component std / dim

    n = len(h_bar)
    common_i = rng.normal(0.0, sigma, size=n)
    common_q = rng.normal(0.0, sigma, size=n)
    indep_u_i = rng.normal(0.0, sigma, size=n)
    indep_u_q = rng.normal(0.0, sigma, size=n)
    indep_d_i = rng.normal(0.0, sigma, size=n)
    indep_d_q = rng.normal(0.0, sigma, size=n)

    a = np.sqrt(rho)
    b = np.sqrt(max(0.0, 1.0 - rho))

    u_i = s + a * common_i + b * indep_u_i
    u_q = a * common_q + b * indep_u_q
    d_i = s + a * common_i + b * indep_d_i
    d_q = a * common_q + b * indep_d_q

    h_u = h_bar * (u_i ** 2 + u_q ** 2)
    h_d = h_bar * (d_i ** 2 + d_q ** 2)
    return h_u, h_d


def shannon_rate_mbps(bandwidth_hz: float, tx_power_w: float,
                       channel_gain: np.ndarray, noise_power_w: float) -> np.ndarray:
    """
    Maximum uplink/downlink transfer rate -- eq. (5) / (6):
        R = W * log2(1 + p * h / sigma^2)
    Returned in Mb/s (divide bits/s by 1e6) so that it is dimensionally
    consistent with data sizes expressed in Mb.
    """
    snr = (tx_power_w * channel_gain) / noise_power_w
    rate_bps = bandwidth_hz * np.log2(1.0 + snr)
    return rate_bps / 1e6


# --------------------------------------------------------------------------- #
# Task generation
# --------------------------------------------------------------------------- #
class TaskGenerator:
    """Generates heterogeneous per-device tasks (i_m(t), phi_m(t), tau_m(t), o_m(t))."""

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.zipf_probs = zipf_probabilities(cfg.K, cfg.zipf_alpha)

    def sample(self):
        cfg = self.cfg
        M = cfg.M
        # tau_m(t): service type requested, drawn i.i.d. from the Zipf distribution (eq. 28)
        tau = self.rng.choice(cfg.K, size=M, p=self.zipf_probs)

        i_lo, i_hi = cfg.input_size_range_mb
        o_lo, o_hi = cfg.output_size_range_mb
        c_lo, c_hi = cfg.cpu_cycles_range

        i_size = self.rng.uniform(i_lo, i_hi, size=M) * cfg.task_scale_factor
        o_size = self.rng.uniform(o_lo, o_hi, size=M) * cfg.task_scale_factor
        phi = self.rng.uniform(c_lo, c_hi, size=M) * cfg.cpu_scale_factor

        return i_size, phi, tau, o_size


# --------------------------------------------------------------------------- #
# Local / Edge / Cloud computing time & energy models (eq. 3, 4, 7-13)
# --------------------------------------------------------------------------- #
def local_computing(phi, cfg):
    """
    eq. (3): T_l = phi / f_local
    eq. (4): E_l = kappa * f_local^2 * phi
    """
    T_l = phi / cfg.f_local_hz
    E_l = cfg.kappa_energy_coeff * (cfg.f_local_hz ** 2) * phi
    return T_l, E_l


def edge_computing(i_size, phi, o_size, Ru, Rd, n_edge_devices, cfg):
    """
    eq. (7): T_e = i/Ru + phi / (f_edge / sum_m d_e_m) + o/Rd
    eq. (8): E_e = p_tr * i/Ru + p_id * phi / (f_edge / sum_m d_e_m) + p_re * o/Rd

    n_edge_devices : number of devices concurrently offloading to the edge
                     at this time slot (sum_{m=1}^{M} d^e_m(t)); the MEC
                     server equally shares its computing resource among them.
    """
    n_edge_devices = max(1, n_edge_devices)
    shared_f_edge = cfg.f_edge_hz / n_edge_devices

    T_e = i_size / Ru + phi / shared_f_edge + o_size / Rd
    E_e = (cfg.p_tr_w * (i_size / Ru)
           + cfg.p_idle_w * (phi / shared_f_edge)
           + cfg.p_receive_w * (o_size / Rd))
    return T_e, E_e


def cloud_computing(i_size, o_size, Ru, Rd, cfg):
    """
    eq. (9):  T_c = i/Ru + (i+o)/Rc + o/Rd
    eq. (10): E_c = p_tr * i/Ru + p_id * (i+o)/Rc + p_re * o/Rd
    """
    Rc = cfg.cloud_rate_mbps
    T_c = i_size / Ru + (i_size + o_size) / Rc + o_size / Rd
    E_c = (cfg.p_tr_w * (i_size / Ru)
           + cfg.p_idle_w * ((i_size + o_size) / Rc)
           + cfg.p_receive_w * (o_size / Rd))
    return T_c, E_c


def task_cost(T_m, E_m, cfg):
    """eq. (13): Z_m(t) = lambda_t * T_m(t) + lambda_e * E_m(t)"""
    return cfg.lambda_t * T_m + cfg.lambda_e * E_m
