"""
environment.py
===============
Implements the end-edge-cloud MEC environment as the MDP described in
Section IV-A of the paper:

    state  s_t = {i(t), phi(t), tau(t), o(t), h(t)}                 (eq. 15)
    action a_t = {c(t), d_l(t), d_e(t), d_c(t)}                     (eq. 16)
    reward r_t = - sum_m [lambda_t*T_m(t) + lambda_e*E_m(t)]        (eq. 17)

Constraints C1-C5 (eq. 14):
    C1: c_k(t) in {0,1}
    C2: sum_k c_k(t) u_k <= U                    (storage capacity)
    C3: d^l_m, d^e_m, d^c_m in {0,1}
    C4: d^l_m + d^e_m + d^c_m = 1                (single execution mode)
    C5: d^e_m(t) <= c_{tau_m(t)}(t)               (service must be cached
                                                    to offload to the edge)

Dynamic action masking (eq. 22) is exposed through `get_cache_mask` and
`get_offload_mask`, which are consumed by the PPO agent / networks when
building masked categorical distributions.
"""

from __future__ import annotations
import numpy as np

from system_model import (
    TaskGenerator, avg_channel_gain, correlated_rician_gains,
    shannon_rate_mbps, local_computing, edge_computing, cloud_computing,
    task_cost,
)

LOCAL, EDGE, CLOUD = 0, 1, 2  # offloading-mode encoding, d_m(t) in {0,1,2}
NOT_CACHE, CACHE = 0, 1


class MECEnvironment:
    """
    Gym-like environment for the joint computation-offloading and
    service-caching MDP.
    """

    def __init__(self, cfg, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.task_gen = TaskGenerator(cfg, self.rng)

        # Fixed random device positions within the cell radius (explicit, Sec. V-A)
        self.device_distance = self.rng.uniform(5.0, cfg.cell_radius, size=cfg.M)

        self.t = 0
        self._state = None

    # ------------------------------------------------------------------ #
    # State-space dimensionality:  6*M   (eq. state has i, phi, tau, o -> 4M
    #                                      plus h^u, h^d -> 2M)
    # Action-space dimensionality: 3*M + 2*K (M offloading categoricals of
    #                                      size 3 + K caching categoricals
    #                                      of size 2), matching the paper's
    #                                      stated actor-network output dim.
    # ------------------------------------------------------------------ #
    @property
    def state_dim(self) -> int:
        return 6 * self.cfg.M

    def reset(self):
        self.t = 0
        self._state = self._sample_state()
        return self._state

    # ------------------------------------------------------------------ #
    # State generation
    # ------------------------------------------------------------------ #
    def _sample_state(self):
        cfg = self.cfg
        i_size, phi, tau, o_size = self.task_gen.sample()

        h_bar = avg_channel_gain(self.device_distance, cfg.carrier_freq_hz(),
                                  cfg.antenna_gain_A, cfg.path_loss_exponent_E)
        h_u, h_d = correlated_rician_gains(h_bar, cfg.rician_los_ratio,
                                            cfg.channel_correlation, self.rng)

        return {
            "i": i_size, "phi": phi, "tau": tau, "o": o_size,
            "h_u": h_u, "h_d": h_d,
        }

    def state_vector(self, state=None) -> np.ndarray:
        """Flatten the state dict into the 6M-dimensional vector of eq. (15)."""
        s = state if state is not None else self._state
        return np.concatenate([
            s["i"], s["phi"], s["tau"].astype(np.float64), s["o"], s["h_u"], s["h_d"],
        ]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Dynamic masking helpers (eq. 22)
    # ------------------------------------------------------------------ #
    def get_cache_mask(self, remaining_budget_mb: float) -> np.ndarray:
        """
        Mask for a single service's caching decision {not-cache, cache}.
        "cache" (index 1) is invalid if its storage requirement exceeds the
        remaining MEC storage budget (constraint C2).
        """
        mask = np.array([1.0, 1.0], dtype=np.float32)
        if self.cfg.service_storage_mb > remaining_budget_mb:
            mask[CACHE] = 0.0
        return mask

    def get_offload_mask(self, service_type: int, cache_decision: np.ndarray) -> np.ndarray:
        """
        Mask for a single device's offloading decision {local, edge, cloud}.
        "edge" (index 1) is invalid unless the requested service is cached
        on the MEC server (constraint C5): d^e_m(t) <= c_{tau_m(t)}(t).
        """
        mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if cache_decision[service_type] == NOT_CACHE:
            mask[EDGE] = 0.0
        return mask

    # ------------------------------------------------------------------ #
    # Environment transition
    # ------------------------------------------------------------------ #
    def step(self, cache_decision: np.ndarray, offload_decision: np.ndarray):
        """
        Executes one joint action a_t = {c(t), d_l(t), d_e(t), d_c(t)} and
        returns (next_state, reward, done, info).

        cache_decision    : (K,) binary array, c_k(t)
        offload_decision  : (M,) array in {0,1,2}, d_m(t)
        """
        cfg = self.cfg
        s = self._state
        i_size, phi, tau, o_size = s["i"], s["phi"], s["tau"], s["o"]
        h_u, h_d = s["h_u"], s["h_d"]

        # Enforce storage constraint C2 defensively (should already hold if
        # the mask in get_cache_mask() was respected during sampling).
        used = cache_decision.astype(np.float64).sum() * cfg.service_storage_mb
        assert used <= cfg.mec_storage_capacity_mb + 1e-6, "C2 storage constraint violated"

        # Enforce constraint C5 defensively.
        edge_mask_ok = np.array([cache_decision[tau[m]] for m in range(cfg.M)])
        assert np.all((offload_decision != EDGE) | (edge_mask_ok == CACHE)), \
            "C5 offload-to-uncached-service constraint violated"

        Ru = shannon_rate_mbps(cfg.bandwidth_hz(), cfg.p_tr_w, h_u, cfg.noise_power_w)
        Rd = shannon_rate_mbps(cfg.bandwidth_hz(), cfg.p_bs_w, h_d, cfg.noise_power_w)

        T_l, E_l = local_computing(phi, cfg)
        n_edge = int(np.sum(offload_decision == EDGE))
        T_e, E_e = edge_computing(i_size, phi, o_size, Ru, Rd, n_edge, cfg)
        T_c, E_c = cloud_computing(i_size, o_size, Ru, Rd, cfg)

        # eq. (11), (12): select the branch matching each device's decision
        T_m = np.where(offload_decision == LOCAL, T_l,
              np.where(offload_decision == EDGE, T_e, T_c))
        E_m = np.where(offload_decision == LOCAL, E_l,
              np.where(offload_decision == EDGE, E_e, E_c))

        Z_m = task_cost(T_m, E_m, cfg)          # eq. (13), per-device cost
        reward = -float(np.sum(Z_m))            # eq. (17)

        info = {
            "T_m": T_m, "E_m": E_m, "Z_m": Z_m,
            "avg_cost": float(np.mean(Z_m)),
            "n_edge": n_edge,
        }

        self.t += 1
        done = self.t >= cfg.T
        next_state = self._sample_state() if not done else None
        self._state = next_state
        return next_state, reward, done, info
