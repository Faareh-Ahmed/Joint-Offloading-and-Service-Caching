"""
baselines.py
============
Non-DRL benchmark schemes from Sec. V-A, used for comparison against the
proposed approach in Figs. 5-12:

  1) Local Computing            -- all tasks executed on local devices
  2) Cloud Computing             -- all tasks transferred to the cloud
  3) Most Popular (MP)           -- MEC caches the most popular services
                                     until its capacity is reached
  4) Least Recently Used (LRU)   -- MEC caches services of most recently
                                     requested tasks
  5) Random                      -- MEC randomly caches services / makes
                                     random offloading decisions among the
                                     currently-valid options

Each scheme exposes `act(env, state, task_types)` returning
(cache_action, offload_action) consistent with `MECEnvironment.step`.
"""

from __future__ import annotations
import numpy as np

from environment import LOCAL, EDGE, CLOUD, NOT_CACHE, CACHE


class BaselineScheme:
    name = "base"

    def reset(self, cfg):
        self.cfg = cfg

    def act(self, env, state, task_types):
        raise NotImplementedError


class LocalComputing(BaselineScheme):
    """All tasks are executed on local mobile devices."""
    name = "Local Computing"

    def act(self, env, state, task_types):
        cache_action = np.zeros(self.cfg.K, dtype=np.int64)  # nothing cached, irrelevant
        offload_action = np.full(self.cfg.M, LOCAL, dtype=np.int64)
        return cache_action, offload_action


class CloudComputing(BaselineScheme):
    """All tasks are transferred to the cloud server for execution."""
    name = "Cloud Computing"

    def act(self, env, state, task_types):
        cache_action = np.zeros(self.cfg.K, dtype=np.int64)
        offload_action = np.full(self.cfg.M, CLOUD, dtype=np.int64)
        return cache_action, offload_action


class MostPopularCaching(BaselineScheme):
    """
    The MEC server caches the Most-Popular requested services until reaching
    its storage capacity. If the requested service is cached, the task is
    offloaded to the edge; otherwise it is executed locally.
    """
    name = "MP Caching and Offloading"

    def reset(self, cfg):
        super().reset(cfg)
        # popularity ranking from the Zipf distribution (static, known a priori)
        from system_model import zipf_probabilities
        probs = zipf_probabilities(cfg.K, cfg.zipf_alpha)
        ranked = np.argsort(-probs)  # descending popularity
        cache_action = np.zeros(cfg.K, dtype=np.int64)
        budget = cfg.mec_storage_capacity_mb
        for k in ranked:
            if cfg.service_storage_mb <= budget:
                cache_action[k] = CACHE
                budget -= cfg.service_storage_mb
        self._cache_action = cache_action

    def act(self, env, state, task_types):
        cache_action = self._cache_action
        offload_action = np.array([
            EDGE if cache_action[task_types[m]] == CACHE else LOCAL
            for m in range(self.cfg.M)
        ], dtype=np.int64)
        return cache_action, offload_action


class LRUCaching(BaselineScheme):
    """
    The MEC server caches services based on a Least-Recently-Used policy:
    the most recently requested services are kept cached, subject to the
    storage-capacity constraint. When the relevant service is cached, the
    task is offloaded to the edge.
    """
    name = "LRU Caching and Offloading"

    def reset(self, cfg):
        super().reset(cfg)
        self._recency = {}  # service_id -> last-used timestamp
        self._clock = 0
        self._cached = set()

    def _evict_if_needed(self, needed_slots):
        used_slots = len(self._cached)
        capacity_slots = int(self.cfg.mec_storage_capacity_mb // self.cfg.service_storage_mb)
        while used_slots + needed_slots > capacity_slots and self._cached:
            # evict least-recently-used
            lru_service = min(self._cached, key=lambda k: self._recency.get(k, -1))
            self._cached.discard(lru_service)
            used_slots -= 1

    def act(self, env, state, task_types):
        cfg = self.cfg
        self._clock += 1
        requested = set(int(t) for t in task_types)

        to_add = [k for k in requested if k not in self._cached]
        if to_add:
            self._evict_if_needed(len(to_add))
            capacity_slots = int(cfg.mec_storage_capacity_mb // cfg.service_storage_mb)
            for k in to_add:
                if len(self._cached) < capacity_slots:
                    self._cached.add(k)

        for k in requested:
            self._recency[k] = self._clock

        cache_action = np.zeros(cfg.K, dtype=np.int64)
        for k in self._cached:
            cache_action[k] = CACHE

        offload_action = np.array([
            EDGE if cache_action[task_types[m]] == CACHE else LOCAL
            for m in range(cfg.M)
        ], dtype=np.int64)
        return cache_action, offload_action


class RandomCaching(BaselineScheme):
    """
    The MEC server randomly caches as many services as possible (subject to
    C2), and makes random (but constraint-respecting) offloading decisions
    based on the resulting cache state.
    """
    name = "Random Caching and Offloading"

    def reset(self, cfg):
        super().reset(cfg)
        self.rng = np.random.default_rng()

    def act(self, env, state, task_types):
        cfg = self.cfg
        order = self.rng.permutation(cfg.K)
        cache_action = np.zeros(cfg.K, dtype=np.int64)
        budget = cfg.mec_storage_capacity_mb
        for k in order:
            if self.rng.random() < 0.5 and cfg.service_storage_mb <= budget:
                cache_action[k] = CACHE
                budget -= cfg.service_storage_mb

        offload_action = np.zeros(cfg.M, dtype=np.int64)
        for m in range(cfg.M):
            valid = [LOCAL, CLOUD]
            if cache_action[task_types[m]] == CACHE:
                valid.append(EDGE)
            offload_action[m] = self.rng.choice(valid)
        return cache_action, offload_action


ALL_BASELINES = {
    "local": LocalComputing,
    "cloud": CloudComputing,
    "mp": MostPopularCaching,
    "lru": LRUCaching,
    "random": RandomCaching,
}
