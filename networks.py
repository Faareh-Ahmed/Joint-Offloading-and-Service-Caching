"""
networks.py
===========
Actor and Critic networks (Sec. IV-C / Fig. 2 of the paper).

  - Actor network: fully connected NN with two hidden layers (256 neurons
    each, explicit), Tanh activation (explicit), taking the state s_t as
    input and outputting the probability distribution of the composed
    action (service caching + device offloading subactions).
  - Critic network: fully connected NN approximating V(s_t), same hidden
    architecture.

Action composition (Sec. IV-C, citing Huang et al. [30] "action
composition"): the joint action space is reorganized as a Cartesian
product of independent per-subaction categorical distributions:
    - K caching subactions, each Categorical({not-cache, cache})
    - M offloading subactions, each Categorical({local, edge, cloud})

Dynamic masking (eq. 22):
    z'_i(t) = z_i(t) + (1 - m_i(t)) * C
where C is an extremely large negative number, so that Softmax(z') assigns
near-zero probability to invalid actions i. This is implemented in
`masked_categorical` below.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF_CONST = -1.0e9  # "C" in eq. (22)


def _activation(name: str):
    if name.lower() == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class ActorNetwork(nn.Module):
    """
    Outputs raw (unmasked) logits for every subaction:
        cache_logits    : (batch, K, 2)
        offload_logits  : (batch, M, 3)
    Masking (eq. 22) is applied outside this module at sampling time,
    since the caching mask depends on the remaining storage budget and the
    offloading mask depends on the just-sampled caching decision.
    """

    def __init__(self, state_dim: int, M: int, K: int,
                 hidden1: int = 256, hidden2: int = 256, activation: str = "tanh"):
        super().__init__()
        self.M, self.K = M, K
        Act = _activation(activation)
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden1), Act(),
            nn.Linear(hidden1, hidden2), Act(),
        )
        self.cache_out = nn.Linear(hidden2, K * 2)
        self.offload_out = nn.Linear(hidden2, M * 3)

        self._orthogonal_init()

    def _orthogonal_init(self):
        """Orthogonal weights and constant biases (Algorithm 1, init step)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor):
        feat = self.body(state)
        cache_logits = self.cache_out(feat).view(-1, self.K, 2)
        offload_logits = self.offload_out(feat).view(-1, self.M, 3)
        return cache_logits, offload_logits


class CriticNetwork(nn.Module):
    """Approximates V(s_t) -- eq. (18) / eq. (27) target."""

    def __init__(self, state_dim: int, hidden1: int = 256, hidden2: int = 256,
                 activation: str = "tanh"):
        super().__init__()
        Act = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1), Act(),
            nn.Linear(hidden1, hidden2), Act(),
            nn.Linear(hidden2, 1),
        )
        self._orthogonal_init()

    def _orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """
    Dynamic invalid-action masking -- eq. (22):
        z'_i = z_i + (1 - m_i) * C
    `mask` has the same shape as `logits`, with 1 = valid, 0 = invalid.
    """
    masked_logits = logits + (1.0 - mask) * NEG_INF_CONST
    return Categorical(logits=masked_logits)


class ActorCritic(nn.Module):
    """Convenience wrapper bundling the actor and critic networks."""

    def __init__(self, state_dim: int, M: int, K: int,
                 hidden1: int = 256, hidden2: int = 256, activation: str = "tanh"):
        super().__init__()
        self.actor = ActorNetwork(state_dim, M, K, hidden1, hidden2, activation)
        self.critic = CriticNetwork(state_dim, hidden1, hidden2, activation)

    def forward(self, state: torch.Tensor):
        cache_logits, offload_logits = self.actor(state)
        value = self.critic(state)
        return cache_logits, offload_logits, value
