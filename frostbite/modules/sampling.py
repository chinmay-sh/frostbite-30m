"""Gumbel-Softmax straight-through sampling for discrete routing actions."""

from __future__ import annotations

from torch import Tensor
from torch.nn.functional import gumbel_softmax


def gumbel_sample(logits: Tensor, tau: float, hard: bool = True) -> Tensor:
    """Sample one-hot actions with gradients flowing to the logits.

    Forward pass: hard one-hot (a genuine discrete choice).
    Backward pass: straight-through estimator uses the soft probabilities,
    so `logits` receive gradients as if the choice were soft.

    :param logits: action logits (B, T, n_actions) or any (..., n_actions).
    :param tau: temperature; high = exploration, low = near-argmax.
    :param hard: return one-hot in the forward pass (straight-through).
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
