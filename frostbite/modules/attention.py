"""Multi-head self-attention with pre-norm residual wrapper."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class SpatialAttention(nn.Module):
    """Multi-head self-attention over the sequence dimension."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P1.U2.
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # Implemented in P1.U2.
        raise NotImplementedError
