"""Temporal embedding engine: sensor projection + positional encoding."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class TemporalEmbedding(nn.Module):
    """Projects raw sensor input to d_model and adds positional encoding."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P1.U1.
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # Implemented in P1.U1.
        raise NotImplementedError
