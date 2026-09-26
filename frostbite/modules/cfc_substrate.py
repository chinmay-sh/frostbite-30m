"""Liquid time-dynamics substrate wrapping the ncps CfC cell."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class CfCSubstrate(nn.Module):
    """Liquid time-dynamics layer wrapped around an ncps CfC cell."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P1.U3.
        raise NotImplementedError

    def forward(self, x: Tensor, hidden: Tensor | None = None) -> Tensor:
        # Implemented in P1.U3.
        raise NotImplementedError
