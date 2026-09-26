"""Reinforced Liquid Block: attention + router + CfC/skip/halt paths."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class ReinforcedLiquidBlock(nn.Module):
    """One hybrid layer with internal Route/Skip/Halt control flow."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P3.U1.
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # Implemented in P3.U1.
        raise NotImplementedError
