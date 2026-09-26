"""Global Laya-style cortex: Choice, Score and Noul heads."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class LayaCortex(nn.Module):
    """Three output heads: Choice (routing), Score (ordinal), Noul (confidence)."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P3.U2.
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Implemented in P3.U2.
        raise NotImplementedError
