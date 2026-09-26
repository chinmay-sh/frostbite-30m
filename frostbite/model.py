"""Full Frostbite-30M model assembly."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class FrostbiteModel(nn.Module):
    """Embedding → 6 Reinforced Liquid Blocks → Laya cortex."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P3.U3.
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Implemented in P3.U3.
        raise NotImplementedError
        # Placeholder return type annotation for P3.U3 implementation.
        # (choice, score, noul)
