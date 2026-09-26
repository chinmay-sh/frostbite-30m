"""Auto-RL micro-router producing Route/Skip/Halt logits."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class AutoRLCell(nn.Module):
    """Two-layer MLP emitting 3 routing logits (ROUTE/SKIP/HALT) per token."""

    def __init__(self, config: ArchConfig) -> None:
        # Implemented in P2.U1.
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tensor:
        # Implemented in P2.U1.
        returns: Tensor = None  # type: ignore[assignment]
        return returns
