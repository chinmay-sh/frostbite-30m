"""Auto-RL micro-router producing Route/Skip/Halt logits."""

from __future__ import annotations

from enum import IntEnum

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class RoutingAction(IntEnum):
    """Discrete actions of the internal RL cell."""

    ROUTE = 0  # pass through the CfC substrate
    SKIP = 1  # bypass the substrate via residual
    HALT = 2  # exit the block stack early


class AutoRLCell(nn.Module):
    """Two-layer MLP emitting one logit per routing action for every token."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.router.hidden_dim),
            nn.GELU(),
            nn.Linear(config.router.hidden_dim, len(RoutingAction)),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Map attention output (B, T, d_model) to action logits (B, T, 3)."""
        if z.dim() != 3:
            raise ValueError(f"Expected 3-D input (B, T, d_model), got {z.dim()}-D")
        return self.net(z)
