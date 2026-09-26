"""Per-token halting bookkeeping across the block stack."""

from __future__ import annotations

import torch
from torch import Tensor

from frostbite.modules.auto_rl_cell import RoutingAction


class RoutingState:
    """Tracks which tokens have halted and their frozen representations."""

    def __init__(self, halted: Tensor | None, carried: Tensor | None) -> None:
        self.halted = halted  # (B, T) bool
        self.carried = carried  # (B, T, d_model) snapshot at halt time

    @classmethod
    def fresh(cls, batch: int, seq_len: int, d_model: int, device: torch.device) -> RoutingState:
        """Create the initial (nothing halted) state."""
        return cls(
            halted=torch.zeros(batch, seq_len, dtype=torch.bool, device=device),
            carried=torch.zeros(batch, seq_len, d_model, device=device),
        )

    def update(self, output: Tensor, actions: Tensor) -> RoutingState:
        """Freeze tokens that chose HALT in this block; returns the new state."""
        newly_halted = actions[..., int(RoutingAction.HALT)] > 0.5
        halted = newly_halted if self.halted is None else (self.halted | newly_halted)
        carried = torch.where(newly_halted.unsqueeze(-1), output, self.carried)
        return RoutingState(halted, carried)

    def resolve(self, current: Tensor) -> Tensor:
        """Halted tokens get their frozen rep; active tokens get the current one."""
        if self.halted is None:
            return current
        return torch.where(self.halted.unsqueeze(-1), self.carried, current)

    def all_halted(self) -> bool:
        """True when every token has halted — the block stack can stop."""
        return bool(self.halted.all())
