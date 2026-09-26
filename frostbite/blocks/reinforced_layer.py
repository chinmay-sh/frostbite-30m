"""Reinforced Liquid Block: attention + router + CfC/skip/halt paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from frostbite.config.arch import ArchConfig
from frostbite.modules.attention import SpatialAttention
from frostbite.modules.auto_rl_cell import AutoRLCell
from frostbite.modules.branch_executor import BranchExecutor
from frostbite.modules.cfc_substrate import CfCSubstrate
from frostbite.modules.routing_state import RoutingState
from frostbite.modules.sampling import gumbel_sample


@dataclass
class BlockOutput:
    """Everything one block produces for the stack and for later RL credit."""

    output: Tensor  # (B, T, d_model) block output
    state: RoutingState  # halting bookkeeping after this block
    logits: Tensor  # (B, T, 3) routing logits (for logging / REINFORCE)


class ReinforcedLiquidBlock(nn.Module):
    """One hybrid layer with internal Route/Skip/Halt control flow."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.attention = SpatialAttention(config)
        self.router = AutoRLCell(config)
        self.executor = BranchExecutor(CfCSubstrate(config))
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Tensor, state: RoutingState, tau: float = 1.0
    ) -> BlockOutput:
        """Attend, route, mix branches, and update halting state.

        Training samples actions via straight-through Gumbel; evaluation uses
        deterministic argmax one-hot (parity guaranteed by P2.U3 tests).
        """
        z = self.attention(x)
        logits = self.router(z)
        actions = self._select_actions(logits, tau)
        mixed, _ = self.executor(z, actions)
        output = self.norm(mixed)
        return BlockOutput(output, state.update(output, actions), logits)

    def _select_actions(self, logits: Tensor, tau: float) -> Tensor:
        """One-hot actions: sampled while training, argmax at evaluation."""
        if self.training:
            return gumbel_sample(logits, tau=tau)
        return torch.nn.functional.one_hot(
            logits.argmax(dim=-1), num_classes=logits.size(-1)
        ).to(logits.dtype)
