"""Full Frostbite-30M model assembly."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from frostbite.blocks.reinforced_layer import ReinforcedLiquidBlock
from frostbite.config.arch import ArchConfig
from frostbite.heads.laya_cortex import CortexOutput, LayaCortex
from frostbite.modules.embedding import TemporalEmbedding
from frostbite.modules.routing_state import RoutingState


@dataclass
class ModelOutput:
    """Cortex outputs plus routing telemetry from the forward pass."""

    cortex: CortexOutput
    halt_layer: Tensor  # (B,) index of the layer at which each sample halted
    route_probs: Tensor  # (n_blocks, B, T, 3) routing probabilities per block


class FrostbiteModel(nn.Module):
    """Embedding → 6 Reinforced Liquid Blocks → Laya cortex."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.n_blocks = config.n_blocks
        self.embedding = TemporalEmbedding(config)
        self.blocks = nn.ModuleList(
            [ReinforcedLiquidBlock(config) for _ in range(config.n_blocks)]
        )
        self.cortex = LayaCortex(config)

    def forward(self, x: Tensor, tau: float = 1.0) -> ModelOutput:
        """Run the full stack; HALT short-circuits the remaining blocks."""
        hidden = self.embedding(x)
        batch, seq_len, d_model = hidden.shape
        state = RoutingState.fresh(batch, seq_len, d_model, hidden.device)

        route_probs: list[Tensor] = []
        halt_layer = torch.full(
            (batch,), self.n_blocks, dtype=torch.long, device=hidden.device
        )
        for i, block in enumerate(self.blocks):
            if not self.training and state.all_halted():
                break  # every token halted: remaining blocks are dead compute
            result = block(hidden, state, tau=tau)
            hidden, state = result.output, result.state
            route_probs.append(torch.softmax(result.logits, dim=-1))
            # A sample halts when its last active token halts.
            sample_halted = state.halted.all(dim=1)
            halted_now = sample_halted & (halt_layer == self.n_blocks)
            halt_layer = torch.where(halted_now, i, halt_layer)

        hidden = state.resolve(hidden)  # frozen reps for halted tokens
        return ModelOutput(self.cortex(hidden), halt_layer, torch.stack(route_probs))
