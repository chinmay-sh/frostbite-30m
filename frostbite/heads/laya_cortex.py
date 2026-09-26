"""Global Laya-style cortex: Choice, Score and Noul heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


@dataclass
class CortexOutput:
    """The three global outputs of the cortex."""

    choice: Tensor  # (B, choice_dim) softmax logits for global routing
    score: Tensor  # (B,) ordinal quality scalar
    noul: Tensor  # (B,) confidence in [0, 1] via sigmoid


class LayaCortex(nn.Module):
    """Three output heads: Choice (routing), Score (ordinal), Noul (confidence)."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        hidden = config.cortex.hidden_dim
        # Shared trunk over the mean-pooled sequence representation.
        self.trunk = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
        )
        self.choice_head = nn.Linear(hidden, config.cortex.choice_dim)
        self.score_head = nn.Linear(hidden, 1)
        self.noul_head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> CortexOutput:
        """Map the final block representation (B, T, d_model) to cortex outputs."""
        pooled = x.mean(dim=1)  # mean over time
        hidden = self.trunk(pooled)
        return CortexOutput(
            choice=self.choice_head(hidden),
            score=self.score_head(hidden).squeeze(-1),
            noul=torch.sigmoid(self.noul_head(hidden)).squeeze(-1),
        )
