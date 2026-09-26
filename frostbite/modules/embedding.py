"""Temporal embedding engine: sensor projection + positional encoding."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class TemporalEmbedding(nn.Module):
    """Projects raw sensor input to d_model and adds a learned positional encoding."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.max_len = config.embedding.max_len
        self.input_proj = nn.Linear(config.embedding.sensor_dim, config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(self.max_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Embed (B, T, sensor_dim) sensor windows into (B, T, d_model)."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D input (B, T, sensor_dim), got {x.dim()}-D")
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        hidden = self.input_proj(x) + self.pos_embedding[:seq_len]
        return self.dropout(hidden)

