"""Multi-head self-attention with pre-norm residual wrapper."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class SpatialAttention(nn.Module):
    """Pre-norm multi-head self-attention over the sequence dimension."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
            )
        self.causal = config.attention.causal
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Attend over (B, T, d_model); returns same shape (residual included)."""
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dropout_p = self.dropout.p if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return residual + self.dropout(self.proj(out))
