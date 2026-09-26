"""Sanity tests for reproducibility and parameter counting (P0.U4)."""

from __future__ import annotations

import torch
from torch import nn

from frostbite.utils import count_params, seed_everything


def test_seed_everything_is_deterministic():
    seed_everything(123)
    a = torch.randn(16)

    seed_everything(123)
    b = torch.randn(16)

    assert torch.equal(a, b)


def test_count_params_matches_numel():
    linear = nn.Linear(8, 12, bias=True)
    # 8*12 weights + 12 biases = 108
    assert count_params(linear) == 108


def test_count_params_ignores_frozen():
    linear = nn.Linear(8, 12, bias=False)
    linear.weight.requires_grad_(False)
    assert count_params(linear) == 0
