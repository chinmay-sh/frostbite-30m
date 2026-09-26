"""Reproducibility, device and parameter-count helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Return the best available device (cuda if present, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(module: nn.Module) -> int:
    """Count total trainable parameters of a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
