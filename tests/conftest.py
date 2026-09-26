"""Shared pytest fixtures: deterministic seeds, device selection, test configs."""

from __future__ import annotations

import pytest

from frostbite.config.arch import (
    ArchConfig,
    AttentionConfig,
    CfcConfig,
    CortexConfig,
    EmbeddingConfig,
    RouterConfig,
)
from frostbite.utils import resolve_device, seed_everything

DEFAULT_SEED = 42


@pytest.fixture()
def device():
    """Best available device (cuda if present, else cpu)."""
    return resolve_device()


@pytest.fixture()
def arch_config() -> ArchConfig:
    """Small architecture config sized for fast CPU tests."""
    return ArchConfig(
        d_model=64,
        n_blocks=2,
        n_heads=4,
        dropout=0.0,
        param_cap=30_000_000,
        embedding=EmbeddingConfig(sensor_dim=32, max_len=128),
        attention=AttentionConfig(causal=True),
        cfc=CfcConfig(
            units=64,
            mode="default",
            backbone_units=32,
            backbone_layers=1,
            mixed_memory=False,
        ),
        router=RouterConfig(hidden_dim=32),
        cortex=CortexConfig(choice_dim=8, hidden_dim=64),
    )


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Seed all RNGs before every test for reproducibility."""
    seed_everything(DEFAULT_SEED)
    yield
