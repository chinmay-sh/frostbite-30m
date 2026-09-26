"""Shared pytest fixtures: deterministic seeds and device selection."""

from __future__ import annotations

import pytest

from frostbite.utils import resolve_device, seed_everything

DEFAULT_SEED = 42


@pytest.fixture()
def device():
    """Best available device (cuda if present, else cpu)."""
    return resolve_device()


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Seed all RNGs before every test for reproducibility."""
    seed_everything(DEFAULT_SEED)
    yield
