"""Data pipeline tests (P3.U4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frostbite.config.train import TrainConfig
from training.data import SyntheticTelemetry, make_loaders

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def train_config() -> TrainConfig:
    return TrainConfig.from_yaml(CONFIGS / "train_phase1.yaml")


def test_deterministic_per_split(train_config):
    a = SyntheticTelemetry(train_config, "train", n_series=4)
    b = SyntheticTelemetry(train_config, "train", n_series=4)
    assert torch.equal(a[0][0], b[0][0])


def test_splits_differ(train_config):
    train = SyntheticTelemetry(train_config, "train", n_series=4)
    val = SyntheticTelemetry(train_config, "val", n_series=4)
    assert not torch.equal(train[0][0], val[0][0])


def test_window_and_target_shapes(train_config):
    ds = SyntheticTelemetry(train_config, "train", n_series=2)
    window, target = ds[0]
    assert window.shape == (train_config.data.seq_len, 128)
    assert target.shape == (128,)


def test_rejects_bad_split(train_config):
    with pytest.raises(ValueError, match="train or val"):
        SyntheticTelemetry(train_config, "test")


def test_make_loaders_batch_shape(train_config):
    train_loader, val_loader = make_loaders(train_config, n_series=8)
    windows, targets = next(iter(train_loader))
    assert windows.shape == (8, train_config.data.seq_len, 128)
    assert targets.shape == (8, 128)
    windows, _ = next(iter(val_loader))
    assert windows.shape[1:] == (train_config.data.seq_len, 128)
