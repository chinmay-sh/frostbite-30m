"""Phase-1 trainer tests (P3.U5) — Milestone 2: training runs without gradient breaks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from frostbite.config.arch import ArchConfig
from frostbite.config.train import AmpConfig, CheckpointConfig, DataConfig, GumbelConfig, OptimConfig, TrainConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import seed_everything
from training.data import SyntheticTelemetry
from training.phase1_supervised import Phase1Trainer

CONFIGS = Path(__file__).parent.parent / "configs"


def _tiny_config(tmp_path: Path, amp_enabled: bool = False) -> TrainConfig:
    return TrainConfig(
        run_name="test",
        seed=42,
        data=DataConfig(seq_len=8, batch_size=4, num_workers=0),
        optim=OptimConfig(optimizer="adamw", lr=1e-3, weight_decay=0.0,
                          grad_clip=1.0, warmup_steps=1, schedule="cosine"),
        gumbel=GumbelConfig(tau_start=2.0, tau_end=0.5, anneal_epochs=2),
        amp=AmpConfig(enabled=amp_enabled, dtype="bfloat16"),
        checkpoint=CheckpointConfig(dir=str(tmp_path / "runs"), save_every_steps=0, resume=None),
    )


def _tiny_arch(arch_config: ArchConfig) -> ArchConfig:
    return arch_config


def _loader(config: TrainConfig, sensor_dim: int = 32) -> DataLoader:
    ds = SyntheticTelemetry(config, "train", n_series=8, sensor_dim=sensor_dim)
    return DataLoader(ds, batch_size=config.data.batch_size)


def test_loss_decreases(tmp_path, arch_config):
    """Milestone 2 AC: the loop runs and loss decreases over epochs."""
    seed_everything(42)
    config = _tiny_config(tmp_path)
    model = FrostbiteModel(_tiny_arch(arch_config))
    trainer = Phase1Trainer(config, model)
    loader = _loader(config)

    first = trainer.train_epoch(loader, epoch=0, total_steps=20)
    for epoch in range(1, 6):
        metrics = trainer.train_epoch(loader, epoch=epoch, total_steps=20)
    assert metrics["loss"] < first["loss"], (
        f"loss did not decrease: {first['loss']:.4f} -> {metrics['loss']:.4f}"
    )
    assert 0.0 <= metrics["route_frac"] <= 1.0


def test_tau_anneals(tmp_path, arch_config):
    trainer = Phase1Trainer(_tiny_config(tmp_path), FrostbiteModel(arch_config))
    assert trainer.tau_at(0) == pytest.approx(2.0)
    assert trainer.tau_at(1) == pytest.approx(1.25)
    assert trainer.tau_at(5) == pytest.approx(0.5)  # clamped after anneal window


def test_lr_schedule_warmup_then_decay(tmp_path, arch_config):
    trainer = Phase1Trainer(_tiny_config(tmp_path), FrostbiteModel(arch_config))
    assert trainer.lr_at(0, 100) == 0.0
    assert trainer.lr_at(1, 100) == pytest.approx(trainer.config.optim.lr)
    assert trainer.lr_at(50, 100) < trainer.config.optim.lr
    assert trainer.lr_at(100, 100) == pytest.approx(0.0, abs=1e-6)


def test_checkpoint_roundtrip(tmp_path, arch_config):
    config = _tiny_config(tmp_path)
    trainer = Phase1Trainer(config, FrostbiteModel(arch_config))
    trainer.global_step = 7
    path = trainer.save_checkpoint()

    other = Phase1Trainer(config, FrostbiteModel(arch_config))
    other.load_checkpoint(path)
    assert other.global_step == 7
    for (name, a), (_, b) in zip(
        trainer.model.state_dict().items(), other.model.state_dict().items()
    ):
        assert torch.equal(a, b), name


def test_smoke_on_prod_config(tmp_path):
    """Full production model trains 2 steps on CPU without gradient breaks."""
    seed_everything(0)
    arch = ArchConfig.from_yaml(CONFIGS / "arch_30m.yaml")
    config = _tiny_config(tmp_path)
    config = TrainConfig(
        **{**config.__dict__, "data": DataConfig(seq_len=16, batch_size=2, num_workers=0)}
    )
    trainer = Phase1Trainer(config, FrostbiteModel(arch))
    loader = DataLoader(
        SyntheticTelemetry(config, "train", n_series=4, sensor_dim=arch.embedding.sensor_dim),
        batch_size=2,
    )
    metrics = trainer.train_epoch(loader, epoch=0, total_steps=2)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
