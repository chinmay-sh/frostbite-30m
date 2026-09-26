"""Config loading and validation tests (P0.U2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from frostbite.config import ArchConfig, TrainConfig

CONFIGS = Path(__file__).parent.parent / "configs"


class TestArchConfig:
    def test_loads_arch_30m(self):
        config = ArchConfig.from_yaml(CONFIGS / "arch_30m.yaml")
        assert config.d_model == 256
        assert config.n_blocks == 6
        assert config.n_heads == 4
        assert config.param_cap == 30_000_000
        assert config.embedding.sensor_dim == 128

    def test_rejects_unknown_field(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "model: {d_model: 256, n_blocks: 6, n_heads: 4, dropout: 0.1, param_cap: 30000000}\n"
            "embedding: {sensor_dim: 128, max_len: 1024, bogus: 1}\n"
            "attention: {causal: true}\n"
            "cfc: {units: 256, mode: default, backbone_units: 128, backbone_layers: 1, mixed_memory: false}\n"
            "router: {hidden_dim: 128}\n"
            "cortex: {choice_dim: 8, hidden_dim: 256}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid config structure"):
            ArchConfig.from_yaml(bad)


class TestTrainConfig:
    def test_loads_train_phase1(self):
        config = TrainConfig.from_yaml(CONFIGS / "train_phase1.yaml")
        assert config.data.batch_size == 128
        assert config.optim.lr == pytest.approx(3.0e-4)
        assert config.gumbel.tau_start == 2.0
        assert config.amp.dtype == "bfloat16"

    def test_rejects_missing_key(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("run_name: x\nseed: 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            TrainConfig.from_yaml(bad)
