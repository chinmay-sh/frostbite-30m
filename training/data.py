"""Synthetic time-series dataset for Phase-1 next-state prediction.

Each series is a mixture of slow sinusoids + noise; the task is to predict
the next sensor vector from the current window. Deterministic per-seed.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from frostbite.config.train import TrainConfig


class SyntheticTelemetry(Dataset):
    """Windows of synthetic sensor telemetry with next-state targets."""

    def __init__(
        self, config: TrainConfig, split: str, n_series: int = 64, sensor_dim: int = 128
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val, got {split}")
        self.seq_len = config.data.seq_len
        self.sensor_dim = sensor_dim
        # Different but deterministic seeds per split.
        seed = 1000 if split == "train" else 2000
        generator = torch.Generator().manual_seed(seed)

        total_len = self.seq_len + 1  # window + next-state target
        phase = torch.rand(n_series, 1, self.sensor_dim, generator=generator) * 6.2832
        freq = 0.2 + torch.rand(n_series, 1, self.sensor_dim, generator=generator)
        t = torch.linspace(0, total_len, total_len)
        series = torch.sin(phase + freq * t.unsqueeze(-1))  # (S, L, sensor)
        series = series + 0.05 * torch.randn(series.shape, generator=generator)
        self.series = series

    def __len__(self) -> int:
        return self.series.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return (window, next_state): (T, sensor_dim), (sensor_dim,)."""
        chunk = self.series[index]
        return chunk[:-1], chunk[-1]


def make_loaders(config: TrainConfig, n_series: int = 64) -> tuple:
    """Build deterministic train/val DataLoader pairs for a training run."""
    from torch.utils.data import DataLoader

    train = SyntheticTelemetry(config, "train", n_series)
    val = SyntheticTelemetry(config, "val", max(8, n_series // 8))
    common = dict(batch_size=config.data.batch_size, num_workers=0)
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(val, shuffle=False, **common),
    )