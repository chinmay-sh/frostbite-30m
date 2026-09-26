"""Simulated telemetry environment for Phase-2 RL training.

Emits sensor windows whose dynamics difficulty varies by segment (static →
oscillatory → chaotic). Ground-truth difficulty lets P4.U3 prove the router
spends CfC compute on complex dynamics and skips on static ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from frostbite.utils import seed_everything


@dataclass
class EnvStep:
    """One environment interaction."""

    window: Tensor  # (T, sensor_dim) input window for the model
    difficulty: int  # 0=static, 1=oscillatory, 2=chaotic
    next_state: Tensor  # (sensor_dim,) the true next sensor vector


class TelemetryEnv:
    """Deterministic environment emitting difficulty-segmented telemetry."""

    SEGMENTS = ("static", "oscillatory", "chaotic")

    def __init__(
        self,
        seq_len: int = 256,
        sensor_dim: int = 128,
        seed: int = 7,
    ) -> None:
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.generator = torch.Generator().manual_seed(seed)

    def _series(self, difficulty: int) -> Tensor:
        """Generate one (seq_len+1, sensor_dim) series of the given difficulty."""
        length = self.seq_len + 1
        t = torch.linspace(0, 6.2832, length).unsqueeze(-1)
        phase = torch.rand(1, self.sensor_dim, generator=self.generator) * 6.2832

        if difficulty == 0:  # static: near-constant signals
            base = torch.rand(1, self.sensor_dim, generator=self.generator) * 2 - 1
            series = base.repeat(length, 1) + 0.02 * torch.randn(
                length, self.sensor_dim, generator=self.generator
            )
        elif difficulty == 1:  # oscillatory: slow sinusoid mixtures
            freq = 0.5 + torch.rand(1, self.sensor_dim, generator=self.generator)
            series = torch.sin(phase + freq * t)
        else:  # chaotic: Lorenz-like coupled triplets + noise
            freq = 2.0 + 2.0 * torch.rand(1, self.sensor_dim, generator=self.generator)
            series = torch.sin(phase + freq * t) * torch.cos(1.7 * t + phase)
            series = series + 0.1 * torch.sin(11.0 * t + phase)

        return series + 0.05 * torch.randn(length, self.sensor_dim, generator=self.generator)

    def sample_batch(self, batch: int) -> tuple[Tensor, Tensor, Tensor]:
        """Draw a mixed-difficulty batch: (windows, difficulties, next_states)."""
        windows, difficulties, nexts = [], [], []
        for _ in range(batch):
            difficulty = int(torch.randint(0, 3, (1,), generator=self.generator))
            series = self._series(difficulty)
            windows.append(series[:-1])
            difficulties.append(difficulty)
            nexts.append(series[-1])
        return torch.stack(windows), torch.tensor(difficulties), torch.stack(nexts)


def task_reward(predicted: Tensor, target: Tensor) -> Tensor:
    """Extrinsic reward: +1 when next-state prediction is within tolerance."""
    error = (predicted - target).abs().mean(dim=-1)
    return (error < 0.5).float() * 2.0 - 1.0  # +1 correct / -1 failure
