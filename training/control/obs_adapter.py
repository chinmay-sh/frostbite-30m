"""Observation adapter: raw env observations -> Frostbite sensor windows."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn


class ObsAdapter(nn.Module):
    """Projects raw observations to sensor_dim and builds a rolling window."""

    def __init__(self, obs_dim: int, sensor_dim: int, window: int = 32) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.sensor_dim = sensor_dim
        self.window = window
        self.proj = nn.Linear(obs_dim, sensor_dim)

    def forward(self, history: Tensor) -> Tensor:
        """Project a padded observation history (B, T, obs_dim) to sensor space."""
        if history.dim() != 3:
            raise ValueError(f"Expected 3-D history (B, T, obs_dim), got {history.dim()}-D")
        if history.size(-1) != self.obs_dim:
            raise ValueError(f"Expected obs_dim {self.obs_dim}, got {history.size(-1)}")
        return self.proj(history)


class ObsWindow:
    """Rolling buffer of the last `window` observations (zero-padded early)."""

    def __init__(self, window: int, obs_dim: int) -> None:
        self.buffer = deque(maxlen=window)
        self.window = window
        self.obs_dim = obs_dim

    def reset(self) -> None:
        """Clear the buffer at episode start."""
        self.buffer.clear()

    def push(self, obs: Tensor) -> None:
        """Append one raw observation (obs_dim,)."""
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected obs ({self.obs_dim},), got {tuple(obs.shape)}")
        self.buffer.append(obs)

    def tensor(self) -> Tensor:
        """Return the zero-padded window as (1, T, obs_dim)."""
        pad = self.window - len(self.buffer)
        rows = list(self.buffer)
        if pad > 0:
            rows = [torch.zeros(self.obs_dim)] * pad + rows
        return torch.stack(rows).unsqueeze(0)
