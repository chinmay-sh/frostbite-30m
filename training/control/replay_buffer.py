"""Experience replay buffer for off-policy control training."""

from __future__ import annotations

import random
from collections import deque

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity FIFO store of (window, action, reward, next_window, done)."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self, window: Tensor, action: int, reward: float,
        next_window: Tensor, done: bool,
    ) -> None:
        """Store one decision transition (windows are (1, T, obs_dim))."""
        self.buffer.append((window, action, reward, next_window, done))

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Draw a random minibatch, stacked and moved to `device`."""
        draws = self.rng.sample(list(self.buffer), batch_size)
        windows, actions, rewards, nexts, dones = zip(*draws)
        return {
            "windows": torch.cat(windows, dim=0).to(device),
            "actions": torch.tensor(actions, device=device),
            "rewards": torch.tensor(rewards, device=device, dtype=torch.float32),
            "next_windows": torch.cat(nexts, dim=0).to(device),
            "dones": torch.tensor(dones, device=device, dtype=torch.float32),
        }
