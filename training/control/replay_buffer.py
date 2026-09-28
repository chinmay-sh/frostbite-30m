"""Experience replay buffer for off-policy control training."""

from __future__ import annotations

import random
from collections import deque

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity store with uniform OR prioritized (PER) sampling."""

    def __init__(self, capacity: int, seed: int = 0, alpha: float = 0.6) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.priorities: deque = deque(maxlen=capacity)  # aligned with buffer
        self.alpha = alpha  # 0 = uniform, 1 = fully priority-driven

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self, window: Tensor, action: int, reward: float,
        next_window: Tensor, done: bool, priority: float | None = None,
    ) -> None:
        """Store one decision transition (windows are (1, T, obs_dim))."""
        self.buffer.append((window, action, reward, next_window, done))
        # New transitions get max priority (guaranteed to be sampled) unless set.
        top = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(priority if priority is not None else top)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Draw a uniform minibatch, stacked and moved to `device`."""
        draws = self.rng.sample(list(self.buffer), batch_size)
        windows, actions, rewards, nexts, dones = zip(*draws)
        return {
            "windows": torch.cat(windows, dim=0).to(device),
            "actions": torch.tensor(actions, device=device),
            "rewards": torch.tensor(rewards, device=device, dtype=torch.float32),
            "next_windows": torch.cat(nexts, dim=0).to(device),
            "dones": torch.tensor(dones, device=device, dtype=torch.float32),
        }

    def sample_prioritized(
        self, batch_size: int, device: torch.device, beta: float = 0.4
    ) -> tuple[dict[str, Tensor], Tensor, list[int]]:
        """PER draw: probabilities ∝ priority^alpha; returns importance weights.

        The weights correct the priority-induced bias in the gradient and are
        annealed toward 1 as beta -> 1 over training.
        """
        prios = torch.tensor(self.priorities, dtype=torch.float64) ** self.alpha
        probs = prios / prios.sum()
        indices = torch.multinomial(probs, batch_size).tolist()

        transitions = [self.buffer[i] for i in indices]
        windows, actions, rewards, nexts, dones = zip(*transitions)

        # Importance-sampling weights: w = (N * p)^-beta, normalized by max.
        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights = (weights / weights.max()).float().to(device)

        batch = {
            "windows": torch.cat(windows, dim=0).to(device),
            "actions": torch.tensor(actions, device=device),
            "rewards": torch.tensor(rewards, device=device, dtype=torch.float32),
            "next_windows": torch.cat(nexts, dim=0).to(device),
            "dones": torch.tensor(dones, device=device, dtype=torch.float32),
        }
        return batch, weights, indices

    def update_priorities(self, indices: list[int], td_errors: Tensor) -> None:
        """Refresh priorities from |TD error| (+ small epsilon)."""
        for i, error in zip(indices, td_errors.detach().cpu().tolist()):
            self.priorities[i] = abs(error) + 1e-5
