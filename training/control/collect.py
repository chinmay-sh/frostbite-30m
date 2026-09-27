"""P6.U2: collect random-policy LunarLander trajectories as a Dataset."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch
from torch import Tensor
from torch.utils.data import Dataset

from frostbite.utils import seed_everything


class TrajectoryDataset(Dataset):
    """Window -> next-observation pairs drawn from random-policy episodes."""

    def __init__(self, windows: Tensor, targets: Tensor) -> None:
        self.windows = windows  # (N, T, obs_dim)
        self.targets = targets  # (N, obs_dim)

    def __len__(self) -> int:
        return self.windows.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.windows[index], self.targets[index]


def collect_random_trajectories(
    env_id: str = "LunarLander-v3",
    window: int = 32,
    n_episodes: int = 400,
    seed: int = 123,
) -> TrajectoryDataset:
    """Run random episodes; every step becomes one (window, next_obs) sample."""
    seed_everything(seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)  # reset() alone does not re-seed action sampling

    windows: list[Tensor] = []
    targets: list[Tensor] = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        history: list[Tensor] = []

        def current_window() -> Tensor:
            recent = history[-window:]  # only the last `window` observations
            pad = window - len(recent)
            rows = [torch.zeros(8)] * pad + recent if pad else recent
            return torch.stack(rows)

        done = False
        while not done:
            history.append(torch.as_tensor(obs, dtype=torch.float32))
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            windows.append(current_window())
            targets.append(torch.as_tensor(obs, dtype=torch.float32))
            done = terminated or truncated

    env.close()
    return TrajectoryDataset(torch.stack(windows), torch.stack(targets))


def save_dataset(dataset: TrajectoryDataset, path: str) -> None:
    """Persist the dataset tensor pair."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"windows": dataset.windows, "targets": dataset.targets}, path
    )


def load_dataset(path: str) -> TrajectoryDataset:
    """Load a persisted dataset."""
    payload = torch.load(path, weights_only=True)
    return TrajectoryDataset(payload["windows"], payload["targets"])
