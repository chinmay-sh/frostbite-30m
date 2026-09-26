"""Seeded on-policy episode collection on a Gymnasium environment."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from torch import Tensor


@dataclass
class Episode:
    """One episode of experience (aligned step lists)."""

    windows: list[Tensor]  # (1, T, obs_dim) padded windows
    actions: list[int]
    rewards: list[float]
    values: list[float]
    route_fracs: list[float]
    total_reward: float
    length: int


def run_episode(env: gym.Env, policy, seed: int | None = None) -> Episode:
    """Run one full episode with the current policy; returns collected experience."""
    from training.control.obs_adapter import ObsWindow

    obs, _ = env.reset(seed=seed)
    window = ObsWindow(policy.window, policy.adapter.obs_dim)

    windows, actions, rewards, values, route_fracs = [], [], [], [], []
    done = False
    total = 0.0

    while not done:
        window.push(torch.as_tensor(obs, dtype=torch.float32))
        decision = policy.act(window.tensor())
        obs, reward, terminated, truncated, _ = env.step(decision.action)

        windows.append(window.tensor())
        actions.append(decision.action)
        rewards.append(float(reward))
        values.append(float(decision.value.item()))
        route_fracs.append(decision.route_frac)
        total += float(reward)
        done = terminated or truncated

    return Episode(
        windows=windows,
        actions=actions,
        rewards=rewards,
        values=values,
        route_fracs=route_fracs,
        total_reward=total,
        length=len(actions),
    )


def random_baseline(env_id: str, episodes: int = 10, seed: int = 0) -> list[float]:
    """Uniform-random policy returns for calibration."""
    env = gym.make(env_id)
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return returns
