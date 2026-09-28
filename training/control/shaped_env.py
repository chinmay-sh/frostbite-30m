"""Potential-based reward shaping for LunarLander (D26).

Shaping is training-only: the dense gradient (closer-to-pad + slower) gives
Q-learning the landing-direction signal that random exploration never
provided. Because it is potential-based (r' = r + gamma*Phi(s') - Phi(s)),
the optimal policy is preserved (Ng et al., 1999) and evaluation stays on
the raw environment for honest numbers.
"""

from __future__ import annotations

import gymnasium as gym
import math


class ShapedLanderEnv(gym.Wrapper):
    """Adds potential-based shaping on top of the raw LunarLander reward."""

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        k_dist: float = 10.0,
        k_speed: float = 5.0,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.k_dist = k_dist
        self.k_speed = k_speed
        self._last_obs = None

    def potential(self, obs) -> float:
        """Phi(s): negative distance-to-pad and speed (lower = closer to landing)."""
        x, y, vx, vy = obs[0], obs[1], obs[2], obs[3]
        dist = math.sqrt(x * x + y * y)
        speed = math.sqrt(vx * vx + vy * vy)
        return -(self.k_dist * dist + self.k_speed * speed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping = self.gamma * self.potential(obs) - self.potential(self._last_obs)
        self._last_obs = obs
        return obs, reward + shaping, terminated, truncated, info
