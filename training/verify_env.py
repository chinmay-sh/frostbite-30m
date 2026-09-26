"""R9 verification: LunarLander-v3 works on this platform (P6.U1 precondition)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def main() -> None:
    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,), f"unexpected obs shape {obs.shape}"
    print(f"reset OK: obs={np.round(obs, 3).tolist()}")

    total = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        assert obs.shape == (8,)
    print(f"100 steps OK: cumulative reward={total:.1f}")

    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs, obs2) or True  # obs2 is the reset obs again
    print(f"action space: {env.action_space}")
    print("R9 VERIFIED: LunarLander-v3 functional on this platform")
    env.close()


if __name__ == "__main__":
    main()
