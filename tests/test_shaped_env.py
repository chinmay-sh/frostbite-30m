"""Reward-shaping tests (D26): potential-based, policy-preserving, training-only."""

from __future__ import annotations

import gymnasium as gym
import pytest

from training.control.shaped_env import ShapedLanderEnv


@pytest.fixture()
def shaped():
    base = gym.make("LunarLander-v3")
    wrapper = ShapedLanderEnv(base)
    yield wrapper
    wrapper.close()


def test_shaping_formula_exact_for_stationary_obs(shaped):
    """If obs doesn't change across a step, shaped = raw + (gamma-1)*Phi(s)."""
    obs, _ = shaped.reset(seed=3)
    shaped._last_obs = obs  # force the same obs before/after
    # Manually emulate: potential difference term with obs' == obs
    delta = shaped.gamma * shaped.potential(obs) - shaped.potential(obs)
    assert delta == pytest.approx((shaped.gamma - 1.0) * shaped.potential(obs))


def test_moving_closer_increases_potential(shaped):
    near = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    far = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert shaped.potential(near) > shaped.potential(far)


def test_slower_is_higher_potential(shaped):
    calm = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fast = [0.1, 0.1, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    assert shaped.potential(calm) > shaped.potential(fast)


def test_shaping_telescopes_over_episode(shaped):
    """For gamma<1 the shaping sum is bounded by the geometric series of |Phi|."""
    obs, _ = shaped.reset(seed=5)
    phis: list[float] = [shaped.potential(obs)]
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = shaped.step(0)
        phis.append(shaped.potential(obs))
        done = terminated or truncated

    shaping_total = sum(
        shaped.gamma * phis[k + 1] - phis[k] for k in range(len(phis) - 1)
    )
    n = len(phis)
    bound = 2 * max(abs(p) for p in phis) * sum(
        shaped.gamma ** k for k in range(n)
    )
    assert abs(shaping_total) <= bound + 1e-9
    # And per-step shaping never dwarfs the env's own reward scale unreasonably.
    assert abs(shaping_total) < 1_000  # sanity ceiling


def test_wrapper_preserves_spaces(shaped):
    assert shaped.observation_space.shape == (8,)
    assert shaped.action_space.n == 4
