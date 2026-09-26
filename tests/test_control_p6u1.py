"""P6.U1 tests: obs adapter, policy wrapper, rollouts (CPU, tiny arch)."""

from __future__ import annotations

import statistics

import gymnasium as gym
import pytest
import torch

from frostbite.model import FrostbiteModel
from frostbite.utils import count_params, seed_everything
from training.control.obs_adapter import ObsAdapter, ObsWindow
from training.control.policy_wrapper import ControlPolicy
from training.control.rollout import random_baseline, run_episode

ENV_ID = "LunarLander-v3"


@pytest.fixture(scope="module")
def env():
    environment = gym.make(ENV_ID)
    yield environment
    environment.close()


class TestObsAdapter:
    def test_window_pads_early_steps(self):
        window = ObsWindow(8, 4)
        window.push(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        tensor = window.tensor()
        assert tensor.shape == (1, 8, 4)
        assert tensor[0, -1, 0] == 1.0  # newest last
        assert tensor[0, 0].abs().sum() == 0.0  # zero pad at front

    def test_window_rolls_when_full(self):
        window = ObsWindow(4, 1)
        for i in range(6):
            window.push(torch.tensor([float(i)]))
        tensor = window.tensor()
        assert tensor[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]

    def test_adapter_projects(self, arch_config):
        adapter = ObsAdapter(8, arch_config.embedding.sensor_dim, window=16)
        out = adapter(torch.randn(2, 16, 8))
        assert out.shape == (2, 16, arch_config.embedding.sensor_dim)

    def test_adapter_rejects_bad_obs_dim(self, arch_config):
        adapter = ObsAdapter(8, arch_config.embedding.sensor_dim)
        with pytest.raises(ValueError, match="obs_dim"):
            adapter(torch.randn(2, 16, 9))


class TestControlPolicy:
    def test_decision_fields(self, arch_config):
        seed_everything(0)
        policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
        decision = policy.act(torch.randn(1, policy.window, 8))
        assert decision.action in range(4)
        assert torch.isfinite(decision.log_prob)
        assert 0.0 <= float(decision.confidence) <= 1.0
        assert 0.0 <= decision.route_frac <= 1.0

    def test_training_params_exclude_trunk(self, arch_config):
        policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
        trainable = {id(p) for p in policy.parameters_for_training()}
        for block in policy.model.blocks:
            for p in block.attention.parameters():
                assert id(p) not in trainable
            for p in block.executor.parameters():
                assert id(p) not in trainable


class TestRollout:
    def test_random_baseline_runs(self):
        returns = random_baseline(ENV_ID, episodes=3, seed=0)
        assert len(returns) == 3
        assert all(statistics.isfinite(r) for r in returns)

    def test_policy_episode_end_to_end(self, arch_config, env):
        """AC: random-init agent completes full episodes; shapes consistent."""
        seed_everything(1)
        policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
        episode = run_episode(env, policy, seed=7)
        assert episode.length >= 50  # landed or crashed, but ran
        assert len(episode.actions) == episode.length
        assert all(a in range(4) for a in episode.actions)
        assert all(w.shape == (1, policy.window, 8) for w in episode.windows)

    def test_seeded_rollouts_reproducible(self, arch_config, env):
        seed_everything(2)
        policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
        a = run_episode(env, policy, seed=11)
        seed_everything(2)
        policy2 = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
        b = run_episode(env, policy2, seed=11)
        assert a.actions == b.actions
        assert a.total_reward == pytest.approx(b.total_reward)


def test_adapter_param_cost_tiny(arch_config):
    """Adapter + action head must not threaten the 30M cap."""
    model = FrostbiteModel(arch_config)
    policy = ControlPolicy(model, obs_dim=8)
    extra = count_params(policy.adapter) + count_params(policy.action_head)
    assert extra < 50_000, f"control additions cost {extra:,} params"
