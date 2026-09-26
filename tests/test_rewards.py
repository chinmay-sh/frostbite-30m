"""Environment and reward tests (P4.U1)."""

from __future__ import annotations

import pytest
import torch

from training.environment import EnvStep, TelemetryEnv, task_reward
from training.rewards import RewardCalculator, RewardConfig


class TestTelemetryEnv:
    def test_batch_shapes(self):
        env = TelemetryEnv(seq_len=64, sensor_dim=32, seed=1)
        windows, difficulties, nexts = env.sample_batch(4)
        assert windows.shape == (4, 64, 32)
        assert difficulties.shape == (4,)
        assert nexts.shape == (4, 32)
        assert set(difficulties.tolist()) <= {0, 1, 2}

    def test_deterministic_per_seed(self):
        a = TelemetryEnv(seq_len=16, sensor_dim=8, seed=3).sample_batch(2)
        b = TelemetryEnv(seq_len=16, sensor_dim=8, seed=3).sample_batch(2)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_difficulty_affects_dynamics(self):
        """Static series must have lower variance than chaotic ones."""
        env = TelemetryEnv(seq_len=256, sensor_dim=64, seed=5)
        static = env._series(0)
        chaotic = env._series(2)
        assert static.var(dim=0).mean() < chaotic.var(dim=0).mean()

    def test_task_reward_signs(self):
        pred = torch.tensor([[0.0, 0.0]])
        close = torch.tensor([[0.1, 0.1]])   # mean err 0.1 < 0.5
        far = torch.tensor([[2.0, 2.0]])     # mean err 2.0 > 0.5
        assert task_reward(pred, close).item() == pytest.approx(1.0)
        assert task_reward(pred, far).item() == pytest.approx(-1.0)


class TestRewardCalculator:
    def test_compute_reward_matches_plan_values(self):
        calc = RewardCalculator()
        actions = torch.tensor([[0, 1, 2]])  # ROUTE, SKIP, HALT
        rewards = calc.compute_reward(actions)
        expected = torch.tensor([[-0.1, 0.1, 0.2]])
        assert torch.allclose(rewards, expected, atol=1e-7)

    def test_credit_is_layer_local(self):
        """Each layer's reward reflects ITS actions, not other layers'."""
        calc = RewardCalculator(RewardConfig(beta=1.0, alpha=0.0))
        task = torch.zeros(1)
        layer_actions = [torch.zeros(1, 4, dtype=torch.long),  # all ROUTE
                         torch.ones(1, 4, dtype=torch.long)]   # all SKIP
        rewards = calc.layer_rewards(task, layer_actions)
        assert rewards[0].item() == pytest.approx(-0.1)
        assert rewards[1].item() == pytest.approx(0.1)

    def test_total_combines_alpha_beta(self):
        calc = RewardCalculator(RewardConfig(alpha=2.0, beta=0.5))
        task = torch.tensor([1.0])
        rewards = calc.layer_rewards(task, [torch.full((1, 4), 2, dtype=torch.long)])
        # 2.0*1.0 + 0.5*0.2 = 2.1
        assert rewards[0].item() == pytest.approx(2.1)
