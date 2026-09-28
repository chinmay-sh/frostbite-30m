"""Double DQN + frame-skip tests (Tier 2, D23)."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from frostbite.model import FrostbiteModel
from frostbite.utils import seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.replay_buffer import ReplayBuffer
from training.control.train_dqn import DQNConfig, DQNTrainer


@pytest.fixture()
def env():
    environment = gym.make("LunarLander-v3")
    environment.action_space.seed(0)
    yield environment
    environment.close()


def test_replay_buffer_roundtrip():
    buffer = ReplayBuffer(capacity=100, seed=0)
    window = torch.randn(1, 8, 8)
    for i in range(10):
        buffer.push(window, i % 4, float(i), window + 1, i % 3 == 0)
    assert len(buffer) == 10
    batch = buffer.sample(4, torch.device("cpu"))
    assert batch["windows"].shape == (4, 8, 8)
    assert batch["actions"].shape == (4,)
    assert batch["dones"].dtype == torch.float32


def test_replay_buffer_evicts_oldest():
    buffer = ReplayBuffer(capacity=5, seed=0)
    window = torch.zeros(1, 4, 8)
    for i in range(8):
        buffer.push(window + i, 0, float(i), window, False)
    assert len(buffer) == 5
    # Only the 5 newest remain; the sample must contain only those rewards.
    rewards = set()
    for _ in range(30):
        rewards.update(buffer.sample(5, torch.device("cpu"))["rewards"].tolist())
    assert rewards == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_epsilon_schedule():
    trainer = DQNTrainer.__new__(DQNTrainer)
    trainer.config = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_decisions=100)
    assert trainer.epsilon_at(0) == pytest.approx(1.0)
    assert trainer.epsilon_at(50) == pytest.approx(0.525)
    assert trainer.epsilon_at(999) == pytest.approx(0.05)  # clamped


def test_q_features_detached_and_shaped(arch_config):
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    features = policy.q_features(torch.randn(2, policy.window, 8))
    assert features.shape == (2, arch_config.d_model)
    assert not features.requires_grad


def test_dqn_trains_and_moves_head(arch_config, env):
    """A short run fills the buffer, learns, and updates the Q-head."""
    seed_everything(0)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    config = DQNConfig(
        episodes=1, warmup_decisions=4, batch_size=8, action_repeat=2,
        log_every=1, target_sync_steps=2,
    )
    trainer = DQNTrainer(policy, env, config, seed=0)
    before = policy.action_head.weight.detach().clone()
    ret, decisions, loss = trainer._run_episode(first=True)
    assert decisions > 0
    assert not torch.allclose(before, policy.action_head.weight.detach())
    assert torch.isfinite(torch.tensor(loss))


def test_target_full_policy_copy(arch_config, env):
    """D24: after sync steps, the FULL target policy matches online."""
    seed_everything(1)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    config = DQNConfig(episodes=1, warmup_decisions=2, batch_size=8,
                       action_repeat=2, target_sync_steps=3)
    trainer = DQNTrainer(policy, env, config, seed=0)
    while trainer.gradient_steps < 3:
        if len(trainer.buffer) >= config.batch_size:
            trainer._gradient_step()
        else:
            trainer.buffer.push(
                torch.randn(1, policy.window, 8), 0, 0.0,
                torch.randn(1, policy.window, 8), False,
            )
    for name, online in policy.state_dict().items():
        target_val = trainer.target.state_dict()[name]
        assert torch.equal(online, target_val), name


def test_trunk_group_join_when_enabled(arch_config, env):
    """D24: trunk_lr>0 adds the second optimizer group; substrates excluded."""
    seed_everything(0)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    config = DQNConfig(episodes=1, trunk_lr=1e-5)
    trainer = DQNTrainer(policy, env, config, seed=0)
    assert len(trainer.optimizer.param_groups) == 2
    trunk_ids = {id(p) for p in trainer.optimizer.param_groups[1]["params"]}
    for block in policy.model.blocks:
        for p in block.executor.parameters():
            assert id(p) not in trunk_ids
        for p in block.attention.parameters():
            assert id(p) in trunk_ids
