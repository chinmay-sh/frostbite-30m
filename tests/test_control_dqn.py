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


def test_prioritized_sampling_favors_high_priority():
    """D32: PER draws high-priority transitions more often.

    p(0) = 100^0.6 / (100^0.6 + 9) ~= 0.64 -> inclusion in a 4-of-10
    no-replacement draw ~= 0.99 (uniform would be 0.4).
    """
    buffer = ReplayBuffer(capacity=10, seed=0)
    window = torch.zeros(1, 4, 8)
    for i in range(10):
        buffer.push(window, 0, float(i), window, False,
                    priority=100.0 if i == 0 else 1.0)
    draws_with_zero = 0
    total_draws = 200
    for _ in range(total_draws):
        _, _, indices = buffer.sample_prioritized(4, torch.device("cpu"))
        draws_with_zero += 1 if 0 in indices else 0
    assert draws_with_zero / total_draws > 0.8  # vs 0.4 uniform


def test_priority_refresh_from_td_error():
    buffer = ReplayBuffer(capacity=10, seed=0)
    window = torch.zeros(1, 4, 8)
    for i in range(4):
        buffer.push(window, 0, 0.0, window, False)
    buffer.update_priorities([0, 2], torch.tensor([5.0, 0.1]))
    assert buffer.priorities[0] == pytest.approx(5.0 + 1e-5)
    assert buffer.priorities[2] == pytest.approx(0.1 + 1e-5)


def test_soft_target_update_moves_slowly(arch_config, env):
    """D32: tau=0.005 nudges the target a tiny fraction toward online."""
    seed_everything(0)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    config = DQNConfig(episodes=1, tau=0.005)
    trainer = DQNTrainer(policy, env, config, seed=0)

    before = trainer.target.action_head.weight.detach().clone()
    # Nudge every online parameter far away.
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(1.0)
    trainer._update_target()
    after = trainer.target.action_head.weight.detach()
    delta = (after - before).abs().max().item()
    assert 0.0 < delta <= 0.005 + 1e-6  # tau fraction of the 1.0 nudge


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
    """D32: tau=0 uses the legacy hard sync — full copy after sync steps."""
    seed_everything(1)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    config = DQNConfig(episodes=1, warmup_decisions=2, batch_size=8,
                       action_repeat=2, target_sync_steps=3, tau=0.0)
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
