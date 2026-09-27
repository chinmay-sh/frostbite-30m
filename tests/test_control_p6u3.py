"""P6.U3 tests: control fine-tuning mechanics (tiny scale, few updates)."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from frostbite.model import FrostbiteModel
from frostbite.utils import seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.train_control import ControlRLConfig, ControlTrainer, returns_to_go


def test_returns_to_go():
    rewards = [1.0, 1.0, 1.0]
    out = returns_to_go(rewards, gamma=0.5)
    # R0 = 1 + .5*(1 + .5*1) = 1.75, R1 = 1.5, R2 = 1.0
    assert out.tolist() == pytest.approx([1.75, 1.5, 1.0])


def test_beta_anneal_schedule():
    config = ControlRLConfig(beta_start=0.2, beta_end=0.02, updates=100, beta_anneal=0.5)
    trainer = ControlTrainer.__new__(ControlTrainer)
    trainer.config = config
    assert trainer.beta_at(0) == pytest.approx(0.2)
    assert trainer.beta_at(50) == pytest.approx(0.02)
    assert trainer.beta_at(99) == pytest.approx(0.02)  # clamped


def test_resume_uses_beta_floor(arch_config):
    """D19: a resumed trainer must continue beta at the floor, not restart."""
    seed_everything(0)
    env = gym.make("LunarLander-v3")
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = ControlTrainer(
        policy, env,
        ControlRLConfig(beta_start=0.2, beta_end=0.02, updates=10,
                        episodes_per_update=1, log_every=100),
        resume=True,
    )
    assert trainer.beta_floor == pytest.approx(0.02)
    env.close()


def test_trunk_frozen_by_default(arch_config):
    """trunk_lr=0: optimizer holds only heads; embedding/attention stay frozen."""
    seed_everything(0)
    env = gym.make("LunarLander-v3")
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = ControlTrainer(policy, env,
                             ControlRLConfig(episodes_per_update=1, log_every=100))
    optimized = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert len(trainer.optimizer.param_groups) == 1
    for p in policy.model.embedding.parameters():
        assert id(p) not in optimized
    env.close()


def test_stage2_unfreezes_trunk_at_low_lr(arch_config):
    """trunk_lr>0: two param groups, trunk at the lower LR, substrates frozen."""
    seed_everything(0)
    env = gym.make("LunarLander-v3")
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = ControlTrainer(
        policy, env,
        ControlRLConfig(episodes_per_update=1, log_every=100,
                        trunk_lr=1e-5, lr=3e-4),
    )
    assert len(trainer.optimizer.param_groups) == 2
    heads_group, trunk_group = trainer.optimizer.param_groups
    assert heads_group["lr"] == pytest.approx(3e-4)
    assert trunk_group["lr"] == pytest.approx(1e-5)
    trunk_ids = {id(p) for p in trunk_group["params"]}
    for block in policy.model.blocks:
        for p in block.executor.parameters():  # substrates never in trunk group
            assert id(p) not in trunk_ids
        for p in block.attention.parameters():
            assert id(p) in trunk_ids
    env.close()


def test_update_runs_and_moves_policy(arch_config):
    """One update collects episodes, backprops, and changes trainable weights."""
    seed_everything(0)
    env = gym.make("LunarLander-v3")
    env.action_space.seed(0)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = ControlTrainer(
        policy, env,
        ControlRLConfig(episodes_per_update=1, updates=1, log_every=1),
    )
    before = policy.action_head.weight.detach().clone()
    metrics = trainer.update(0)
    env.close()

    assert "return" in metrics and "route_frac" in metrics
    assert not torch.allclose(before, policy.action_head.weight.detach())


def test_loss_recomputation_matches_rollout_mode(arch_config):
    """The loss path must use eval routing (argmax), same as acting."""
    seed_everything(3)
    env = gym.make("LunarLander-v3")
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = ControlTrainer(
        policy, env,
        ControlRLConfig(episodes_per_update=1, updates=1),
    )
    from training.control.rollout import run_episode

    episode = run_episode(env, policy, seed=1)
    policy.model.eval()
    loss, _, entropy = trainer._episode_loss(episode, episode.total_reward)
    loss.backward()  # must be differentiable
    assert policy.action_head.weight.grad is not None
    assert torch.isfinite(policy.action_head.weight.grad).all()
    assert 0.0 <= entropy <= torch.log(torch.tensor(4.0)).item() + 1e-6
    env.close()
