"""PPO trainer tests (option c, D22) + trunk action head (option b, D21)."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from frostbite.model import FrostbiteModel
from frostbite.utils import seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.rollout import run_episode
from training.control.train_ppo import PPOConfig, PPOControlTrainer, compute_gae


@pytest.fixture()
def env():
    environment = gym.make("LunarLander-v3")
    environment.action_space.seed(0)
    yield environment
    environment.close()


def test_gae_terminal_bootstrap():
    """Single-step episode: A_0 = r_0 - V_0, R_0 = r_0."""
    advs, rets = compute_gae([1.0], [0.4], gamma=0.99, lam=0.95)
    assert advs[0].item() == pytest.approx(0.6)
    assert rets[0].item() == pytest.approx(1.0)


def test_gae_longer_episode():
    """GAE with lam=1: A_1 = 0.5; A_0 = delta_0 + gamma*A_1 = 0.75 + 0.25."""
    rewards, values = [1.0, 1.0], [0.5, 0.5]
    advs, _ = compute_gae(rewards, values, gamma=0.5, lam=1.0)
    assert advs[1].item() == pytest.approx(0.5)
    assert advs[0].item() == pytest.approx(1.0)


def test_action_head_reads_trunk(arch_config):
    """D21: the action head input dim is d_model (trunk), not choice_dim."""
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    assert policy.action_head.in_features == arch_config.d_model
    decision = policy.act(torch.randn(1, policy.window, 8))
    assert decision.action in range(4)


def test_trunk_exposed_in_model_output(arch_config):
    out = FrostbiteModel(arch_config).eval()(torch.randn(1, 4, arch_config.embedding.sensor_dim))
    assert out.trunk.shape == (1, 4, arch_config.d_model)


def test_ppo_update_moves_policy(arch_config, env):
    """One PPO iteration collects, runs clipped epochs, and updates weights."""
    seed_everything(0)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = PPOControlTrainer(
        policy, env,
        PPOConfig(episodes_per_update=1, updates=1, ppo_epochs=2,
                  minibatch_size=64, log_every=1),
    )
    episode = run_episode(env, policy, seed=1)
    before = policy.action_head.weight.detach().clone()
    metrics = trainer.update(0, [episode])
    assert not torch.allclose(before, policy.action_head.weight.detach())
    assert 0.0 <= metrics["clip_frac"] <= 1.0
    assert 0.0 <= metrics["entropy"] <= torch.log(torch.tensor(4.0)).item() + 1e-6


def test_ppo_trunk_group_respected(arch_config, env):
    """Stage-2 trunk_lr creates the second, lower-LR group; substrates excluded."""
    seed_everything(0)
    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    trainer = PPOControlTrainer(
        policy, env,
        PPOConfig(episodes_per_update=1, updates=1, trunk_lr=1e-5),
    )
    assert len(trainer.optimizer.param_groups) == 2
    trunk_ids = {id(p) for p in trainer.optimizer.param_groups[1]["params"]}
    for block in policy.model.blocks:
        for p in block.executor.parameters():
            assert id(p) not in trunk_ids
