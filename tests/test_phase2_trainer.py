"""Phase-2 REINFORCE trainer tests (P4.U2)."""

from __future__ import annotations

import pytest
import torch

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import count_params, seed_everything
from training.environment import TelemetryEnv
from training.phase2_reinforce import Phase2Trainer, RLConfig
from training.rewards import RewardConfig


@pytest.fixture()
def rl_setup(arch_config):
    seed_everything(0)
    model = FrostbiteModel(arch_config)
    env = TelemetryEnv(seq_len=8, sensor_dim=arch_config.embedding.sensor_dim, seed=1)
    config = RLConfig(batch_size=4, updates=3, log_every=1)
    trainer = Phase2Trainer(model, env, RewardConfig(), config)
    trainer.freeze_trunk()
    return model, trainer


def test_only_routers_and_cortex_train(rl_setup):
    """AC: embedding & attention frozen; only router/cortex in the optimizer."""
    model, trainer = rl_setup
    assert not any(p.requires_grad for p in model.embedding.parameters())
    for block in model.blocks:
        assert not any(p.requires_grad for p in block.attention.parameters())
        assert not any(p.requires_grad for p in block.executor.parameters())
        assert all(p.requires_grad for p in block.router.parameters())
    assert all(p.requires_grad for p in model.cortex.parameters())

    optimized = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    trainables = [
        p for p in model.parameters() if p.requires_grad and id(p) in optimized
    ]
    expected = count_params(model.cortex) + sum(
        count_params(b.router) for b in model.blocks
    )
    assert sum(p.numel() for p in trainables) == expected


def test_policy_gradient_updates_routers(rl_setup):
    """AC: updates change the routing distribution (weights move)."""
    model, trainer = rl_setup
    before = model.blocks[0].router.net[0].weight.detach().clone()
    trainer.update()
    after = model.blocks[0].router.net[0].weight.detach()
    assert not torch.allclose(before, after)


def test_entropy_does_not_collapse_early(rl_setup):
    """R5 guard: after a few updates the policy is not single-action."""
    model, trainer = rl_setup
    metrics = [trainer.update() for _ in range(5)]
    max_entropy = torch.log(torch.tensor(3.0)).item()
    assert all(m["entropy"] > 0.2 * max_entropy for m in metrics)


def test_metrics_are_finite(rl_setup):
    _, trainer = rl_setup
    metrics = trainer.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert 0.0 <= metrics["route_frac"] <= 1.0
