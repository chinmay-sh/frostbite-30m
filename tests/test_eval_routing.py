"""Routing-behavior evaluation tests (P4.U3)."""

from __future__ import annotations

import torch

from frostbite.model import FrostbiteModel
from frostbite.utils import seed_everything
from training.environment import TelemetryEnv
from training.eval_routing import (
    BehaviorReport,
    RoutingBehaviorProbe,
    compute_monotonicity,
    route_monotonicity,
)


def test_compute_monotonicity_criterion():
    """The Milestone-3 criterion: compute depth grows with difficulty."""
    good = BehaviorReport(
        route_prob={"static": 0.3, "oscillatory": 0.3, "chaotic": 0.3},
        halt_mean={"static": 2.0, "oscillatory": 4.0, "chaotic": 5.0},
        computes_per_sample={"static": 0.1, "oscillatory": 0.5, "chaotic": 1.0},
    )
    assert compute_monotonicity(good)

    flat = BehaviorReport(
        route_prob={"static": 0.3, "oscillatory": 0.3, "chaotic": 0.3},
        halt_mean={"static": 4.0, "oscillatory": 4.0, "chaotic": 4.0},
        computes_per_sample={"static": 0.5, "oscillatory": 0.5, "chaotic": 0.5},
    )
    assert not compute_monotonicity(flat)


def _probe(arch_config, seed=0) -> RoutingBehaviorProbe:
    seed_everything(seed)
    model = FrostbiteModel(arch_config).eval()
    env = TelemetryEnv(
        seq_len=8, sensor_dim=arch_config.embedding.sensor_dim, seed=3
    )
    return RoutingBehaviorProbe(model, env, batch=2)


def test_report_structure(arch_config):
    report = _probe(arch_config).report(batches=2)
    assert set(report.route_prob) == {"static", "oscillatory", "chaotic"}
    assert set(report.halt_mean) == {"static", "oscillatory", "chaotic"}
    for value in report.route_prob.values():
        assert 0.0 <= value <= 1.0


def test_deterministic_reports(arch_config):
    a = _probe(arch_config, seed=1).report(batches=2)
    b = _probe(arch_config, seed=1).report(batches=2)
    assert a.route_prob == b.route_prob


def test_reinforce_steers_router(arch_config):
    """Mechanism test: REINFORCE moves P(ROUTE) in the rewarded direction.

    Difficulty-monotonicity (P(ROUTE): static < oscillatory < chaotic) is the
    Milestone-3 claim and is measured on the real trained checkpoint by
    `training/run_phase2.py` — a unit-sized run is too RNG-sensitive to assert
    it reliably (calibrated: separation ~0.01 at 50 updates).
    """
    from training.phase2_reinforce import Phase2Trainer, RLConfig
    from training.rewards import RewardConfig

    seed_everything(2)
    model = FrostbiteModel(arch_config)
    env = TelemetryEnv(seq_len=8, sensor_dim=arch_config.embedding.sensor_dim, seed=9)

    def reward_route(predicted, target, difficulties, actions):
        frac = torch.stack([(a == 0).float().mean(dim=-1) for a in actions]).mean(dim=0)
        return 4.0 * frac - 1.0  # strong, action-dependent

    def reward_skip(predicted, target, difficulties, actions):
        frac = torch.stack([(a == 1).float().mean(dim=-1) for a in actions]).mean(dim=0)
        return 4.0 * frac - 1.0

    common = dict(batch_size=8, updates=40, lr=1e-3, entropy_coef=0.001, log_every=1000)

    trainer = Phase2Trainer(model, env, RewardConfig(beta=0.0), RLConfig(**common),
                            task_reward_fn=reward_route)
    trainer.freeze_trunk()
    before = RoutingBehaviorProbe(model, env, batch=8).report(batches=4).route_prob
    trainer.train()
    after_route = RoutingBehaviorProbe(model, env, batch=8).report(batches=4).route_prob

    mean_before = sum(before.values()) / 3
    mean_after = sum(after_route.values()) / 3
    assert mean_after > mean_before + 0.05, (
        f"rewarding ROUTE did not raise P(ROUTE): {mean_before:.3f} -> {mean_after:.3f}"
    )

    trainer2 = Phase2Trainer(model, env, RewardConfig(beta=0.0), RLConfig(**common),
                             task_reward_fn=reward_skip)
    trainer2.freeze_trunk()
    trainer2.train()
    after_skip = RoutingBehaviorProbe(model, env, batch=8).report(batches=4).route_prob
    mean_skip = sum(after_skip.values()) / 3
    assert mean_skip < mean_after - 0.05, (
        f"rewarding SKIP did not lower P(ROUTE): {mean_after:.3f} -> {mean_skip:.3f}"
    )
