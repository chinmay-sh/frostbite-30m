"""P6.U4 tests: evaluation + ablation mechanics (tiny scale)."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from frostbite.model import FrostbiteModel
from frostbite.modules.auto_rl_cell import RoutingAction
from frostbite.utils import seed_everything
from training.control.ablate import force_action
from training.control.evaluate import evaluate, phase_telemetry
from training.control.policy_wrapper import ControlPolicy


@pytest.fixture()
def policy(arch_config):
    seed_everything(4)
    return ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)


def test_force_action_drives_argmax(policy):
    """Forcing SKIP zeroes routing; ROUTE raises it above the unforced value."""
    force_action(policy.model, RoutingAction.SKIP)
    decision = policy.act(torch.randn(1, policy.window, 8))
    assert decision.route_frac == pytest.approx(0.0, abs=1e-6)

    force_action(policy.model, RoutingAction.ROUTE)
    decision = policy.act(torch.randn(1, policy.window, 8))
    assert decision.route_frac > 0.9


def test_evaluate_returns_summary(policy):
    result = evaluate(policy, episodes=2)
    assert result.episodes == 2
    assert result.mean_return < 300  # random-ish policy on LunarLander
    assert result.std_return >= 0.0


def test_phase_telemetry_partitions_steps(policy):
    telemetry = phase_telemetry(policy, episodes=1)
    assert set(telemetry) == {"early", "mid", "late"}
    assert all(0.0 <= s["mean_route_frac"] <= 1.0 for s in telemetry.values())
