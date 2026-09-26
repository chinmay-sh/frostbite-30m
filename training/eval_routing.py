"""Routing-behavior evaluation: does compute track dynamics difficulty? (P4.U3)

Milestone 3 evidence: the router should spend CfC compute (ROUTE) on complex
dynamics and skip/halt on static ones. Reports P(ROUTE) per difficulty
segment and the halt-layer distribution, for any checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from frostbite.model import FrostbiteModel
from frostbite.modules.auto_rl_cell import RoutingAction
from frostbite.modules.routing_state import RoutingState
from training.environment import TelemetryEnv


@dataclass
class BehaviorReport:
    """Routing statistics grouped by environment difficulty."""

    route_prob: dict[str, float]  # segment name -> mean P(ROUTE)
    halt_mean: dict[str, float]  # segment name -> mean halt layer (n_blocks = never)
    computes_per_sample: dict[str, float]  # mean ROUTE+substrate calls per sample


class RoutingBehaviorProbe:
    """Measures routing behavior of a FrostbiteModel on TelemetryEnv segments."""

    def __init__(self, model: FrostbiteModel, env: TelemetryEnv, batch: int = 8) -> None:
        self.model = model.eval()
        self.env = env
        self.batch = batch

    @torch.no_grad()
    def _segment_stats(self, difficulty: int, batches: int) -> dict[str, float]:
        routes, halts, computes = [], [], []
        device = next(self.model.parameters()).device
        for _ in range(batches):
            series = [self.env._series(difficulty) for _ in range(self.batch)]
            windows = torch.stack([s[:-1] for s in series]).to(device)
            out = self.model(windows)
            probs = out.route_probs  # (ran_blocks, B, T, 3)
            routes.append(probs[..., int(RoutingAction.ROUTE)].mean().item())
            halts.append(out.halt_layer.float().mean().item())
            computes.append(
                (probs[..., int(RoutingAction.ROUTE)].mean(dim=(1, 2)) > 0.5)
                .float().sum().item()
            )
        return {
            "route": sum(routes) / len(routes),
            "halt": sum(halts) / len(halts),
            "computes": sum(computes) / len(computes),
        }

    def report(self, batches: int = 8) -> BehaviorReport:
        """Aggregate routing statistics per difficulty segment."""
        route_prob, halt_mean, computes = {}, {}, {}
        for difficulty, name in enumerate(self.env.SEGMENTS):
            stats = self._segment_stats(difficulty, batches)
            route_prob[name] = stats["route"]
            halt_mean[name] = stats["halt"]
            computes[name] = stats["computes"]
        return BehaviorReport(route_prob, halt_mean, computes)


def print_report(title: str, report: BehaviorReport) -> None:
    """Human-readable dump of a behavior report."""
    print(f"\n== {title} ==")
    print(f"{'segment':<14}{'P(ROUTE)':>10}{'halt@':>8}{'substrates':>12}")
    for name in report.route_prob:
        print(
            f"{name:<14}{report.route_prob[name]:>10.3f}"
            f"{report.halt_mean[name]:>8.2f}"
            f"{report.computes_per_sample[name]:>12.2f}"
        )


def route_monotonicity(report: BehaviorReport) -> bool:
    """P(ROUTE) increases with dynamics difficulty (diagnostic only)."""
    r = report.route_prob
    return r["static"] < r["oscillatory"] < r["chaotic"]


def compute_monotonicity(report: BehaviorReport) -> bool:
    """Milestone-3 criterion (PLAN §3): compute spent grows with difficulty.

    Effective depth = halt layer + substrate usage: the model should run a
    deeper stack and more CfC substrates on chaotic segments than static ones.
    """
    depth = report.halt_mean
    substrates = report.computes_per_sample
    return (
        depth["static"] < depth["oscillatory"] < depth["chaotic"]
        and substrates["static"] < substrates["chaotic"]
    )
