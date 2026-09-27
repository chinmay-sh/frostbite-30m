"""P6.U4: routing ablations — force SKIP-only / ROUTE-all, same trunk."""

from __future__ import annotations

import torch

from frostbite.model import FrostbiteModel
from frostbite.modules.auto_rl_cell import RoutingAction
from training.control.policy_wrapper import ControlPolicy


def force_action(model: FrostbiteModel, action: RoutingAction) -> None:
    """Bias every router to always pick `action` (argmax-safe, no surgery)."""
    with torch.no_grad():
        for block in model.blocks:
            block.router.net[-1].bias.copy_(
                torch.tensor(
                    [50.0 if i == int(action) else -50.0 for i in range(3)],
                    device=block.router.net[-1].bias.device,
                )
            )


def ablation_suite(
    arch_path: str, checkpoint_path: str, episodes: int = 30
) -> dict[str, "EvalResult"]:
    """Evaluate learned router vs SKIP-only vs ROUTE-all on one shared trunk."""
    from training.control.evaluate import EvalResult, evaluate, load_policy

    results = {}

    learned = load_policy(arch_path, checkpoint_path)
    results["learned"] = evaluate(learned, episodes=episodes)

    skip = load_policy(arch_path, checkpoint_path)
    force_action(skip.model, RoutingAction.SKIP)
    results["skip_only"] = evaluate(skip, episodes=episodes)

    route = load_policy(arch_path, checkpoint_path)
    force_action(route.model, RoutingAction.ROUTE)
    results["route_all"] = evaluate(route, episodes=episodes)

    return results
