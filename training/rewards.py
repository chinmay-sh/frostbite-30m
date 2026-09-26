"""Intrinsic and extrinsic reward functions for internal RL cells."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from frostbite.modules.auto_rl_cell import RoutingAction


@dataclass(frozen=True)
class RewardConfig:
    """Weights and intrinsic shaping terms (PLAN §3, Phase 2)."""

    alpha: float = 1.0  # task reward weight
    beta: float = 0.2   # compute reward weight
    skip_bonus: float = 0.1
    halt_bonus: float = 0.2
    route_cost: float = -0.1


# Intrinsic compute reward per action, indexed by RoutingAction value.
COMPUTE_REWARDS = {
    int(RoutingAction.ROUTE): -0.1,
    int(RoutingAction.SKIP): 0.1,
    int(RoutingAction.HALT): 0.2,
}


class RewardCalculator:
    """Combines extrinsic task reward with per-layer intrinsic compute rewards."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute_reward(self, actions: Tensor) -> Tensor:
        """Intrinsic reward for one layer's chosen actions (B, T) -> (B, T)."""
        weights = torch.tensor(
            [self.config.route_cost, self.config.skip_bonus, self.config.halt_bonus],
            device=actions.device,
        )
        return weights[actions]

    def layer_rewards(
        self, task_reward_per_sample: Tensor, actions_per_layer: list[Tensor]
    ) -> list[Tensor]:
        """Total reward per layer: broadcast task term + layer-local compute term.

        PLAN: credit the compute reward to the specific layer that made each
        routing choice; the task reward is shared across layers (it is a
        sequence-level outcome).
        """
        rewards = []
        for actions in actions_per_layer:
            intrinsic = self.compute_reward(actions).mean(dim=-1)  # (B,)
            total = (
                self.config.alpha * task_reward_per_sample
                + self.config.beta * intrinsic
            )
            rewards.append(total)
        return rewards

    def total(self, task_reward_per_sample: Tensor, actions_per_layer: list[Tensor]) -> Tensor:
        """Total reward across layers, per sample (B,) — for reporting."""
        return torch.stack(self.layer_rewards(task_reward_per_sample, actions_per_layer)).sum(dim=0)
