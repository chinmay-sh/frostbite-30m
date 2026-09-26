"""Phase-2 REINFORCE training loop for routers and cortex."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from collections.abc import Callable

from frostbite.model import FrostbiteModel
from frostbite.modules.auto_rl_cell import RoutingAction
from frostbite.modules.routing_state import RoutingState
from training.environment import TelemetryEnv
from training.rewards import RewardCalculator, RewardConfig

# Optional seam: (predicted, target_mean, difficulties, actions_per_layer) -> (B,).
# Must depend on the ACTIONS, else the policy gradient vanishes in expectation.
TaskRewardFn = Callable[..., Tensor]


@dataclass
class RLConfig:
    """Hyperparameters of the Phase-2 policy-gradient loop."""

    lr: float = 3e-4
    entropy_coef: float = 0.01
    baseline_decay: float = 0.95
    batch_size: int = 16
    updates: int = 200
    log_every: int = 10


@dataclass
class Rollout:
    """One batch of on-policy experience."""

    log_probs: list[Tensor]  # per layer, (B,) summed token log-probs
    entropies: list[Tensor]  # per layer, scalar policy entropy
    rewards: list[Tensor]  # per layer, (B,) total rewards
    route_frac: Tensor  # (n_blocks,) mean P(ROUTE) diagnostic


class Phase2Trainer:
    """Trains routers + cortex with REINFORCE; substrate trunk frozen (PLAN §3)."""

    def __init__(
        self,
        model: FrostbiteModel,
        env: TelemetryEnv,
        reward_config: RewardConfig | None = None,
        config: RLConfig | None = None,
        task_reward_fn: TaskRewardFn | None = None,
    ) -> None:
        self.model = model
        self.env = env
        self.rewards = RewardCalculator(reward_config or RewardConfig())
        self.config = config or RLConfig()
        self.task_reward_fn = task_reward_fn  # None -> default Score-head reward
        self.baseline = 0.0
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Only routers + cortex heads train in Phase 2 (trunk frozen)."""
        params: list[torch.nn.Parameter] = list(self.model.cortex.parameters())
        for block in self.model.blocks:
            params += list(block.router.parameters())
        return torch.optim.Adam(params, lr=self.config.lr)

    def freeze_trunk(self) -> None:
        """Freeze embedding, attention, substrates (PLAN Phase-2 protocol)."""
        self.model.embedding.requires_grad_(False)
        for block in self.model.blocks:
            block.attention.requires_grad_(False)
            block.executor.requires_grad_(False)

    def _collect_rollout(self) -> Rollout:
        """One env batch through the model with categorical sampling."""
        windows, difficulties, next_states = self.env.sample_batch(
            self.config.batch_size
        )
        device = next(self.model.parameters()).device
        windows = windows.to(device)
        next_states = next_states.to(device)

        hidden = self.model.embedding(windows)
        batch, seq_len, d_model = hidden.shape
        state = RoutingState.fresh(batch, seq_len, d_model, hidden.device)
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        actions_per_layer: list[Tensor] = []
        route_frac: list[Tensor] = []

        for block in self.model.blocks:
            z = block.attention(hidden)
            logits = block.router(z)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()  # (B, T) categorical sample
            one_hot = torch.nn.functional.one_hot(
                actions, num_classes=logits.size(-1)
            ).to(logits.dtype)
            mixed, _ = block.executor(z, one_hot)
            hidden = block.norm(mixed)
            state = state.update(hidden, one_hot)
            log_probs.append(dist.log_prob(actions).sum(dim=-1))  # (B,)
            entropies.append(dist.entropy().mean())
            actions_per_layer.append(actions)
            route_frac.append((actions == int(RoutingAction.ROUTE)).float().mean())

        hidden = state.resolve(hidden)
        predicted = self.model.cortex(hidden).score
        target_mean = next_states.mean(dim=-1)

        if self.task_reward_fn is not None:
            task = self.task_reward_fn(
                predicted, target_mean, difficulties, actions_per_layer
            )
        else:
            # Default extrinsic reward: +1 when the Score head lands near the
            # true mean next-state, -1 otherwise.
            error = (predicted - target_mean).abs()
            task = torch.where(error < 0.5, 1.0, -1.0)

        return Rollout(
            log_probs=log_probs,
            entropies=entropies,
            rewards=self.rewards.layer_rewards(task, actions_per_layer),
            route_frac=torch.stack(route_frac),
        )

    def update(self) -> dict[str, float]:
        """One REINFORCE update: advantage = reward - moving-average baseline."""
        rollout = self._collect_rollout()

        policy_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)
        for log_probs, entropy, reward in zip(
            rollout.log_probs, rollout.entropies, rollout.rewards
        ):
            mean_reward = reward.mean().item()
            self.baseline = (
                self.config.baseline_decay * self.baseline
                + (1 - self.config.baseline_decay) * mean_reward
            )
            advantage = (reward - self.baseline).detach()
            policy_loss = policy_loss - (log_probs * advantage).mean()
            entropy_loss = entropy_loss - entropy  # maximize entropy

        loss = policy_loss + self.config.entropy_coef * entropy_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward": torch.stack(rollout.rewards).mean().item(),
            "route_frac": rollout.route_frac.mean().item(),
            "entropy": torch.stack(rollout.entropies).mean().item(),
            "baseline": self.baseline,
        }

    def train(self) -> list[dict[str, float]]:
        """Run the configured number of updates; returns metrics history."""
        history: list[dict[str, float]] = []
        for step in range(self.config.updates):
            metrics = self.update()
            history.append(metrics)
            if (step + 1) % self.config.log_every == 0:
                print(
                    f"step {step + 1:4d} | reward {metrics['reward']:+.3f} "
                    f"| route {metrics['route_frac']:.3f} "
                    f"| entropy {metrics['entropy']:.3f} "
                    f"| baseline {metrics['baseline']:+.3f}",
                    flush=True,
                )
        return history
