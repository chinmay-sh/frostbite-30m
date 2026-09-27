"""P6: PPO control fine-tuning — clipped surrogate with GAE (option c, D22)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from training.control.policy_wrapper import ControlPolicy
from training.control.rollout import Episode, run_episode
from training.control.train_control import ControlRLConfig


@dataclass
class PPOConfig(ControlRLConfig):
    """PPO hyperparameters (inherits the A2C knobs incl. trunk_lr)."""

    clip_eps: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gae_lambda: float = 0.95
    value_coef: float = 0.5


def compute_gae(
    rewards: list[float], values: list[float], gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    """GAE advantages and returns for one episode."""
    n = len(rewards)
    advantages = torch.zeros(n)
    last_adv = 0.0
    next_value = 0.0  # terminal value
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_value - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + torch.tensor(values)
    return advantages, returns


class PPOControlTrainer:
    """Fine-tunes the control policy with PPO (clipped surrogate + GAE)."""

    def __init__(
        self,
        policy: ControlPolicy,
        env,
        config: PPOConfig | None = None,
        resume: bool = False,
    ) -> None:
        self.policy = policy
        self.env = env
        self.config = config or PPOConfig()
        self.resume = resume
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Adam:
        """Same layout as the A2C trainer: heads at lr, trunk at trunk_lr."""
        heads: list[nn.Parameter] = list(self.policy.action_head.parameters())
        heads += list(self.policy.model.cortex.parameters())
        for block in self.policy.model.blocks:
            heads += list(block.router.parameters())

        groups: list[dict] = [{"params": heads, "lr": self.config.lr}]
        if self.config.trunk_lr > 0:
            trunk: list[nn.Parameter] = list(self.policy.adapter.parameters())
            trunk += list(self.policy.model.embedding.parameters())
            for block in self.policy.model.blocks:
                trunk += list(block.attention.parameters())
            self.policy.model.embedding.requires_grad_(True)
            for block in self.policy.model.blocks:
                block.attention.requires_grad_(True)
            groups.append({"params": trunk, "lr": self.config.trunk_lr})
        return torch.optim.Adam(groups)

    def _forward_policy(self, windows: Tensor) -> tuple[Tensor, Tensor]:
        """Action logits + critic values for a batch of windows (eval routing)."""
        out = self.policy.model(self.policy.adapter(windows))
        pooled = out.trunk.mean(dim=1)  # (N, d_model)
        logits = self.policy.action_head(pooled)
        return logits, out.cortex.score

    def _flatten(self, episodes: list[Episode]) -> dict[str, Tensor]:
        """Episodes -> one flat, GAE-augmented training batch."""
        windows = torch.cat([torch.cat(e.windows, dim=0) for e in episodes], dim=0)
        actions = torch.tensor([a for e in episodes for a in e.actions])

        with torch.no_grad():
            device = next(self.policy.parameters()).device
            windows_d = windows.to(device)
            logits, _ = self._forward_policy(windows_d)
            logp_old = torch.distributions.Categorical(
                logits=logits
            ).log_prob(actions.to(device))

        advantages, returns = zip(*[
            compute_gae(e.rewards, e.values, self.config.gamma,
                        self.config.gae_lambda)
            for e in episodes
        ])
        return {
            "windows": windows_d,
            "actions": actions.to(device),
            "logp_old": logp_old.detach(),
            "advantages": torch.cat(advantages).to(device),
            "returns": torch.cat(returns).to(device),
        }

    def update(self, step: int, episodes: list[Episode]) -> dict[str, float]:
        """One PPO iteration over the collected episodes."""
        batch = self._flatten(episodes)
        windows, actions = batch["windows"], batch["actions"]
        logp_old, returns = batch["logp_old"], batch["returns"]
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = windows.size(0)
        clip_total, entropy_total, batches = 0.0, 0.0, 0
        self.policy.model.eval()  # argmax routing, matching the rollout

        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(n, device=windows.device)
            for start in range(0, n, self.config.minibatch_size):
                idx = perm[start:start + self.config.minibatch_size]
                logits, values = self._forward_policy(windows[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])

                ratio = torch.exp(logp - logp_old[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps,
                                    1 + self.config.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    1.0,
                )
                self.optimizer.step()

                clip_total += float(
                    ((ratio - 1).abs() > self.config.clip_eps).float().mean()
                )
                entropy_total += float(entropy)
                batches += 1

        return {
            "return": sum(e.total_reward for e in episodes) / len(episodes),
            "route_frac": sum(
                sum(e.route_fracs) / max(1, len(e.route_fracs)) for e in episodes
            ) / len(episodes),
            "entropy": entropy_total / max(1, batches),
            "clip_frac": clip_total / max(1, batches),
            "beta": self.config.beta_end if self.resume else self.config.beta_start,
        }

    def train(self) -> list[dict[str, float]]:
        """Collect episodes then apply PPO updates, for `updates` rounds."""
        history: list[dict[str, float]] = []
        for step in range(self.config.updates):
            self.policy.eval()
            episodes = [
                run_episode(self.env, self.policy, seed=1000 + step * 10 + i)
                for i in range(self.config.episodes_per_update)
            ]
            metrics = self.update(step, episodes)
            history.append(metrics)
            if (step + 1) % self.config.log_every == 0:
                print(
                    f"update {step + 1:4d} | return {metrics['return']:+8.1f} "
                    f"| route {metrics['route_frac']:.3f} "
                    f"| entropy {metrics['entropy']:.3f} "
                    f"| clip {metrics['clip_frac']:.3f}",
                    flush=True,
                )
        return history
