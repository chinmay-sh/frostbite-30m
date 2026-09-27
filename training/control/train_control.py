"""P6.U3: actor-critic control fine-tuning on LunarLander."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from frostbite.utils import seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.rollout import Episode, run_episode
from training.rewards import RewardConfig


@dataclass
class ControlRLConfig:
    """Hyperparameters of the control fine-tuning loop."""

    lr: float = 3e-4
    trunk_lr: float = 0.0  # stage-2 unfreeze: 0 = trunk frozen (P6 plan)
    gamma: float = 0.99
    entropy_coef: float = 0.002  # 0.01 kept the policy near-uniform (D20)
    episodes_per_update: int = 4
    updates: int = 250
    beta_start: float = 0.2
    beta_end: float = 0.02
    beta_anneal: float = 0.8  # fraction of updates over which beta decays
    log_every: int = 10


def returns_to_go(rewards: list[float], gamma: float) -> Tensor:
    """Discounted reward-to-go for one episode: R_t = r_t + gamma * R_{t+1}."""
    out = torch.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


class ControlTrainer:
    """Fine-tunes the control policy with episodic actor-critic REINFORCE."""

    def __init__(
        self,
        policy: ControlPolicy,
        env,
        config: ControlRLConfig | None = None,
        resume: bool = False,
    ) -> None:
        self.policy = policy
        self.env = env
        self.config = config or ControlRLConfig()
        self.resume = resume  # D19: resumed runs continue beta at the floor
        self.optimizer = self._build_optimizer()
        self.reward_config = RewardConfig(beta=self.config.beta_start)

    def _build_optimizer(self) -> torch.optim.Adam:
        """Two groups: policy heads at full LR; trunk at trunk_lr (0 = frozen).

        Stage-2 unfreeze (P6 plan): embedding + attention adapt at low LR;
        substrates stay frozen regardless — CfC dynamics are protected.
        """
        heads: list[nn.Parameter] = list(self.policy.action_head.parameters())
        heads += list(self.policy.model.cortex.parameters())
        for block in self.policy.model.blocks:
            heads += list(block.router.parameters())

        trunk: list[nn.Parameter] = list(self.policy.adapter.parameters())
        trunk += list(self.policy.model.embedding.parameters())
        for block in self.policy.model.blocks:
            trunk += list(block.attention.parameters())

        groups: list[dict] = [{"params": heads, "lr": self.config.lr}]
        if self.config.trunk_lr > 0:
            self.policy.model.embedding.requires_grad_(True)
            for block in self.policy.model.blocks:
                block.attention.requires_grad_(True)
            groups.append({"params": trunk, "lr": self.config.trunk_lr})
        return torch.optim.Adam(groups)

    def beta_at(self, update: int) -> float:
        """Anneal beta linearly from beta_start to beta_end (R8 mitigation)."""
        c = self.config
        frac = min(1.0, update / max(1, int(c.beta_anneal * c.updates)))
        return c.beta_start + (c.beta_end - c.beta_start) * frac

    @property
    def beta_floor(self) -> float:
        """Terminal beta value — resuming runs continue from here (D19)."""
        return self.config.beta_end

    def _episode_loss(self, episode: Episode, env_reward: float) -> tuple[Tensor, float, float]:
        """Recompute log-probs with grad; advantage = return-to-go - V(s).

        The model stays in eval mode so routing is argmax-deterministic —
        identical to the routing used during rollout (on-policy consistency).
        eval() does not block gradients; only no_grad would.
        """
        windows = torch.cat(episode.windows, dim=0)  # (T, W, obs)
        device = next(self.policy.parameters()).device
        sensors = self.policy.adapter(windows.to(device))
        out = self.policy.model(sensors)

        logits = self.policy.action_head(out.trunk.mean(dim=1))  # D21 trunk head
        actions = torch.tensor(episode.actions, device=device)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        values = out.cortex.score  # (T,) critic
        returns = returns_to_go(episode.rewards, self.config.gamma).to(device)
        raw_advantage = returns - values.detach()
        # Standardize the ADVANTAGE (not returns): noisy short episodes must
        # not get their noise amplified by a small returns.std().
        advantage = (raw_advantage - raw_advantage.mean()) / (
            raw_advantage.std() + 1e-8
        )

        policy_loss = -(log_probs * advantage).mean()
        # Standardize targets before the critic MSE: raw LunarLander returns
        # (~-200) would make value_loss ~4e4 and drown the policy gradient.
        value_loss = nn.functional.mse_loss(
            values, (returns - returns.mean()) / (returns.std() + 1e-8)
        )
        entropy = dist.entropy().mean()

        # Intrinsic compute reward, beta-annealed (R8): mean over blocks.
        route_frac = (out.route_probs[..., 0]).mean().item()
        intrinsic = 0.1 * (1.0 - route_frac) - 0.1 * route_frac
        loss = (
            policy_loss
            + 0.5 * value_loss
            - self.config.entropy_coef * entropy
            - self.reward_config.beta * intrinsic
        )
        return loss, route_frac, float(entropy.item())

    def update(self, step: int) -> dict[str, float]:
        """Collect episodes and apply one gradient update.

        Backprops per episode (not stacked): each episode's autograd graph
        (up to 1000 steps through the 29.4M model) is freed immediately —
        stacking 4 full graphs grows memory until OOM (crashed at ~update 110).
        """
        if self.resume:
            # D19: a resumed run already annealed beta — continue at the floor.
            self.reward_config = RewardConfig(beta=self.beta_floor)
        else:
            self.reward_config = RewardConfig(beta=self.beta_at(step))
        self.policy.eval()

        episodes = [
            run_episode(self.env, self.policy, seed=1000 + step * 10 + i)
            for i in range(self.config.episodes_per_update)
        ]
        env_returns = [e.total_reward for e in episodes]

        self.optimizer.zero_grad(set_to_none=True)
        route_fracs, entropies = [], []
        for episode in episodes:
            loss, route_frac, entropy = self._episode_loss(
                episode, episode.total_reward
            )
            (loss / len(episodes)).backward()  # accumulate the episode mean
            route_fracs.append(route_frac)
            entropies.append(entropy)
        nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]], 1.0
        )
        self.optimizer.step()

        return {
            "return": sum(env_returns) / len(env_returns),
            "route_frac": sum(route_fracs) / len(route_fracs),
            "entropy": sum(entropies) / len(entropies),
            "beta": self.reward_config.beta,
        }

    def train(self) -> list[dict[str, float]]:
        """Run the full fine-tuning; prints progress periodically."""
        history = []
        for step in range(self.config.updates):
            metrics = self.update(step)
            history.append(metrics)
            if (step + 1) % self.config.log_every == 0:
                print(
                    f"update {step + 1:4d} | return {metrics['return']:+8.1f} "
                    f"| route {metrics['route_frac']:.3f} "
                    f"| entropy {metrics['entropy']:.3f} "
                    f"| beta {metrics['beta']:.3f}",
                    flush=True,
                )
        return history
