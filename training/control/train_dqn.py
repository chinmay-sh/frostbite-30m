"""P6 Tier-2: Double DQN with experience replay (D23).

Off-policy control fine-tuning: the frozen trunk provides features, the D21
trunk action head provides Q-values, and only that head trains. Replay buys
many gradient steps per env decision — the sample-efficiency lever the
on-policy runs (A2C/PPO, D19–D22) could not reach. Because the trunk is
frozen and shared, the target network reduces to a copy of the head.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch import nn

from training.control.obs_adapter import ObsWindow
from training.control.policy_wrapper import ControlPolicy
from training.control.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    """Double DQN hyperparameters (a *decision* = one model step, incl. frame-skip)."""

    lr: float = 5e-4                 # Q-head LR (D24: 1e-4 was too slow to escape the idle attractor)
    trunk_lr: float = 0.0            # D24: >0 unfreezes adapter+embedding+attention (substrates stay frozen)
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.02            # D24: 0.05 kept re-crashing the learned policy
    eps_decay_decisions: int = 15_000
    warmup_decisions: int = 1_000  # random acting + no learning before this
    train_freq: int = 1            # gradient steps per decision
    target_sync_steps: int = 500   # legacy hard-sync interval (unused when tau > 0)
    tau: float = 0.005             # D32: Polyak soft-update rate; 0 = hard syncs only
    per_alpha: float = 0.6         # D32: prioritization strength (0 = uniform)
    per_beta_start: float = 0.4    # D32: IS-weight anneal start
    per_beta_end: float = 1.0      # D32: IS-weight anneal end
    action_repeat: int = 3
    episodes: int = 600
    solved_return: float = 200.0   # trailing-20 mean stop criterion
    log_every: int = 10


class DQNTrainer:
    """Double DQN over the frozen trunk; replay buffer + target head + eps-greedy."""

    def __init__(
        self,
        policy: ControlPolicy,
        env,
        config: DQNConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = policy.eval()
        self.target = copy.deepcopy(self.policy).eval()  # full copy (D24)
        self.env = env
        self.config = config or DQNConfig()
        self.seed = seed
        self.buffer = ReplayBuffer(
            self.config.buffer_size, seed=seed, alpha=self.config.per_alpha
        )
        self.rng = random.Random(seed)
        self.device = next(policy.parameters()).device
        self.optimizer = self._build_optimizer()
        self.decision_step = 0
        self.gradient_steps = 0

    def _build_optimizer(self) -> torch.optim.Adam:
        """Q-head at lr; with trunk_lr > 0, adapter+embedding+attention join (D24).

        Substrates (CfC) and routers never train here — CfC dynamics stay
        protected and the routing evidence stays banked.
        """
        groups: list[dict] = [
            {"params": self.policy.action_head.parameters(), "lr": self.config.lr}
        ]
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

    # -- acting --------------------------------------------------------------

    def epsilon_at(self, decision_step: int) -> float:
        """Linear eps decay from eps_start to eps_end over eps_decay_decisions."""
        c = self.config
        frac = min(1.0, decision_step / max(1, c.eps_decay_decisions))
        return c.eps_start + (c.eps_end - c.eps_start) * frac

    def _act(self, window_tensor: torch.Tensor, eps: float) -> int:
        """eps-greedy: a random action bypasses the model forward entirely."""
        if self.rng.random() < eps:
            return self.rng.randrange(4)
        with torch.no_grad():
            q = self.policy.action_head(self.policy.q_features(window_tensor))
        return int(q.argmax(dim=-1).item())

    def _run_episode(self, first: bool) -> tuple[float, int, float]:
        """One frame-skipped episode; learning happens per decision inline."""
        if first:
            obs, _ = self.env.reset(seed=self.seed)
        else:
            obs, _ = self.env.reset()
        window = ObsWindow(self.policy.window, self.policy.adapter.obs_dim)
        window.push(torch.as_tensor(obs, dtype=torch.float32))

        done = False
        total_reward = 0.0
        decisions = 0
        losses: list[float] = []
        while not done:
            before = window.tensor()
            action = self._act(before, self.epsilon_at(self.decision_step))

            reward_sum, done = 0.0, False
            for _ in range(self.config.action_repeat):
                obs, reward, terminated, truncated, _ = self.env.step(action)
                reward_sum += float(reward)
                done = terminated or truncated
                if done:
                    break
            total_reward += reward_sum

            window.push(torch.as_tensor(obs, dtype=torch.float32))
            self.buffer.push(
                before, action, reward_sum, window.tensor(), done
            )
            self.decision_step += 1
            decisions += 1

            if (
                self.decision_step > self.config.warmup_decisions
                and len(self.buffer) >= self.config.batch_size
            ):
                for _ in range(self.config.train_freq):
                    losses.append(self._gradient_step())
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        return total_reward, decisions, mean_loss

    # -- learning ------------------------------------------------------------

    def _gradient_step(self) -> float:
        """One Double-DQN update with PER sampling and importance weights.

        Online path computes Q WITH grads through the (possibly training)
        trunk; target path fully detached. Priorities refresh from |TD error|.
        """
        # Beta anneals from beta_start to beta_end across the training run.
        frac = min(1.0, self.gradient_steps / 40_000)
        beta = (
            self.config.per_beta_start
            + (self.config.per_beta_end - self.config.per_beta_start) * frac
        )
        batch, weights, indices = self.buffer.sample_prioritized(
            self.config.batch_size, self.device, beta=beta
        )

        with torch.no_grad():
            next_features = self.target.q_features(batch["next_windows"])
            best = self.policy.action_head(
                self.policy.q_features(batch["next_windows"])
            ).argmax(dim=-1)
            q_next = self.target.action_head(next_features).gather(
                1, best.unsqueeze(1)
            ).squeeze(1)
            target_q = batch["rewards"] + self.config.gamma * q_next * (
                1 - batch["dones"]
            )

        sensors = self.policy.adapter(batch["windows"])
        out = self.policy.model(sensors)  # grads flow through the trunk when training
        q_pred = self.policy.action_head(out.trunk.mean(dim=1)).gather(
            1, batch["actions"].unsqueeze(1)
        ).squeeze(1)
        td_error = q_pred - target_q
        # Importance weights rescale each sample's loss (PER bias correction).
        loss = (weights * nn.functional.smooth_l1_loss(
            q_pred, target_q, reduction="none"
        )).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]], 10.0
        )
        self.optimizer.step()
        self.buffer.update_priorities(indices, td_error)
        self.gradient_steps += 1
        self._update_target()
        return float(loss.detach().item())

    def _update_target(self) -> None:
        """Polyak soft update (D32): target <- tau*online + (1-tau)*target.

        Soft updates every step replace the hard 500-step sync whose staleness
        caused the v5 late-run regression (D30). tau=0 falls back to hard sync.
        """
        tau = self.config.tau
        if tau <= 0:  # legacy hard-sync path
            if self.gradient_steps % self.config.target_sync_steps == 0:
                self._sync_target()
            return
        with torch.no_grad():
            for online, target in zip(
                self.policy.parameters(), self.target.parameters()
            ):
                target.mul_(1 - tau).add_(online, alpha=tau)

    def _sync_target(self) -> None:
        """Copy the full online policy into the target (legacy hard sync)."""
        self.target.load_state_dict(self.policy.state_dict())

    # -- loop ------------------------------------------------------------------

    def train(self) -> tuple[list[dict[str, float]], bool]:
        """Run episodes until solved or the episode budget is spent."""
        history: list[dict[str, float]] = []
        recent: list[float] = []
        solved = False
        for episode in range(self.config.episodes):
            ret, decisions, loss = self._run_episode(first=(episode == 0))
            recent.append(ret)
            recent = recent[-20:]
            history.append(
                {
                    "episode": episode,
                    "return": ret,
                    "decisions": decisions,
                    "loss": loss,
                    "epsilon": self.epsilon_at(self.decision_step),
                    "buffer": len(self.buffer),
                    "grad_steps": self.gradient_steps,
                }
            )
            if (episode + 1) % self.config.log_every == 0:
                mean20 = sum(recent) / len(recent)
                print(
                    f"episode {episode + 1:4d} | return {ret:+8.1f} "
                    f"| trailing20 {mean20:+8.1f} | eps {history[-1]['epsilon']:.3f} "
                    f"| buffer {len(self.buffer):6d} | loss {loss:.4f}",
                    flush=True,
                )
                if len(recent) == 20 and mean20 >= self.config.solved_return:
                    print(f"SOLVED: trailing-20 mean {mean20:+.1f} "
                          f">= {self.config.solved_return}")
                    solved = True
                    break
        return history, solved
