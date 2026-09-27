"""P6.U4: evaluate the control agent — returns, ablations, phase telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch

from frostbite.model import FrostbiteModel
from training.control.policy_wrapper import ControlPolicy
from training.control.rollout import run_episode


@dataclass
class EvalResult:
    """Evaluation summary for one policy configuration."""

    name: str
    mean_return: float
    std_return: float
    landings: int  # episodes ending terminated (not truncated)
    episodes: int


def load_policy(arch_path: str, checkpoint_path: str, obs_dim: int = 8) -> ControlPolicy:
    """Rebuild a ControlPolicy from a P6 control checkpoint."""
    from frostbite.config.arch import ArchConfig

    arch = ArchConfig.from_yaml(arch_path)
    model = FrostbiteModel(arch)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    policy = ControlPolicy(model, obs_dim=obs_dim)
    if "adapter" in payload:
        policy.adapter.load_state_dict(payload["adapter"])
    if "action_head" in payload:
        policy.action_head.load_state_dict(payload["action_head"])
    return policy


def evaluate(policy: ControlPolicy, episodes: int = 20, seed_base: int = 5000) -> EvalResult:
    """Run eval episodes with fixed seeds; returns summary statistics."""
    env = gym.make("LunarLander-v3")
    env.action_space.seed(seed_base)
    policy.eval()

    returns, landings = [], 0
    for i in range(episodes):
        episode = run_episode(env, policy, seed=seed_base + i)
        returns.append(episode.total_reward)
        landings += int(episode.length < 900)  # truncation kicks in at 1000
    env.close()

    mean = sum(returns) / len(returns)
    std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5
    return EvalResult(policy.__class__.__name__, mean, std, landings, episodes)


def phase_telemetry(policy: ControlPolicy, episodes: int = 5) -> dict[str, dict[str, float]]:
    """Routing stats by flight phase: early (ascent) / mid / late (descent)."""
    env = gym.make("LunarLander-v3")
    env.action_space.seed(99)
    phases: dict[str, list[tuple[float, float]]] = {"early": [], "mid": [], "late": []}

    for i in range(episodes):
        episode = run_episode(env, policy, seed=200 + i)
        n = episode.length
        for t, route in enumerate(episode.route_fracs):
            phase = "early" if t < n / 3 else ("mid" if t < 2 * n / 3 else "late")
            phases[phase].append(route)
    env.close()

    return {
        name: {"mean_route_frac": sum(v) / len(v), "steps": len(v)}
        for name, v in phases.items()
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="P6.U4 evaluation")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--checkpoint", default="runs/p6/control.pt")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    policy = load_policy(args.arch, args.checkpoint)
    result = evaluate(policy, episodes=args.episodes)
    print(f"== {result.name} ==")
    print(f"return: {result.mean_return:+.1f} +/- {result.std_return:.1f} "
          f"over {result.episodes} episodes")
    print(f"non-truncated episodes: {result.landings}/{result.episodes}")

    telemetry = phase_telemetry(policy)
    print("\nper-phase routing (mean P(ROUTE)):")
    for phase, stats in telemetry.items():
        print(f"  {phase:<6} {stats['mean_route_frac']:.3f}  ({stats['steps']} steps)")


if __name__ == "__main__":
    main()
