"""Per-episode breakdown of a control checkpoint eval (fixed seeds)."""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym

from training.control.evaluate import load_policy
from training.control.rollout import run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--checkpoint", default="runs/p6/control_dqn_v4.pt")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=5000)
    args = parser.parse_args()

    policy = load_policy(args.arch, args.checkpoint)
    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed_base)

    returns = []
    for i in range(args.episodes):
        ep = run_episode(env, policy, seed=args.seed_base + i)
        returns.append((i, ep.total_reward, ep.length))
        marker = " <== near-landing!" if ep.total_reward > -50 else ""
        print(f"episode {i:2d} | return {ep.total_reward:+8.1f} | {ep.length:3d} decisions{marker}")

    values = [r for _, r, _ in returns]
    print(f"\nmean {statistics.mean(values):+.1f} | median {statistics.median(values):+.1f} "
          f"| best {max(values):+.1f} | worst {min(values):+.1f}")
    print(f"episodes better than -100: {sum(1 for v in values if v > -100)}/{len(values)}")
    env.close()


if __name__ == "__main__":
    main()
