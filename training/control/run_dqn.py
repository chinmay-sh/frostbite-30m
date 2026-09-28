"""P6 Tier-2 entrypoint: Double DQN control training (D23)."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import resolve_device, seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.train_dqn import DQNConfig, DQNTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 Double DQN control training")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--resume", default="runs/p6/control.pt")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", default="runs/p6/control_dqn.pt")
    args = parser.parse_args()

    device = resolve_device()
    seed_everything(args.seed)
    arch = ArchConfig.from_yaml(args.arch)
    model = FrostbiteModel(arch).to(device)
    payload = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    print(f"resumed trunk from {args.resume} (frozen feature extractor)")

    policy = ControlPolicy(model, obs_dim=8).to(device)
    policy.adapter.load_state_dict(payload["adapter"])
    # action_head starts fresh: Q-values, not the old action probabilities.

    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed)

    config = DQNConfig(
        episodes=args.episodes,
        action_repeat=args.action_repeat,
    )
    trainer = DQNTrainer(policy, env, config, seed=args.seed)
    print(f"frame-skip {config.action_repeat} | buffer {config.buffer_size:,} "
          f"| warmup {config.warmup_decisions} decisions | lr {config.lr}")

    history, solved = trainer.train()

    returns = [h["return"] for h in history]
    print(f"\nepisodes: {len(returns)} | solved: {solved}")
    print(f"first-10 mean return: {sum(returns[:10]) / min(10, len(returns)):+.1f}")
    print(f"last-10 mean return:  {sum(returns[-10:]) / min(10, len(returns)):+.1f}")
    print(f"env decisions: {trainer.decision_step:,} | gradient steps: "
          f"{trainer.gradient_steps:,}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "adapter": policy.adapter.state_dict(),
            "action_head": policy.action_head.state_dict(),
            "history": history,
            "solved": solved,
        },
        args.save,
    )
    print(f"saved {args.save}")
    env.close()


if __name__ == "__main__":
    main()
