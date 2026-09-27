"""P6: PPO control fine-tuning entrypoint (options b+c, D21/D22)."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import resolve_device, seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.train_ppo import PPOConfig, PPOControlTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 PPO control fine-tuning")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--resume", default="runs/p6/control.pt")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--trunk-lr", type=float, default=1e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", default="runs/p6/control_ppo.pt")
    args = parser.parse_args()

    device = resolve_device()
    seed_everything(args.seed)
    arch = ArchConfig.from_yaml(args.arch)
    model = FrostbiteModel(arch).to(device)
    payload = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    print(f"resumed model from {args.resume}")

    policy = ControlPolicy(model, obs_dim=8).to(device)
    policy.adapter.load_state_dict(payload["adapter"])
    if "action_head" in payload:
        try:
            policy.action_head.load_state_dict(payload["action_head"])
        except RuntimeError:
            print("action_head shape changed (choice->trunk head); starting fresh head")

    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed)

    config = PPOConfig(
        episodes_per_update=args.episodes_per_update,
        updates=args.updates,
        trunk_lr=args.trunk_lr,
        entropy_coef=args.entropy_coef,
    )
    trainer = PPOControlTrainer(policy, env, config, resume=True)
    if config.trunk_lr > 0:
        print(f"stage-2: trunk unfrozen at lr {config.trunk_lr}")

    history = trainer.train()

    returns = [m["return"] for m in history]
    print(f"\nfirst-10 mean return: {sum(returns[:10]) / 10:+.1f}")
    print(f"last-10 mean return:  {sum(returns[-10:]) / 10:+.1f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "adapter": policy.adapter.state_dict(),
            "action_head": policy.action_head.state_dict(),
            "history": history,
        },
        args.save,
    )
    print(f"saved {args.save}")
    env.close()


if __name__ == "__main__":
    main()
