"""P6.U3 entrypoint: fine-tune Frostbite on LunarLander from the warm-start ckpt."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import resolve_device, seed_everything
from training.control.policy_wrapper import ControlPolicy
from training.control.train_control import ControlRLConfig, ControlTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="P6.U3 control fine-tuning")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--warm", default="runs/p6/warm.pt")
    parser.add_argument("--resume", default=None, help="control checkpoint to continue from")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--save", default="runs/p6/control.pt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = resolve_device()
    seed_everything(args.seed)
    arch = ArchConfig.from_yaml(args.arch)
    model = FrostbiteModel(arch).to(device)

    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        print(f"resumed model from {args.resume}")
    else:
        payload = torch.load(args.warm, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        print(f"loaded warm-start from {args.warm} "
              f"(loss {payload['first_loss']:.5f} -> {payload['last_loss']:.5f})")

    policy = ControlPolicy(model, obs_dim=8).to(device)
    policy.adapter.load_state_dict(payload["adapter"])
    if args.resume:
        policy.action_head.load_state_dict(payload["action_head"])

    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed)

    config = ControlRLConfig(
        episodes_per_update=args.episodes_per_update,
        updates=args.updates,
    )
    trainer = ControlTrainer(policy, env, config)
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
            "resumed_from": args.resume,
        },
        args.save,
    )
    print(f"saved {args.save}")
    env.close()


if __name__ == "__main__":
    main()
