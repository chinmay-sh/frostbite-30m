"""Real Phase-2 REINFORCE run: loads the Phase-1 checkpoint, trains routers
+cortex on TelemetryEnv, then evaluates routing behavior (Milestone 3)."""

from __future__ import annotations

import argparse

import torch

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import resolve_device, seed_everything
from training.environment import TelemetryEnv
from training.eval_routing import (
    RoutingBehaviorProbe,
    compute_monotonicity,
    print_report,
    route_monotonicity,
)
from training.phase2_reinforce import Phase2Trainer, RLConfig
from training.rewards import RewardConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 REINFORCE run")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--checkpoint", default="runs/phase1/final.pt")
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="runs/phase2/final.pt")
    args = parser.parse_args()

    device = resolve_device()
    seed_everything(args.seed)
    arch = ArchConfig.from_yaml(args.arch)
    model = FrostbiteModel(arch).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    print(f"loaded {args.checkpoint} (step {payload.get('global_step', '?')})")

    env = TelemetryEnv(
        seq_len=64, sensor_dim=arch.embedding.sensor_dim, seed=args.seed
    )
    trainer = Phase2Trainer(
        model,
        env,
        RewardConfig(alpha=1.0, beta=0.2),
        RLConfig(
            batch_size=args.batch_size,
            updates=args.updates,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
        ),
    )
    trainer.freeze_trunk()

    print("== before ==")
    before = RoutingBehaviorProbe(model, env, batch=8).report(batches=4)
    print_report("phase-1 checkpoint (pre-RL)", before)

    trainer.train()

    print("== after ==")
    after = RoutingBehaviorProbe(model, env, batch=8).report(batches=8)
    print_report(f"phase-2 (post-REINFORCE, {args.updates} updates)", after)

    monotonic = compute_monotonicity(after)
    print(f"\nP(ROUTE) monotonic in difficulty: {route_monotonicity(after)}")
    print(f"compute (halt depth + substrates) monotonic: {monotonic}")
    if monotonic:
        print("MILESTONE 3: router allocates CfC compute by dynamics complexity")

    from pathlib import Path

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "arch": args.arch},
        args.save,
    )
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
