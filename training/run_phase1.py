"""Real Phase-1 training run: produces the frozen checkpoint for Phase-2 RL."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from frostbite.config.arch import ArchConfig
from frostbite.config.train import TrainConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import count_params, seed_everything
from training.data import SyntheticTelemetry
from training.phase1_supervised import Phase1Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 supervised run")
    parser.add_argument("--config", default="configs/train_phase1.yaml")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-series", type=int, default=512)
    parser.add_argument("--out", default="runs/phase1/final.pt")
    args = parser.parse_args()

    seed_everything(42)
    arch = ArchConfig.from_yaml(args.arch)
    tcfg = TrainConfig.from_yaml(args.config)
    model = FrostbiteModel(arch)
    print(f"model: {count_params(model):,} params")

    loader = DataLoader(
        SyntheticTelemetry(tcfg, "train", n_series=args.n_series,
                           sensor_dim=arch.embedding.sensor_dim),
        batch_size=tcfg.data.batch_size,
        shuffle=True,
    )
    trainer = Phase1Trainer(tcfg, model)
    total_steps = args.epochs * len(loader)

    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(loader, epoch=epoch, total_steps=total_steps)
        print(f"epoch {epoch:3d} | loss {metrics['loss']:.5f} "
              f"| route {metrics['route_frac']:.3f} | tau {metrics['tau']:.2f} "
              f"| lr {metrics['lr']:.2e}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "arch": args.arch, "global_step": trainer.global_step},
        out,
    )
    print(f"saved {out} (step {trainer.global_step})")


if __name__ == "__main__":
    main()
