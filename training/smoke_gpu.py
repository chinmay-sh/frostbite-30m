"""One-shot GPU smoke run for Phase-1 training (Milestone 2 evidence)."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from frostbite.config.arch import ArchConfig
from frostbite.config.train import TrainConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import seed_everything
from training.data import SyntheticTelemetry
from training.phase1_supervised import Phase1Trainer


def main() -> None:
    seed_everything(42)
    arch = ArchConfig.from_yaml("configs/arch_30m.yaml")
    tcfg = TrainConfig.from_yaml("configs/train_phase1.yaml")

    model = FrostbiteModel(arch)
    loader = DataLoader(
        SyntheticTelemetry(tcfg, "train", n_series=256, sensor_dim=arch.embedding.sensor_dim),
        batch_size=tcfg.data.batch_size,
    )
    trainer = Phase1Trainer(tcfg, model)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    first = trainer.train_epoch(loader, epoch=0, total_steps=2)
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    print(f"epoch0 : loss={first['loss']:.4f} route_frac={first['route_frac']:.3f} tau={first['tau']:.2f}")
    print(f"steps  : {trainer.global_step} | batch={tcfg.data.batch_size} seq={tcfg.data.seq_len}")
    print(f"device : {trainer.device} ({torch.cuda.get_device_name(0)})")
    print(f"peak VRAM: {peak_gib:.2f} GiB / 12 GiB budget | epoch wall: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
