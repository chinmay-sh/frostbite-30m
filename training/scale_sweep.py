"""One-off scale sweep: compare three ~29.4M capacity allocations.

A: d=256, 6 blocks, fat backbone (current D13)
B: d=320, 6 blocks, balanced backbone (units=d_model)
C: d=256, 8 blocks, deep stack

Deleted after the decision is recorded (D14).
"""

from __future__ import annotations

import dataclasses
import time

import torch
from torch.utils.data import DataLoader

from frostbite.config.arch import (
    AttentionConfig,
    ArchConfig,
    CfcConfig,
    CortexConfig,
    EmbeddingConfig,
    RouterConfig,
)
from frostbite.config.train import TrainConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import count_params, seed_everything
from training.data import SyntheticTelemetry
from training.phase1_supervised import Phase1Trainer

TARGET = 29_400_000


def _config(d_model: int, n_blocks: int) -> ArchConfig:
    return ArchConfig(
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=4,
        dropout=0.1,
        param_cap=30_000_000,
        embedding=EmbeddingConfig(sensor_dim=128, max_len=1024),
        attention=AttentionConfig(causal=True),
        cfc=CfcConfig(units=d_model, mode="default", backbone_units=1,
                      backbone_layers=1, mixed_memory=False),
        router=RouterConfig(hidden_dim=320),
        cortex=CortexConfig(choice_dim=8, hidden_dim=256),
    )


def solve_backbone(base: ArchConfig) -> ArchConfig:
    """Pick backbone width so the full model lands exactly on TARGET."""
    d, u, n = base.d_model, base.cfc.units, base.n_blocks
    coef = n * (d + 5 * u + 1)  # params are linear in backbone width
    probe = count_params(FrostbiteModel(base))
    width = round((TARGET - (probe - coef)) / coef)
    return dataclasses.replace(
        base, cfc=dataclasses.replace(base.cfc, backbone_units=width)
    )


def measure(name: str, config: ArchConfig, tcfg: TrainConfig) -> dict:
    seed_everything(42)
    model = FrostbiteModel(config)
    params = count_params(model)

    loader = DataLoader(
        SyntheticTelemetry(tcfg, "train", n_series=512, sensor_dim=128),
        batch_size=tcfg.data.batch_size,
    )
    trainer = Phase1Trainer(tcfg, model)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metrics = [trainer.train_epoch(loader, epoch=e, total_steps=12) for e in range(3)]
    wall = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    model.eval()
    x = torch.randn(1, 256, 128, device=trainer.device)
    with torch.no_grad():
        model(x)
        t0 = time.perf_counter()
        model(x)
        cpu_ms = time.perf_counter() - t0

    print(
        f"{name}: params={params:,} backbone={config.cfc.backbone_units} "
        f"| VRAM={peak_gib:.2f} GiB ms/step={wall / trainer.global_step * 1000:.0f} "
        f"| loss {metrics[0]['loss']:.4f}->{metrics[-1]['loss']:.4f} "
        f"| GPU-latency BS1/seq256={cpu_ms * 1000:.0f} ms"
    )
    return {"name": name, "params": params, "loss": metrics[-1]["loss"]}


def main() -> None:
    tcfg = TrainConfig.from_yaml("configs/train_phase1.yaml")
    measure("A d=256 blk=6 (current)", solve_backbone(_config(256, 6)), tcfg)
    measure("B d=320 blk=6          ", solve_backbone(_config(320, 6)), tcfg)
    measure("C d=256 blk=8          ", solve_backbone(_config(256, 8)), tcfg)


if __name__ == "__main__":
    main()
