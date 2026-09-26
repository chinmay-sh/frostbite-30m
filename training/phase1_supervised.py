"""Phase-1 supervised training loop with Gumbel-Softmax relaxation."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from frostbite.config.train import TrainConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import resolve_device, seed_everything


class Phase1Trainer:
    """Orchestrates Phase-1 supervised training of a FrostbiteModel."""

    def __init__(self, config: TrainConfig, model: FrostbiteModel) -> None:
        self.config = config
        self.model = model.to(resolve_device())
        self.device = next(model.parameters()).device
        self.optimizer = self._build_optimizer()
        self.global_step = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )

    def lr_at(self, step: int, total_steps: int) -> float:
        """Warmup then cosine decay to ~0."""
        warmup = self.config.optim.warmup_steps
        if step < warmup:
            return self.config.optim.lr * step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return self.config.optim.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def tau_at(self, epoch: int) -> float:
        """Linear anneal from tau_start to tau_end over `anneal_epochs`."""
        g = self.config.gumbel
        frac = min(1.0, epoch / max(1, g.anneal_epochs))
        return g.tau_start + (g.tau_end - g.tau_start) * frac

    def train_epoch(
        self, loader: DataLoader, epoch: int, total_steps: int
    ) -> dict[str, float]:
        """Train one epoch; returns mean loss and routing diagnostics."""
        self.model.train()
        tau = self.tau_at(epoch)
        losses: list[float] = []
        route_frac: list[float] = []

        amp_dtype = (
            torch.bfloat16 if self.config.amp.dtype == "bfloat16" else torch.float16
        )
        use_amp = self.config.amp.enabled and self.device.type == "cuda"

        for windows, targets in loader:
            windows = windows.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            lr = self.lr_at(self.global_step, total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast(self.device.type, dtype=amp_dtype, enabled=use_amp):
                out = self.model(windows, tau=tau)
                predicted = out.cortex.score  # scalar head as next-state proxy
                loss = nn.functional.mse_loss(predicted, targets.mean(dim=1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.grad_clip
                )
            self.optimizer.step()
            self.global_step += 1
            losses.append(loss.item())
            route_frac.append(
                out.route_probs[..., 0].mean().item()  # P(ROUTE) average
            )
            if (
                self.config.checkpoint.save_every_steps > 0
                and self.global_step % self.config.checkpoint.save_every_steps == 0
            ):
                self.save_checkpoint()

        return {
            "loss": sum(losses) / len(losses),
            "route_frac": sum(route_frac) / len(route_frac),
            "tau": tau,
            "lr": lr,
        }

    def fit(self, loader: DataLoader, epochs: int) -> list[dict[str, float]]:
        """Run the full training; returns per-epoch metrics."""
        return [
            self.train_epoch(loader, epoch, total_steps=epochs * len(loader))
            for epoch in range(epochs)
        ]

    def save_checkpoint(self) -> Path:
        """Persist model + optimizer + step to the configured directory."""
        path = Path(self.config.checkpoint.dir) / f"step_{self.global_step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model + optimizer + step from a checkpoint file."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = payload["global_step"]


def run(config_path: str, epochs: int = 1, n_series: int = 32) -> list[dict[str, float]]:
    """CLI entry: train Phase-1 on synthetic telemetry from a config YAML."""
    config = TrainConfig.from_yaml(config_path)
    from frostbite.config.arch import ArchConfig
    from training.data import make_loaders

    seed_everything(config.seed)
    arch = ArchConfig.from_yaml(Path(config_path).parent / "arch_30m.yaml")
    model = FrostbiteModel(arch)
    train_loader, _ = make_loaders(config, n_series=n_series)
    trainer = Phase1Trainer(config, model)
    return trainer.fit(train_loader, epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase-1 supervised training")
    parser.add_argument("--config", default="configs/train_phase1.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-series", type=int, default=32)
    args = parser.parse_args()
    metrics = run(args.config, args.epochs, args.n_series)
    for row in metrics:
        print(row)
