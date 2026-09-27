"""P6.U2: warm-start the trunk on env dynamics (next-obs prediction)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import count_params, resolve_device, seed_everything
from training.control.collect import collect_random_trajectories
from training.control.obs_adapter import ObsAdapter


class WarmStartTrainer:
    """Fits adapter + full trunk on next-observation prediction (MSE)."""

    def __init__(self, model: FrostbiteModel, adapter: ObsAdapter, lr: float = 1e-3) -> None:
        self.device = next(model.parameters()).device
        self.model = model.train()
        self.adapter = adapter.train()
        params = list(adapter.parameters()) + list(model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.criteria = nn.MSELoss()

    def epoch(self, loader: DataLoader) -> float:
        """One pass of window->next-obs regression; returns mean loss."""
        losses = []
        for windows, targets in loader:
            windows = windows.to(self.device)
            targets = targets.to(self.device)
            sensors = self.adapter(windows)
            out = self.model(sensors)
            # Score head (B,) predicts the mean next observation.
            loss = self.criteria(out.cortex.score, targets.mean(dim=-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="P6.U2 dynamics warm-start")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--checkpoint", default="runs/phase1/final.pt")
    parser.add_argument("--save", default="runs/p6/warm.pt")
    args = parser.parse_args()

    device = resolve_device()
    seed_everything(0)
    arch = ArchConfig.from_yaml(args.arch)
    model = FrostbiteModel(arch).to(device)
    if Path(args.checkpoint).exists():
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        print(f"loaded trunk from {args.checkpoint}")

    dataset = collect_random_trajectories(
        n_episodes=args.episodes, window=32, seed=123
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    adapter = ObsAdapter(8, arch.embedding.sensor_dim, window=32).to(device)
    trainer = WarmStartTrainer(model, adapter)

    first = trainer.epoch(loader)
    for i in range(args.epochs - 1):
        loss = trainer.epoch(loader)
        print(f"epoch {i + 1}: loss {loss:.5f}", flush=True)
    print(f"warm-start: {first:.5f} -> {loss:.5f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "adapter": adapter.state_dict(),
            "first_loss": first,
            "last_loss": loss,
        },
        args.save,
    )
    total = count_params(model) + count_params(adapter) + 4 * arch.cortex.choice_dim + 4
    print(f"saved {args.save} | total params incl. control additions: {total:,}")


if __name__ == "__main__":
    main()
