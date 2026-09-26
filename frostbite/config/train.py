"""Training config, loaded from configs/train_phase1.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline settings."""

    seq_len: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and LR schedule settings."""

    optimizer: str
    lr: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int
    schedule: str


@dataclass(frozen=True)
class GumbelConfig:
    """Gumbel-Softmax annealing settings."""

    tau_start: float
    tau_end: float
    anneal_epochs: int


@dataclass(frozen=True)
class AmpConfig:
    """Mixed-precision settings."""

    enabled: bool
    dtype: str

    @property
    def torch_dtype(self) -> str:
        """AMP dtype as a torch-compatible string."""
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError(f"Unsupported AMP dtype: {self.dtype}")
        return self.dtype


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpointing settings."""

    dir: str
    save_every_steps: int
    resume: str | None


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training settings for a single phase."""

    run_name: str
    seed: int
    data: DataConfig
    optim: OptimConfig
    gumbel: GumbelConfig
    amp: AmpConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load and validate a training config from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(
                run_name=raw["run_name"],
                seed=raw["seed"],
                data=DataConfig(**raw["data"]),
                optim=OptimConfig(**raw["optim"]),
                gumbel=GumbelConfig(**raw["gumbel"]),
                amp=AmpConfig(**raw["amp"]),
                checkpoint=CheckpointConfig(**raw["checkpoint"]),
            )
        except KeyError as err:
            raise ValueError(f"Missing key in {path}: {err}") from err
        except TypeError as err:
            raise ValueError(f"Invalid config structure in {path}: {err}") from err
