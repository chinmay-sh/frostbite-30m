"""Architecture config, loaded from configs/arch_30m.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    """Input projection settings."""

    sensor_dim: int
    max_len: int


@dataclass(frozen=True)
class AttentionConfig:
    """Multi-head self-attention settings."""

    causal: bool


@dataclass(frozen=True)
class CfcConfig:
    """Liquid CfC substrate settings."""

    units: int
    mode: str
    backbone_units: int
    backbone_layers: int
    mixed_memory: bool


@dataclass(frozen=True)
class RouterConfig:
    """Auto-RL micro-router settings."""

    hidden_dim: int


@dataclass(frozen=True)
class CortexConfig:
    """Global Laya-style cortex settings."""

    choice_dim: int
    hidden_dim: int


@dataclass(frozen=True)
class ArchConfig:
    """Top-level architecture settings for Frostbite-30M."""

    d_model: int
    n_blocks: int
    n_heads: int
    dropout: float
    param_cap: int
    embedding: EmbeddingConfig
    attention: AttentionConfig
    cfc: CfcConfig
    router: RouterConfig
    cortex: CortexConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArchConfig:
        """Load and validate an architecture config from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(
                d_model=raw["model"]["d_model"],
                n_blocks=raw["model"]["n_blocks"],
                n_heads=raw["model"]["n_heads"],
                dropout=raw["model"]["dropout"],
                param_cap=raw["model"]["param_cap"],
                embedding=EmbeddingConfig(**raw["embedding"]),
                attention=AttentionConfig(**raw["attention"]),
                cfc=CfcConfig(**raw["cfc"]),
                router=RouterConfig(**raw["router"]),
                cortex=CortexConfig(**raw["cortex"]),
            )
        except KeyError as err:
            raise ValueError(f"Missing key in {path}: {err}") from err
        except TypeError as err:
            # Unknown or missing fields in a sub-config.
            raise ValueError(f"Invalid config structure in {path}: {err}") from err
