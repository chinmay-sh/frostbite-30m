"""CLI entry point (`uv run frostbite` / `python -m frostbite`)."""

from __future__ import annotations

import argparse

import torch

import frostbite
from frostbite.config import ArchConfig
from frostbite.utils import count_params, resolve_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frostbite", description="Frostbite-30M CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Show environment and package info")
    info.add_argument("--config", default="configs/arch_30m.yaml")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    args = _build_parser().parse_args(argv)

    if args.command == "info":
        device = resolve_device()
        print(f"frostbite-30m v{frostbite.__version__}")
        print(f"device: {device}")
        if device.type == "cuda":
            print(f"gpu: {torch.cuda.get_device_name(0)}")
        config = ArchConfig.from_yaml(args.config)
        print(f"config: d_model={config.d_model} blocks={config.n_blocks} "
              f"heads={config.n_heads} cap={config.param_cap:,}")


if __name__ == "__main__":
    main()
