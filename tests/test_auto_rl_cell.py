"""AutoRLCell tests (P2.U1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frostbite.config.arch import ArchConfig
from frostbite.modules.auto_rl_cell import AutoRLCell, RoutingAction
from frostbite.utils import count_params

CONFIGS = Path(__file__).parent.parent / "configs"


def test_logit_shape(arch_config):
    router = AutoRLCell(arch_config)
    z = torch.randn(2, 16, arch_config.d_model)
    logits = router(z)
    assert logits.shape == (2, 16, len(RoutingAction))


def test_logits_are_deterministic(arch_config):
    """The cell only produces scores; sampling lives outside (by design)."""
    router = AutoRLCell(arch_config).eval()
    z = torch.randn(2, 8, arch_config.d_model)
    with torch.no_grad():
        assert torch.equal(router(z), router(z))


def test_gradients_flow(arch_config):
    router = AutoRLCell(arch_config)
    z = torch.randn(2, 8, arch_config.d_model)
    router(z).sum().backward()
    for name, param in router.named_parameters():
        assert param.grad is not None and param.grad.abs().sum().item() > 0, name


def test_param_formula(arch_config):
    d, h = arch_config.d_model, arch_config.router.hidden_dim
    expected = d * h + h + h * len(RoutingAction) + len(RoutingAction)
    assert count_params(AutoRLCell(arch_config)) == expected


def test_prod_router_budget():
    """6 routers must stay near PLAN's ~0.5M allocation (d=320 design: ~0.62M)."""
    config = ArchConfig.from_yaml(CONFIGS / "arch_30m.yaml")
    per_block = count_params(AutoRLCell(config))
    total = per_block * config.n_blocks
    assert total <= 700_000, f"router stack {total:,} exceeds the ~0.5M ballpark"


def test_rejects_2d_input(arch_config):
    with pytest.raises(ValueError, match="3-D"):
        AutoRLCell(arch_config)(torch.randn(8, arch_config.d_model))
