"""CfCSubstrate tests (P1.U3)."""

from __future__ import annotations

import pytest
import torch

from frostbite.modules.cfc_substrate import CfCSubstrate


def test_output_shape_and_hidden(arch_config):
    substrate = CfCSubstrate(arch_config).eval()
    x = torch.randn(2, 16, arch_config.d_model)
    out, hx = substrate(x)
    assert out.shape == (2, 16, arch_config.d_model)
    assert hx.shape == (2, arch_config.cfc.units)


def test_state_carryover_matches_full_pass(arch_config):
    """Chunked BPTT: (full pass) == (chunk A then chunk B with warm start)."""
    substrate = CfCSubstrate(arch_config).eval()
    x = torch.randn(2, 8, arch_config.d_model)
    with torch.no_grad():
        full, _ = substrate(x)
        first, hx = substrate(x[:, :4])
        second, _ = substrate(x[:, 4:], hx)
    joined = torch.cat([first, second], dim=1)
    assert torch.allclose(full, joined, atol=1e-5)


def test_gradient_flow(arch_config):
    substrate = CfCSubstrate(arch_config)
    x = torch.randn(2, 8, arch_config.d_model)
    out, _ = substrate(x)
    out.sum().backward()
    grads = [p.grad for p in substrate.cfc.rnn_cell.ff1.parameters()]
    assert all(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_rejects_2d_input(arch_config):
    substrate = CfCSubstrate(arch_config)
    with pytest.raises(ValueError, match="3-D"):
        substrate(torch.randn(8, arch_config.d_model))


def test_cfc_params_scale(arch_config):
    """Sanity: backbone width drives param count (matters for the 12.5M budget)."""
    from frostbite.utils import count_params
    from frostbite.config.arch import CfcConfig

    wide = CfCSubstrate(arch_config)
    assert count_params(wide) > 0

    # Same config but wider backbone must have strictly more params.
    bigger = arch_config.__class__(
        **{**arch_config.__dict__,
           "cfc": CfcConfig(units=64, mode="default", backbone_units=128,
                            backbone_layers=1, mixed_memory=False)})
    wide2 = CfCSubstrate(bigger)
    assert count_params(wide2) > count_params(wide)
