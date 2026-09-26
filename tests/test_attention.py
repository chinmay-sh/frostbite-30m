"""SpatialAttention tests (P1.U2)."""

from __future__ import annotations

import pytest
import torch

from frostbite.modules.attention import SpatialAttention


def test_output_shape(arch_config):
    attention = SpatialAttention(arch_config).eval()
    x = torch.randn(2, 16, arch_config.d_model)
    assert attention(x).shape == x.shape


def test_residual_connection_present(arch_config):
    """With all-heads output zeroed, output must equal the input (residual)."""
    attention = SpatialAttention(arch_config).eval()
    x = torch.randn(2, 8, arch_config.d_model)
    with torch.no_grad():
        attention.proj.weight.zero_()
        attention.proj.bias.zero_()
        # Zero weights of Q,K so attention scores become uniform (value path still
        # active but proj=0 kills it), output should be exactly the residual.
        assert torch.allclose(attention(x), x, atol=1e-6)


def test_causal_mask_blocks_future(arch_config):
    """Changing a future token must not change earlier outputs."""
    attention = SpatialAttention(arch_config).eval()
    x = torch.randn(1, 12, arch_config.d_model)
    with torch.no_grad():
        base = attention(x)
        x_future_changed = x.clone()
        x_future_changed[:, 8:] += 10.0  # perturb the future
        changed = attention(x_future_changed)
    assert torch.allclose(base[:, :8], changed[:, :8], atol=1e-5)
    assert not torch.allclose(base[:, 8:], changed[:, 8:], atol=1e-4)


def test_rejects_bad_head_config():
    from frostbite.config.arch import (
        ArchConfig,
        AttentionConfig,
        CfcConfig,
        CortexConfig,
        EmbeddingConfig,
        RouterConfig,
    )

    bad = ArchConfig(
        d_model=66,
        n_blocks=2,
        n_heads=4,
        dropout=0.0,
        param_cap=30_000_000,
        embedding=EmbeddingConfig(sensor_dim=32, max_len=128),
        attention=AttentionConfig(causal=True),
        cfc=CfcConfig(units=64, mode="default", backbone_units=32,
                      backbone_layers=1, mixed_memory=False),
        router=RouterConfig(hidden_dim=32),
        cortex=CortexConfig(choice_dim=8, hidden_dim=64),
    )
    with pytest.raises(ValueError, match="divisible"):
        SpatialAttention(bad)


def test_gradients_flow(arch_config):
    attention = SpatialAttention(arch_config)
    x = torch.randn(2, 8, arch_config.d_model, requires_grad=True)
    attention(x).sum().backward()
    for name, param in attention.named_parameters():
        assert param.grad is not None, f"no grad: {name}"
        assert param.grad.abs().sum().item() > 0, f"zero grad: {name}"


def test_param_count_matches_formula(arch_config):
    from frostbite.utils import count_params

    d = arch_config.d_model
    attention = SpatialAttention(arch_config)
    # LN (2d) + qkv (3d² + 3d) + proj (d² + d)
    expected = 2 * d + 3 * d * d + 3 * d + d * d + d
    assert count_params(attention) == expected
