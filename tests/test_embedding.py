"""TemporalEmbedding tests (P1.U1)."""

from __future__ import annotations

import pytest
import torch

from frostbite.modules.embedding import TemporalEmbedding


def test_output_shape(arch_config):
    embedding = TemporalEmbedding(arch_config)
    x = torch.randn(2, 16, arch_config.embedding.sensor_dim)
    out = embedding(x)
    assert out.shape == (2, 16, arch_config.d_model)


def test_gradient_flows_to_projection(arch_config):
    embedding = TemporalEmbedding(arch_config)
    x = torch.randn(2, 16, arch_config.embedding.sensor_dim)
    embedding(x).sum().backward()
    assert embedding.input_proj.weight.grad is not None
    assert embedding.input_proj.weight.grad.abs().sum().item() > 0


def test_gradient_flows_to_positional_encoding(arch_config):
    embedding = TemporalEmbedding(arch_config)
    x = torch.randn(2, 16, arch_config.embedding.sensor_dim)
    embedding(x).sum().backward()
    assert embedding.pos_embedding.grad is not None
    assert embedding.pos_embedding.grad[:16].abs().sum().item() > 0


def test_rejects_2d_input(arch_config):
    embedding = TemporalEmbedding(arch_config)
    with pytest.raises(ValueError, match="3-D"):
        embedding(torch.randn(16, arch_config.embedding.sensor_dim))


def test_rejects_overlong_sequence(arch_config):
    embedding = TemporalEmbedding(arch_config)
    too_long = arch_config.embedding.max_len + 1
    x = torch.randn(1, too_long, arch_config.embedding.sensor_dim)
    with pytest.raises(ValueError, match="max_len"):
        embedding(x)


def test_param_budget_small(arch_config):
    from frostbite.utils import count_params

    embedding = TemporalEmbedding(arch_config)
    # proj 32*64 + 64 bias + pos 128*64 = 8,768
    assert count_params(embedding) == 32 * 64 + 64 + 128 * 64
