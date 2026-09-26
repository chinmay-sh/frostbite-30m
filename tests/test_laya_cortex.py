"""LayaCortex tests (P3.U2)."""

from __future__ import annotations

import torch

from frostbite.heads.laya_cortex import LayaCortex


def test_output_shapes(arch_config):
    cortex = LayaCortex(arch_config)
    x = torch.randn(2, 16, arch_config.d_model)
    out = cortex(x)
    assert out.choice.shape == (2, arch_config.cortex.choice_dim)
    assert out.score.shape == (2,)
    assert out.noul.shape == (2,)


def test_noul_in_unit_interval(arch_config):
    out = LayaCortex(arch_config)(torch.randn(4, 8, arch_config.d_model))
    assert (out.noul >= 0).all() and (out.noul <= 1).all()


def test_heads_are_independent(arch_config):
    """Perturbing one head's weights must not change the others' outputs."""
    cortex = LayaCortex(arch_config).eval()
    x = torch.randn(2, 8, arch_config.d_model)
    with torch.no_grad():
        before = cortex(x)
        cortex.score_head.weight.mul_(10.0)
        after = cortex(x)
    assert torch.equal(before.choice, after.choice)
    assert torch.equal(before.noul, after.noul)
    assert not torch.allclose(before.score, after.score)


def test_gradients_reach_all_heads_and_trunk(arch_config):
    cortex = LayaCortex(arch_config)
    out = cortex(torch.randn(2, 8, arch_config.d_model))
    (out.choice.sum() + out.score.sum() + out.noul.sum()).backward()
    for name, param in cortex.named_parameters():
        assert param.grad is not None and param.grad.abs().sum().item() > 0, name


def test_prod_cortex_budget():
    from pathlib import Path

    from frostbite.config.arch import ArchConfig
    from frostbite.utils import count_params

    config = ArchConfig.from_yaml(
        Path(__file__).parent.parent / "configs" / "arch_30m.yaml"
    )
    total = count_params(LayaCortex(config))
    # PLAN allocates 3.5M + 1.7M + 1.8M = 7.0M; our compact heads are far below.
    assert total <= 7_000_000, f"cortex {total:,} exceeds the 7.0M allocation"
