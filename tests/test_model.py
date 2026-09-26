"""FrostbiteModel full-assembly tests (P3.U3)."""

from __future__ import annotations

from pathlib import Path

import torch

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import count_params

CONFIGS = Path(__file__).parent.parent / "configs"


def test_forward_shapes(arch_config):
    model = FrostbiteModel(arch_config).eval()
    x = torch.randn(2, 8, arch_config.embedding.sensor_dim)
    out = model(x)
    assert out.cortex.choice.shape == (2, arch_config.cortex.choice_dim)
    assert out.cortex.score.shape == (2,)
    assert out.cortex.noul.shape == (2,)
    assert out.route_probs.shape == (
        arch_config.n_blocks, 2, 8, 3,
    )
    assert out.halt_layer.shape == (2,)


def test_eval_deterministic(arch_config):
    model = FrostbiteModel(arch_config).eval()
    x = torch.randn(1, 8, arch_config.embedding.sensor_dim)
    a = model(x)
    b = model(x)
    assert torch.equal(a.cortex.choice, b.cortex.choice)


def test_all_halt_short_circuits_stack(arch_config):
    """Biased routers: everything HALTs at block 0 -> stack exits early."""
    model = FrostbiteModel(arch_config).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.router.net[-1].bias.copy_(torch.tensor([-50.0, -50.0, 50.0]))
    x = torch.randn(2, 8, arch_config.embedding.sensor_dim)
    out = model(x)
    assert (out.halt_layer == 0).all()
    assert out.route_probs.shape[0] == 1  # only block 0 ran


def test_gradients_end_to_end(arch_config):
    model = FrostbiteModel(arch_config).train()
    x = torch.randn(2, 8, arch_config.embedding.sensor_dim)
    out = model(x, tau=1.0)
    target = torch.randn_like(out.cortex.score)
    loss = (out.cortex.score - target).pow(2).mean()
    loss.backward()
    for group in ("embedding", "blocks.0.attention", "blocks.0.router", "cortex"):
        total = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for n, p in model.named_parameters() if n.startswith(group)
        )
        assert total > 0, f"no gradient reached {group}"


def test_full_model_under_30m_cap():
    """The complete production model must stay under the hard 30M cap."""
    config = ArchConfig.from_yaml(CONFIGS / "arch_30m.yaml")
    model = FrostbiteModel(config)
    total = count_params(model)
    print(f"\nFULL MODEL: {total:,} params")
    assert total <= config.param_cap, f"{total:,} exceeds the 30M cap"


def test_forward_latency_baseline():
    """BS=1, seq=256 CPU forward as the P5 latency reference point."""
    import time

    config = ArchConfig.from_yaml(CONFIGS / "arch_30m.yaml")
    model = FrostbiteModel(config).eval()
    x = torch.randn(1, 256, config.embedding.sensor_dim)
    with torch.no_grad():
        model(x)  # warmup
        start = time.perf_counter()
        model(x)
        elapsed = time.perf_counter() - start
    print(f"\nlatency BS=1 seq=256 (cpu): {elapsed * 1000:.1f} ms")
    assert elapsed < 5.0, f"CPU forward unexpectedly slow: {elapsed:.2f}s"
