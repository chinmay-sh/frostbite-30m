"""ReinforcedLiquidBlock tests (P3.U1)."""

from __future__ import annotations

import torch

from frostbite.blocks.reinforced_layer import ReinforcedLiquidBlock
from frostbite.modules.routing_state import RoutingState


def _fresh_state(config, batch=2, seq=8, device=torch.device("cpu")) -> RoutingState:
    return RoutingState.fresh(batch, seq, config.d_model, device)


def test_block_output_shape_and_state(arch_config):
    block = ReinforcedLiquidBlock(arch_config).eval()
    x = torch.randn(2, 8, arch_config.d_model)
    out = block(x, _fresh_state(arch_config))
    assert out.output.shape == x.shape
    assert out.logits.shape == (2, 8, 3)
    assert not out.state.all_halted()


def test_eval_is_deterministic(arch_config):
    """Argmax routing: two eval forwards give identical outputs."""
    block = ReinforcedLiquidBlock(arch_config).eval()
    x = torch.randn(2, 8, arch_config.d_model)
    state = _fresh_state(arch_config)
    a = block(x, state)
    b = block(x, state)
    assert torch.equal(a.output, b.output)


def test_halt_updates_state(arch_config):
    """Biased router -> every token halts -> state records it."""
    block = ReinforcedLiquidBlock(arch_config).eval()
    with torch.no_grad():
        block.router.net[-1].bias.copy_(
            torch.tensor([-50.0, -50.0, 50.0])  # HALT dominates
        )
    out = block(torch.randn(2, 8, arch_config.d_model), _fresh_state(arch_config))
    assert out.state.all_halted()


def _mse_loss(output: torch.Tensor) -> torch.Tensor:
    """A loss with a genuinely nonzero gradient (output.sum() is ~0 for a
    LayerNorm with γ=1, β=0 — its per-token sum is identically zero)."""
    torch.manual_seed(123)
    target = torch.randn_like(output)
    return (output - target).pow(2).mean()


def test_gradients_flow_through_block(arch_config):
    """With tokens guaranteed to sample ROUTE, every component receives grads."""
    block = ReinforcedLiquidBlock(arch_config).train()
    with torch.no_grad():  # bias the router so ROUTE dominates every token
        block.router.net[-1].bias.copy_(torch.tensor([50.0, -50.0, -50.0]))

    x = torch.randn(2, 8, arch_config.d_model, requires_grad=True)
    out = block(x, _fresh_state(arch_config), tau=1.0)
    _mse_loss(out.output).backward()
    for prefix in ("attention", "router", "executor", "norm"):
        grads = [p.grad for n, p in block.named_parameters() if n.startswith(prefix)]
        assert all(g is not None for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0, prefix
    assert x.grad is not None and x.grad.abs().sum().item() > 0


def test_skip_only_still_trains_router_and_attention(arch_config):
    """All-SKIP: substrate gets no grads (nothing routed), router still learns."""
    block = ReinforcedLiquidBlock(arch_config).train()
    with torch.no_grad():  # bias the router so SKIP dominates every token
        block.router.net[-1].bias.copy_(torch.tensor([-50.0, 50.0, -50.0]))

    out = block(
        torch.randn(2, 8, arch_config.d_model), _fresh_state(arch_config), tau=1.0
    )
    _mse_loss(out.output).backward()

    router_total = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for n, p in block.named_parameters() if n.startswith("router")
    )
    attention_total = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for n, p in block.named_parameters() if n.startswith("attention")
    )
    substrate_total = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for n, p in block.named_parameters() if n.startswith("executor")
    )
    assert router_total > 0, "router must learn even when everything skips"
    assert attention_total > 0
    assert substrate_total == 0.0, "nothing routed -> no substrate grads (by design)"
