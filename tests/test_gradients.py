"""Gradient-flow proofs through the masked routing chain (P2.U4).

These tests assemble the smallest full chain — embedding → attention → router
→ Gumbel ST → branch executor → loss — and assert gradients reach every
component, including the classic failure mode where an all-SKIP mask is
mistakenly implemented as `z * 0` (which kills router gradients).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from frostbite.modules.attention import SpatialAttention
from frostbite.modules.auto_rl_cell import AutoRLCell
from frostbite.modules.branch_executor import BranchExecutor
from frostbite.modules.cfc_substrate import CfCSubstrate
from frostbite.modules.embedding import TemporalEmbedding
from frostbite.modules.sampling import gumbel_sample


class _MiniFrostbite(nn.Module):
    """The smallest end-to-end chain: embed → attend → route → mix."""

    def __init__(self, config) -> None:
        super().__init__()
        self.embedding = TemporalEmbedding(config)
        self.attention = SpatialAttention(config)
        self.router = AutoRLCell(config)
        self.executor = BranchExecutor(CfCSubstrate(config))
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        h = self.embedding(x)
        z = self.attention(h)
        logits = self.router(z)
        actions = gumbel_sample(logits, tau=tau)
        mixed, _ = self.executor(z, actions)
        return self.head(mixed).squeeze(-1)


@pytest.fixture()
def model(arch_config) -> _MiniFrostbite:
    return _MiniFrostbite(arch_config)


def _grad_norms(model: _MiniFrostbite) -> dict[str, float]:
    return {
        name: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for name, p in model.named_parameters()
    }


def test_gradients_reach_every_component(model, arch_config):
    """AC: (a) attention, (b) CfC substrate, (c) router logits, (d) embedding."""
    x = torch.randn(2, 8, arch_config.embedding.sensor_dim)
    model(x, tau=1.0).sum().backward()
    norms = _grad_norms(model)

    for prefix in ("embedding", "attention", "router", "executor.substrate", "head"):
        group = {k: v for k, v in norms.items() if k.startswith(prefix)}
        assert group, f"no params under {prefix}"
        assert sum(group.values()) > 0, f"zero grad flow to {prefix}"


def test_skip_mask_does_not_zero_gradients(model, arch_config):
    """Regression: even when every token samples SKIP, the router must learn.

    Because SKIP keeps `z` (which the router also consumed), the router's
    logits still influence the loss through the mask itself. A broken
    implementation returning `z * 0` would leave the router gradient-free.
    """
    torch.manual_seed(3)
    x = torch.randn(2, 8, arch_config.embedding.sensor_dim)
    out = model(x, tau=1.0)
    out.sum().backward()
    router_grads = [p.grad for p in model.router.parameters()]
    assert all(g is not None for g in router_grads)
    assert sum(g.abs().sum().item() for g in router_grads) > 0


def test_high_tau_still_learns_router(model, arch_config):
    """Even at tau=5 the straight-through path delivers usable router grads."""
    x = torch.randn(2, 8, arch_config.embedding.sensor_dim)
    model(x, tau=5.0).sum().backward()
    assert sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.router.parameters()
    ) > 0


def test_masked_mixing_gradcheck(arch_config):
    """Deterministic numerical gradcheck of the masked mixing (double precision)."""
    torch.manual_seed(0)
    d = arch_config.d_model
    executor = BranchExecutor(CfCSubstrate(arch_config)).double().eval()

    z = torch.randn(1, 2, d, dtype=torch.float64, requires_grad=True)
    mask_logits = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)

    def f(z_in: torch.Tensor, m_in: torch.Tensor) -> torch.Tensor:
        actions = torch.softmax(m_in, dim=-1)  # deterministic soft actions
        out, _ = executor(z_in, actions)
        return (out * torch.linspace(-1, 1, d, dtype=torch.float64)).sum()

    assert torch.autograd.gradcheck(
        f, (z, mask_logits), eps=1e-6, atol=1e-4
    ), "masked mixing fails numerical gradcheck"
