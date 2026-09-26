"""Gumbel-Softmax straight-through sampler tests (P2.U2)."""

from __future__ import annotations

import pytest
import torch

from frostbite.modules.sampling import gumbel_sample


def test_forward_output_is_one_hot():
    torch.manual_seed(0)
    actions = gumbel_sample(torch.randn(4, 8, 3), tau=1.0)
    assert ((actions == 0) | (actions == 1)).all()
    assert (actions.sum(dim=-1) == 1).all()


def test_gradients_reach_logits():
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 3, requires_grad=True)
    actions = gumbel_sample(logits, tau=1.0)
    actions.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0


def test_straight_through_gradient_is_softmax_jacobian():
    """ST gradients flow through the soft branch: softmax Jacobian rows sum to 0.

    For loss = sum(actions @ w), d/dlogits = p_i*(w - sum_j p_j w_j), whose
    row-sum vanishes — a deterministic fingerprint of straight-through Gumbel.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 3, requires_grad=True)
    weight = torch.tensor([1.0, 0.0, 0.0])
    actions = gumbel_sample(logits, tau=1.0)
    (actions @ weight).sum().backward()
    assert torch.allclose(logits.grad.sum(dim=-1), torch.zeros(4, 8), atol=1e-5)
    # And the chosen action's logit still gets a nonzero gradient signal.
    assert logits.grad.abs().sum().item() > 0


def test_low_tau_approaches_argmax():
    torch.manual_seed(0)
    logits = torch.tensor([[[5.0, 0.0, 0.0]]] * 4)
    actions = gumbel_sample(logits, tau=0.01)
    assert (actions[..., 0] == 1).all()


def test_tau_must_be_positive():
    with pytest.raises(ValueError, match="tau"):
        gumbel_sample(torch.randn(2, 3, 3), tau=0.0)
