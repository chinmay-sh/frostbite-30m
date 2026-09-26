"""Masked control-flow tests: BranchExecutor + RoutingState (P2.U3)."""

from __future__ import annotations

import torch

from frostbite.modules.auto_rl_cell import RoutingAction
from frostbite.modules.branch_executor import BranchExecutor
from frostbite.modules.cfc_substrate import CfCSubstrate
from frostbite.modules.routing_state import RoutingState


def _one_hot(action: RoutingAction, batch: int = 2, seq: int = 8) -> torch.Tensor:
    actions = torch.zeros(batch, seq, 3)
    actions[..., int(action)] = 1.0
    return actions


def test_all_route_equals_substrate_output(arch_config):
    substrate = CfCSubstrate(arch_config).eval()
    executor = BranchExecutor(substrate).eval()
    z = torch.randn(2, 8, arch_config.d_model)
    out, _ = executor(z, _one_hot(RoutingAction.ROUTE))
    expected, _ = substrate(z)
    assert torch.allclose(out, expected, atol=1e-6)


def test_all_skip_is_identity(arch_config):
    executor = BranchExecutor(CfCSubstrate(arch_config)).eval()
    z = torch.randn(2, 8, arch_config.d_model)
    out, _ = executor(z, _one_hot(RoutingAction.SKIP))
    assert torch.equal(out, z)


def test_eval_skips_substrate_when_nobody_routes(arch_config):
    """True conditional execution: all-SKIP at eval must not touch the substrate."""
    executor = BranchExecutor(CfCSubstrate(arch_config)).eval()
    z = torch.randn(2, 8, arch_config.d_model)
    with _SubstrateSpy(executor.substrate) as spy:
        executor(z, _one_hot(RoutingAction.SKIP))
    assert spy.calls == 0


def test_train_mode_always_runs_substrate(arch_config):
    """Training computes all branches — required for graph-safe mixing."""
    executor = BranchExecutor(CfCSubstrate(arch_config)).train()
    z = torch.randn(2, 8, arch_config.d_model)
    with _SubstrateSpy(executor.substrate) as spy:
        executor(z, _one_hot(RoutingAction.SKIP))
    assert spy.calls == 1


def test_train_eval_parity_with_deterministic_actions(arch_config):
    """AC: identical outputs in both modes when actions are argmax one-hot."""
    torch.manual_seed(7)
    executor = BranchExecutor(CfCSubstrate(arch_config)).eval()
    z = torch.randn(2, 8, arch_config.d_model)
    mixed = [executor(z, _one_hot(a))[0] for a in RoutingAction]
    executor.train()
    branchy = [executor(z, _one_hot(a))[0] for a in RoutingAction]
    for a, b in zip(mixed, branchy):
        assert torch.allclose(a, b, atol=1e-6)


def test_halted_tokens_freeze_across_blocks(arch_config):
    """A token that HALTs keeps its block-1 representation after block 2."""
    state = RoutingState.fresh(1, 4, arch_config.d_model, torch.device("cpu"))
    block1_out = torch.randn(1, 4, arch_config.d_model)
    actions = torch.zeros(1, 4, 3)
    actions[:, 2, int(RoutingAction.HALT)] = 1.0  # token 2 halts

    state = state.update(block1_out, actions)
    block2_out = block1_out + 100.0  # would massively change token 2

    resolved = state.resolve(block2_out)
    assert torch.equal(resolved[:, 2], block1_out[:, 2])  # frozen
    assert torch.equal(resolved[:, 0], block2_out[:, 0])  # active flows on


def test_all_halted_short_circuit(arch_config):
    state = RoutingState.fresh(2, 4, arch_config.d_model, torch.device("cpu"))
    actions = torch.zeros(2, 4, 3)
    actions[..., int(RoutingAction.HALT)] = 1.0
    state = state.update(torch.randn(2, 4, arch_config.d_model), actions)
    assert state.all_halted()
    assert not RoutingState.fresh(2, 4, 4, torch.device("cpu")).all_halted()


class _SubstrateSpy:
    """Context manager counting substrate forward calls."""

    def __init__(self, substrate: CfCSubstrate) -> None:
        self.substrate = substrate
        self.calls = 0
        self._original = None

    def _counting_forward(self, *args, **kwargs):
        self.calls += 1
        return self._original(*args, **kwargs)

    def __enter__(self) -> _SubstrateSpy:
        self._original = self.substrate.forward
        self.substrate.forward = self._counting_forward  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        self.substrate.forward = self._original  # type: ignore[method-assign]
