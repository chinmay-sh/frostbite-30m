"""Masked Route/Skip/Halt branch execution around the CfC substrate."""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.modules.auto_rl_cell import RoutingAction
from frostbite.modules.cfc_substrate import CfCSubstrate


class BranchExecutor(nn.Module):
    """Applies routing decisions around a CfC substrate, graph-safely.

    Training: the substrate always runs and branches are mixed with one-hot
    action masks — no `if/else` on tensors, so autograd never breaks.
    Evaluation: when *no* token chooses ROUTE the substrate call is skipped
    entirely (true conditional execution with exact masked-mixing parity).

    Per-token skipping *inside* the recurrent substrate is impossible (the
    CfC recurrence couples all timesteps); HALT savings come from freezing
    tokens and short-circuiting the block stack (see RoutingState).
    """

    def __init__(self, substrate: CfCSubstrate) -> None:
        super().__init__()
        self.substrate = substrate

    def forward(
        self, z: Tensor, actions: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Mix branches: ROUTE takes the substrate output, SKIP/HALT keep ``z``.

        :param z: attention output (B, T, d_model).
        :param actions: one-hot actions (B, T, 3).
        :param hidden: optional warm-start hidden state for the substrate.
        :returns: (output, next_hidden).
        """
        route_mask = actions[..., int(RoutingAction.ROUTE)]
        if not self.training and bool((route_mask == 0).all()):
            return z, hidden  # nobody routes: skip the heavy branch entirely

        route_out, next_hidden = self.substrate(z, hidden)
        mask = route_mask.unsqueeze(-1)
        output = mask * route_out + (1.0 - mask) * z
        return output, next_hidden
