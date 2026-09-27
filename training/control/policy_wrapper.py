"""Policy wrapper: cortex heads -> control action, value, confidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from frostbite.model import FrostbiteModel
from training.control.obs_adapter import ObsAdapter


@dataclass
class ControlDecision:
    """One control step's outputs for acting and learning."""

    action: int
    log_prob: Tensor  # scalar log-probability of the chosen action
    value: Tensor  # scalar V(s) from the Score head
    confidence: Tensor  # scalar in [0, 1] from the Noul head
    route_frac: float  # mean P(ROUTE) across blocks (telemetry)


class ControlPolicy(nn.Module):
    """Wraps a FrostbiteModel + ObsAdapter into a Gymnasium-compatible agent.

    The action head reads the final TRUNK representation (mean-pooled over
    the window), not the 8-dim choice bottleneck — D21: routing the policy
    through choice logits gave the policy gradient too little signal.
    """

    def __init__(self, model: FrostbiteModel, obs_dim: int, window: int = 32) -> None:
        super().__init__()
        self.model = model
        self.window = window
        self.adapter = ObsAdapter(obs_dim, model.embedding.input_proj.in_features, window)
        self.action_head = nn.Linear(model.embedding.input_proj.out_features, 4)

    @torch.no_grad()
    def act(self, window_tensor: Tensor) -> ControlDecision:
        """Sample an action for one padded window (1, T, obs_dim)."""
        device = next(self.parameters()).device
        sensors = self.adapter(window_tensor.to(device))
        out = self.model(sensors)
        pooled = out.trunk.mean(dim=1)  # (B, d_model) trunk representation
        logits = self.action_head(pooled)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return ControlDecision(
            action=int(action.item()),
            log_prob=dist.log_prob(action),
            value=out.cortex.score[0],
            confidence=out.cortex.noul[0],
            route_frac=out.route_probs[..., 0].mean().item(),
        )

    def parameters_for_training(self) -> list[nn.Parameter]:
        """Trainable set: action head + routers + cortex (trunk frozen)."""
        params = list(self.action_head.parameters())
        params += list(self.model.cortex.parameters())
        for block in self.model.blocks:
            params += list(block.router.parameters())
        return params
