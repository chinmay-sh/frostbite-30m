"""Liquid time-dynamics substrate wrapping the ncps CfC cell.

Isolation layer (decision D6/D7): ``ncps.torch.CfC`` is never used outside this
module. The wrapper pins our conventions:
- strict 3-D (B, T, d_model) input validation (ncps silently reshapes otherwise),
- explicit hidden-state carryover for chunked BPTT,
- output projection back to d_model when CfC units differ.
"""

from __future__ import annotations

from torch import Tensor, nn

from frostbite.config.arch import ArchConfig


class CfCSubstrate(nn.Module):
    """Liquid time-dynamics layer wrapped around an ncps CfC cell."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        from ncps.torch import CfC  # local import: ncps confined to this module

        cfc = config.cfc
        self.units = cfc.units
        self.cfc = CfC(
            input_size=config.d_model,
            units=cfc.units,
            batch_first=True,
            mode=cfc.mode,
            backbone_units=cfc.backbone_units,
            backbone_layers=cfc.backbone_layers,
            mixed_memory=cfc.mixed_memory,
        )
        self.out_proj = (
            nn.Linear(cfc.units, config.d_model)
            if cfc.units != config.d_model
            else nn.Identity()
        )

    def forward(
        self, x: Tensor, hidden: Tensor | tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        """Run liquid dynamics over (B, T, d_model); returns (output, next_hidden)."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D input (B, T, d_model), got {x.dim()}-D")
        out, hx = self.cfc(x, hidden)
        return self.out_proj(out), hx
