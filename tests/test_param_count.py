"""Parameter budget enforcement (P1.U4, PLAN Milestone 1).

Asserts the hard 30M cap and per-module approximate budgets from
`configs/arch_30m.yaml`. The static stack test covers the P1 milestone
(attention + CfC, no router); the full-model test activates in P3.U3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from frostbite.config.arch import ArchConfig
from frostbite.modules.attention import SpatialAttention
from frostbite.modules.cfc_substrate import CfCSubstrate
from frostbite.modules.embedding import TemporalEmbedding
from frostbite.utils import count_params

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def prod_config() -> ArchConfig:
    """Production architecture config (d_model=256, 6 blocks)."""
    return ArchConfig.from_yaml(CONFIGS / "arch_30m.yaml")


class StaticStack(nn.Module):
    """P1 milestone assembly: embedding + n_blocks x (attention + CfC)."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.embedding = TemporalEmbedding(config)
        self.blocks = nn.ModuleList(
            [SpatialAttention(config) for _ in range(config.n_blocks)]
        )
        self.substrates = nn.ModuleList(
            [CfCSubstrate(config) for _ in range(config.n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(x)
        for attention, substrate in zip(self.blocks, self.substrates):
            hidden = attention(hidden)
            hidden, _ = substrate(hidden)
        return hidden


def test_static_stack_budgets(prod_config):
    """Per-module and total parameter budgets for the P1 static stack."""
    import yaml

    budgets = yaml.safe_load((CONFIGS / "arch_30m.yaml").read_text())["budgets"]
    stack = StaticStack(prod_config)

    emb = count_params(stack.embedding)
    att = count_params(stack.blocks[0])
    cfc = count_params(stack.substrates[0])
    total = count_params(stack)

    assert emb <= budgets["embedding"], f"embedding {emb:,} > {budgets['embedding']:,}"
    assert att <= budgets["attention_per_block"], f"attention {att:,} > {budgets['attention_per_block']:,}"
    assert cfc <= budgets["cfc_per_block"], f"cfc {cfc:,} > {budgets['cfc_per_block']:,}"
    assert total <= prod_config.param_cap, f"total {total:,} > cap {prod_config.param_cap:,}"

    # Milestone 1 evidence: CfC-dominant static stack within the 30M envelope.
    print(f"\nembedding={emb:,} attention/blk={att:,} cfc/blk={cfc:,} TOTAL={total:,}")


def test_static_stack_forward(prod_config):
    """Static stack runs a forward pass end to end (tiny batch on CPU)."""
    stack = StaticStack(prod_config).eval()
    x = torch.randn(1, 8, prod_config.embedding.sensor_dim)
    with torch.no_grad():
        out = stack(x)
    assert out.shape == (1, 8, prod_config.d_model)
