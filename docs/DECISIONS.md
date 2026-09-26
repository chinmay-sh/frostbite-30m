# Frostbite-30M — Decision Log

> **Rule:** Every meaningful project decision gets a row here on the day it's made.
> Format: one row, plain language, no essays. If a decision is reversed, don't delete it — add a new row that supersedes it.

| Date | ID | Decision | Reason |
| --- | --- | --- | --- |
| 2026-09-26 | D1 | Adopt phase/unit tracker (`docs/PROGRESS.md`) | PLAN.md is the spec; execution needs finer granularity |
| 2026-09-26 | D2 | uv + Python 3.13 baseline | Matches initialized venv; subject to Risk R1 in PROGRESS.md |
| 2026-09-26 | D3 | Decisions live in this file, not PROGRESS.md | Keep execution status and long-lived decisions separate |
| 2026-09-26 | D4 | Keep code simple & easy to understand; prefer an OOP architecture | Small maintainable codebase; each concept = one clear class |
| 2026-09-26 | D5 | Add `AGENTS.md` at repo root | Single entry point for agent/collaborator conventions |
| 2026-09-26 | D6 | Pin `ncps==1.0.1`, wrap it — never call `ncps.torch.CfC` directly outside `CfCSubstrate` | Library is thinly maintained (last release Aug 2024); wrapper isolates us from upstream quirks and keeps P1.U3 swappable |
| 2026-09-26 | D7 | **Known ncps CfC issues confirmed** (see below) | Verified empirically on RTX 3060 + source inspection (torch 2.11.0+cu128, ncps 1.0.1) |
| 2026-09-26 | D8 | torch from PyTorch cu128 index; bf16 AMP | RTX 3060 (Ampere) supports bf16; verified `is_bf16_supported()=True` |
| 2026-09-26 | D9 | `pyyaml` over `omegaconf` for configs | Fewer deps; typed frozen dataclasses + fail-fast loader is enough |
| 2026-09-26 | D10 | **One commit per unit** (`feat(P2.U3): ...`), status flip + code + decisions in the same commit; WIP sessions use `wip(P2.U3): ...` | Boring, reviewable history aligned with the tracker; formalized in `AGENTS.md` |
| 2026-09-26 | D11 | **Param budget re-calibrated vs PLAN §2.1 table.** Measured: embedding 295K (table: 8.2M — implies ~32K vocab, unreachable with sensor_dim=128); attention 263,680/blk (table: 1.6M/6 ≈ 266K ✓); CfC backbone widened 128→1344 → 2.07M/blk ≈ 12.4M (table: 12.5M ✓). Static stack total: **14.28M**, CfC-dominant, cap unchanged at 30M | PLAN's per-module table doesn't decompose arithmetic-wise for our sensor input; keep PLAN's *intent* (CfC-dominant, ≤30M) not its unreachable numbers; embedding slack becomes headroom for router/cortex |

### D7 detail — confirmed `ncps` CfC issues (2026-09-26)

1. **Time-gate sign inconsistency across backends.** `ncps/torch/cfc_cell.py` computes `t_interp = sigmoid(t_a*ts + t_b)` while `ncps/tf/cfc_cell.py` computes `sigmoid(-t_a*t + t_b)`. Verified empirically: the two conventions produce different outputs (not a no-op). Any reference implementation ported from the paper's TF code will diverge from the PyTorch path. → Our wrapper fixes ONE convention and documents it.
2. **Pure-Python time loop.** `CfC.forward` iterates `for t in range(seq_len)` per step — no fused/cuDNN RNN path. Thousands of tiny kernel launches per forward at batch 128 × seq 256 × 6 blocks. → Expect throughput, not correctness, problems in Phase-1; benchmark early (P3.U5).
3. **Silent shape handling.** Unbatched 2-D input is silently squeezed/expanded; hidden-state batch dim inferred from input. → `CfCSubstrate` must validate shapes explicitly (P1.U3).
4. **What's fine:** hidden-state carryover across chunked calls is exact (`full_pass ≈ cat(chunk_a, chunk_b)`, atol 1e-6) — BPTT chunking is safe. Gradients flow to cell weights on CUDA. Sign issue aside, forward/backward is numerically healthy.

## Template for new entries

```text
| YYYY-MM-DD | D<n> | <what was decided> | <why, one line> |
```
