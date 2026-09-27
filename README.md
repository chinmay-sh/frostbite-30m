# Frostbite-30M

Hybrid 29.8M-parameter edge model: Self-Attention + CfC liquid dynamics + internal RL routing (Route / Skip / Halt), trainable on a single RTX 3060 12 GB.

## Documentation

- `docs/PLAN.md` — architecture specification & implementation plan (the *what*)
- `docs/PROGRESS.md` — phase/unit execution tracker with acceptance criteria (the *how far*)
- `docs/DECISIONS.md` — append-only log of every project decision (the *why*)
- `docs/RUNBOOK.md` — manual runbook: every training/eval command, end to end
- `AGENTS.md` — working conventions for agents & contributors (read before writing code)

## Status

| Phase | Focus | State |
| --- | --- | --- |
| P0–P3 | Scaffolding → substrates → routing → full model + Phase-1 training | ✅ (Milestones 1 & 2) |
| P4 | REINFORCE routing on TelemetryEnv | ✅ **Milestone 3** — compute allocation tracks dynamics difficulty |
| P5 | Edge deployment (ONNX + Go) | ⏸ deferred (D18) |🟨 U1–U2 ✅, U3 fine-tuning run in progress
| P6 | Real-world control fine-tuning (Gymnasium LunarLander-v3) | ⬜ planned — see `docs/PROGRESS.md` §Phase 6 |

Current model: **29.4M params** (d_model=320, 6 blocks, 4 heads, CfC backbone 2242), trains in <5 GiB VRAM.

Always update `docs/PROGRESS.md` at the end of a work session, and log every decision in `docs/DECISIONS.md` (see `AGENTS.md` for the rules).

## Project layout

```text
configs/      # arch_30m.yaml, train_phase1.yaml (created in P0.U2)
frostbite/    # model package: modules/, blocks/, heads/ (P0.U3)
training/     # phase1_supervised.py, phase2_reinforce.py, rewards.py
deployment/   # export_onnx.py, go-inference/
tests/        # test_param_count.py, test_gradients.py
```

## Quickstart

```bash
uv sync                       # install deps from uv.lock
uv run pytest                 # run test suite (~1 min, CPU)
uv run python -m frostbite info   # GPU + config sanity check
```

For training pipelines, evaluation, ablations, and troubleshooting see **`docs/RUNBOOK.md`** — the manual runbook with every command in execution order.
