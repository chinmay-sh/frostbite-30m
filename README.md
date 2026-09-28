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
| P5 | Edge deployment (ONNX + Go) | ⏸ deferred (D18) |
| P6 | Real-world control fine-tuning (Gymnasium LunarLander-v3) | ✅ ablations + landing demo (see Results) |

Current model: **29.4M params** (d_model=320, 6 blocks, 4 heads, CfC backbone 2242), trains in <5 GiB VRAM.

## Results

### Adaptive compute (core thesis)

**Difficulty-monotonic routing on telemetry (Milestone 3)** — after Phase-2 REINFORCE, halt depth 2.36 → 4.70 → 4.98 and substrate usage 0.12 → 0.12 → 1.00 across static/oscillatory/chaotic segments: the router spends CfC compute only when dynamics are complex.

**Routing ablations in real control (LunarLander, same trunk, 20 episodes):**

| Routing | Return | Note |
| --- | --- | --- |
| **Learned adaptive** | **−116.0 ± 51** | the deployed policy |
| Forced SKIP-only | −602.9 ± 432 | no CfC anywhere |
| Forced ROUTE-all | −677.8 ± 360 | full compute everywhere |

Neither compute extreme flies — **the learned per-token mixture is the only working configuration**, and its compute budget shifts across flight phases (P(ROUTE) 0.41/0.31/0.35 early/mid/late).

### Control (LunarLander-v3)

Fine-tuned as a control agent via warm-start + Double-DQN (replay, frame-skip 3, potential-based shaping during training only; evaluation always on the raw env). Best stable checkpoint (`runs/p6/control_dqn_v4.pt`, 25 eval episodes): **median −97.5, best +17.6 (clean landing), 14/25 episodes better than −100** — the lander descends under control and occasionally lands. A follow-up run reached returns up to **+76.8** in training but regressed late-run (D30: known DQN consolidation issue) — the gap to reliable landings is consistency, not capability.

![Landing demo](docs/assets/landing.gif)

*(Deterministic greedy episode, seed 5017, raw environment. Regenerate: `uv run python training/control/render_demo.py`.)*

### Training profile

| Metric | Value |
| --- | --- |
| Phase-1 supervised loss | 0.0045 → 0.0007 (40 epochs) |
| Peak VRAM (batch 128, seq 256, bf16) | 4.87 GiB of 12 GiB |
| CPU latency (BS=1, seq=256) | ~398 ms |

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
