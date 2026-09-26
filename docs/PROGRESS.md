# Frostbite-30M — Progress Tracker & Execution Guide

> **Source of truth:** `docs/PLAN.md` (architecture spec). This file tracks *execution*.
> **Rule:** Work unit-by-unit. A unit is only ✅ when its **Acceptance Criteria (AC)** all pass. Update this file at the end of every work session.
> **Decisions:** All project decisions are tracked separately in **`docs/DECISIONS.md`** — append a row there on the day a decision is made.

---

## 0. How to read this tracker

| Symbol | Meaning |
| --- | --- |
| ⬜ | Not started |
| 🟨 | In progress |
| ✅ | Done — all ACs verified |
| ⏸ | Blocked (note why in the unit) |
| ↩ | Needs rework (regression found) |

**Unit ID convention:** `P<phase>.U<unit>` (e.g., `P2.U3`). **One commit per unit** — when a unit's ACs pass, flip its status here and commit together with the code: `feat(P2.U3): masked halt control flow` (see `AGENTS.md`).

**Global state:** Phase 0 complete · Last updated: 2026-09-26

| Phase | Title | Units | Status | Depends on |
| --- | --- | --- | --- | --- |
| P0 | Environment & Scaffolding | 4 | ✅ | — |
| P1 | Core Substrates (Attention, CfC, Embedding) | 4 | ⬜ | P0 |
| P2 | Auto-RL Cell & Dynamic Routing | 4 | ⬜ | P1 |
| P3 | Full Assembly & Phase-1 Training | 5 | ⬜ | P2 |
| P4 | Phase-2 Reinforcement Learning | 4 | ⬜ | P3 |
| P5 | Edge Deployment (ONNX + Go) | 3 | ⬜ | P4 |

---

## PHASE 0 — Environment & Scaffolding
*Goal: a reproducible dev environment and skeleton every later phase plugs into. No model code yet.*

### P0.U1 — Dependency & toolchain setup ✅
- [x] Deps added via `uv`: `torch==2.11.0+cu128` (PyTorch cu128 index), `ncps==1.0.1`, `numpy`, `pyyaml`, dev: `pytest` (`onnx`/`onnxruntime` deferred to P5 — no need earlier)
- [x] `cuda.is_available()=True`, GPU = RTX 3060, `bf16 supported = True`
- [x] `ncps` CfC forward+backward on CUDA verified; gradients reach cell weights
- [x] **ncps CfC audit done — issues confirmed, logged as D7** (sign inconsistency, python time-loop, silent shapes; carryover exact — see `docs/DECISIONS.md`)

**AC:** ✅ met. R1 (Python 3.13 wheels) did not materialize — torch 2.11 and ncps 1.0.1 both installed cleanly on 3.13.

### P0.U2 — Configuration system ✅
- [x] `configs/arch_30m.yaml`: d_model=256, n_heads=4, n_blocks=6, param_cap, CfC/router/cortex dims
- [x] `configs/train_phase1.yaml`: batch=128, bf16 AMP, LR schedule, Gumbel anneal
- [x] `frostbite/config/`: frozen dataclasses (`ArchConfig`, `TrainConfig`) + fail-fast YAML loader

**AC:** ✅ met — covered by `tests/test_config.py` (loads real configs; rejects unknown fields & missing keys).

### P0.U3 — Package skeleton ✅
- [x] Tree per PLAN §5: `frostbite/{modules,blocks,heads,config}`, `training/`, `deployment/`, `tests/`, `configs/`
- [x] All stubs with docstrings + `NotImplementedError`, each class in its named file
- [x] `python -m frostbite` / `uv run frostbite info` CLI wired; placeholder `main.py` deleted

**AC:** ✅ met — `uv run pytest` collects & passes with zero import errors.

### P0.U4 — Reproducibility & test harness ✅
- [x] `frostbite/utils.py`: `seed_everything`, `resolve_device`, `count_params` (trainable-only)
- [x] `tests/conftest.py`: autouse deterministic seeding + CPU-fallback device fixture
- [x] pytest config in `pyproject.toml` (`testpaths`, `pythonpath`)

**AC:** ✅ met — `tests/test_utils.py`: seeding reproducibility, param counting incl. frozen-param exclusion. **7/7 tests green.**

---

## PHASE 1 — Core Substrates (Sprint 1)
*Goal: the static, fully-differentiable components — attention, CfC liquid layer, embedding — hitting the parameter budget.*

### P1.U1 — Temporal Embedding Engine ⬜
- [ ] `frostbite/modules/embedding.py`: linear projection (sensor-dim → 256) + positional/time encoding for time-series
- [ ] Unit test: output shape `(B, T, 256)`; gradient flows to input projection

**AC:** Shape + grad tests pass. Params ≈ 8.2M budget (±10%).

### P1.U2 — Multi-Head Self-Attention ⬜
- [ ] `frostbite/modules/attention.py`: 4 heads, d=256, pre-norm + residual, optional causal mask
- [ ] Unit tests: shape correctness; causal mask enforced (future positions leak-checked); params/head counted

**AC:** Attention block params ≈ 1.6M/6 ≈ 265K per block. Gradient check passes.

### P1.U3 — CfC Continuous Substrate ⬜
- [ ] `frostbite/modules/cfc_substrate.py`: wrap `ncps torch CfC` as an FFN-replacement module (input proj → CfC cell → output proj)
- [ ] Handle sequence/time-axis semantics + hidden-state carryover across BPTT windows
- [ ] Unit tests: stateful warm start (hidden state continuity), shape `(B, T, 256)`, params ≈ 12.5M/6 per block

**AC:** CfC forward+backward on GPU; hidden-state persistence across chunks verified; per-block param budget met.

### P1.U4 — Parameter budget harness ⬜
- [ ] `tests/test_param_count.py`: total ≤ 30M hard cap; per-module budget table asserted from `arch_30m.yaml`
- [ ] `uv run pytest tests/test_param_count.py` wired into pre-commit / CI habit

**AC:** **PLAN Milestone 1** — assembled static stack (6× Attention+CfC blocks, no router) reports ~29.8M and the test enforces the cap.

---

## PHASE 2 — Auto-RL Cell & Dynamic Routing (Sprint 2, part 1)
*Goal: the MicroRouter and differentiable Route/Skip/Halt control flow.*

### P2.U1 — MicroRouter (Auto-RL cell) ⬜
- [ ] `frostbite/modules/auto_rl_cell.py`: 2-layer MLP → 3 logits (`ROUTE`/`SKIP`/`HALT`), ~0.5M total across 6 blocks
- [ ] Returns logits (not sampled) — sampling strategy injected by caller

**AC:** Logit shape `(B, T, 3)`; per-block params within budget; no sampling logic inside the module.

### P2.U2 — Gumbel-Softmax straight-through estimator ⬜
- [ ] Differentiable `gumbel_sample(logits, tau, hard=True)` utility; temperature `tau` schedulable via config
- [ ] Property test: straight-through path delivers gradients to router logits while forward output is one-hot

**AC:** Finite-gradient assertion on logits through the masked mixing path.

### P2.U3 — Masked control flow (train) & branchy execution (eval) ⬜
- [ ] Training path: compute all three branches, mix via one-hot masks (no `if/else` — graph-safe)
- [ ] Inference path: true conditional execution (`torch.jit`/early-exit friendly) — HALT exits the block stack
- [ ] HALT semantics: aggregate per-token halting into a layer-stack exit signal; pass representation to Cortex

**AC:** Train mode and eval mode produce identical outputs when router is forced argmax-deterministic. **PLAN Milestone 2 precondition.**

### P2.U4 — Gradient-flow verification ⬜
- [ ] `tests/test_gradients.py`: gradients reach (a) attention, (b) CfC substrate, (c) router logits, (d) embedding — through the masked mixture
- [ ] Regression test for the classic failure: SKIP-mask ≠ zeroing gradients on Z

**AC:** All gradient tests green under `torch.autograd.gradcheck`-style finite-difference spot checks.

---

## PHASE 3 — Full Assembly & Phase-1 Training (Sprint 2, part 2)
*Goal: the complete Frostbite-30M model + supervised Gumbel training loop running end to end.*

### P3.U1 — Reinforced Liquid Block assembly ⬜
- [ ] `frostbite/blocks/reinforced_layer.py`: Attention → Router → {CfC | residual | halt} composition, residual + norm
- [ ] Config-driven; logs routing probabilities as diagnostics (TensorBoard/W&B optional)

**AC:** Single block forward/backward within param budget; routing stats logged per step.

### P3.U2 — Global Laya-Style Cortex ⬜
- [ ] `frostbite/heads/laya_cortex.py`: Choice head (softmax routing, 3.5M), Score head (ordinal scalar, 1.7M), Noul head (sigmoid confidence, 1.8M)
- [ ] Shared trunk optional but within budget; heads independently testable

**AC:** Three heads' combined params ≈ 7.0M; shape/type tests for each output.

### P3.U3 — FrostbiteModel full assembly ⬜
- [ ] `frostbite/model.py`: Embedding → 6× Reinforced Blocks → Cortex; HALT short-circuit honored at sequence level
- [ ] Re-run `tests/test_param_count.py` on the *full* model → ~29.8M cap holds
- [ ] Forward pass latency benchmark (BS=1, seq=256) recorded as baseline

**AC:** Full-model forward+backward under mixed precision; total params ≈ 29.8M.

### P3.U4 — Data pipeline & toy tasks ⬜
- [ ] Synthetic time-series dataset (next-state prediction) + optional CartPole-trajectory offline dataset
- [ ] DataLoader with chunked BPTT windows; deterministic splits

**AC:** Batch of shape `(128, T, sensor_dim)` loads on GPU in < 50 ms (pinned memory).

### P3.U5 — Phase-1 supervised training loop ⬜
- [ ] `training/phase1_supervised.py`: MSE / cross-entropy objective, `torch.amp` mixed precision, grad accumulation if needed, LR schedule, checkpointing
- [ ] VRAM telemetry: assert activations peak ≈ 3.5 GB, total < 12 GB during a step
- [ ] Temperature anneal for Gumbel-Softmax over epochs

**AC:** **PLAN Milestone 2** — training runs N steps without gradient breaks; loss decreases; VRAM within envelope; resume-from-checkpoint works.

---

## PHASE 4 — Phase-2 Reinforcement Learning (Sprint 3)
*Goal: train routers + Cortex as true agents with REINFORCE (PPO optional).*

### P4.U1 — Reward functions & credit assignment ⬜
- [ ] `training/rewards.py`: `R_total = α·R_task + β·R_compute`; compute rewards (+0.1 SKIP, +0.2 HALT, −0.1 ROUTE) credited **per routing layer**
- [ ] Baseline subtractor (moving average) to reduce variance
- [ ] Unit tests: reward attribution matches the layer that made each choice

**AC:** Reward vector per layer, per step; deterministic given a routing trace.

### P4.U2 — REINFORCE training loop ⬜
- [ ] `training/phase2_reinforce.py`: categorical sampling replaces Gumbel; log-prob × advantage loss for routers + Cortex Choice head
- [ ] Freeze embeddings & attention (per PLAN); only router/Cortex/CfC-gate params trainable
- [ ] Entropy bonus + KL guard against premature route collapse

**AC:** Policy-gradient updates change routing distributions; entropy doesn't collapse to a single action in the first 1k steps.

### P4.U3 — Evaluation & routing-behavior harness ⬜
- [ ] Eval script: routing frequency per layer per action, halt depth distribution, task accuracy vs. compute-used curve
- [ ] A/B: Gumbel-pretrained router vs. random init router in Phase 2 (validates Phase 1 transfer)

**AC:** Produced evidence that the model uses CfC compute on hard dynamics and skips on static segments. **PLAN Milestone 3.**

### P4.U4 — PPO upgrade (optional stretch) ⬜
- [ ] Replace REINFORCE with clipped-surrogate PPO if variance/instability observed
- [ ] Keep REINFORCE path behind a config flag for comparison

**AC:** PPO runs; sample-efficiency comparison table vs. REINFORCE documented.

---

## PHASE 5 — Edge Deployment (Sprint 4)
*Goal: ONNX export + Go inference wrapper + latency evidence.*

### P5.U1 — ONNX export ⬜
- [ ] `deployment/export_onnx.py`: export with fixed/effective dynamic axes; handle HALT early-exit (two-graph split: trunk + cortex, if single-graph export fails)
- [ ] Parity test: PyTorch vs. ONNX output max-abs-diff < 1e-4 (FP32) / < 1e-2 (FP16)

**AC:** Exported artifact passes parity on 100 random inputs.

### P5.U2 — Go inference wrapper ⬜
- [ ] `deployment/go-inference/`: `model_runner.go` (onnxruntime-go bindings), `main.go` demo CLI feeding telemetry windows
- [ ] Deterministic argmax routing on-edge (no sampling)

**AC:** Go binary loads the ONNX model and returns Choice/Score/Noul for a synthetic window.

### P5.U3 — Latency & footprint validation ⬜
- [ ] Benchmark: p50/p95 latency BS=1 at seq lengths {64, 256, 1024}; per-block halt savings measured
- [ ] Memory footprint (RSS) documented; compare vs. PyTorch runtime

**AC:** Latency table in README; meets edge telemetry targets set in PLAN §1.

---

## Decision Log

All decisions now live in **`docs/DECISIONS.md`** (see D3). This section is kept as a pointer only.

## Guiding principles

- **Simple & readable** — boring, obvious solutions; clear names; short functions (D4).
- **OOP** — one concept = one class; typed dataclass configs; modules own their weights and `forward`.
- See `AGENTS.md` at the repo root for full conventions.

## Risk Register

| ID | Risk | Likelihood | Mitigation |
| --- | --- | --- | --- |
| R1 | `ncps` / torch CUDA wheels unavailable for Python 3.13 | ~~Medium~~ Resolved | Did not occur — torch 2.11.0+cu128 & ncps 1.0.1 installed cleanly on 3.13 (P0.U1) |
| R2 | Masked 3-branch training inflates VRAM (all branches materialized) | Medium | Activation checkpointing on CfC branch; reduce BPTT window; PLAN headroom is 8 GB |
| R3 | Gumbel→categorical distribution shift breaks Phase-2 transfer | Medium | Slow tau anneal; P4.U3 A/B test catches this early |
| R4 | HALT early-exit not ONNX-exportable as a single graph | High | Split export (trunk + cortex); pre-decided in P5.U1 |
| R5 | Router collapse (always SKIP — reward hacking `R_compute`) | Medium | Entropy bonus, β annealing, task-reward gating in P4.U2 |

## Definition of Done (project-level)

1. All units ✅ with ACs passing and checked into `main`.
2. `tests/` green: param cap, gradient flow, ONNX parity.
3. Phase-2 routing-behavior evidence (P4.U3) and latency table (P5.U3) documented in README.
4. Re-producible from scratch: `uv sync && uv run pytest && uv run training/phase1_supervised.py --smoke`.
