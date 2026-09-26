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

**Global state:** Phase 3 complete · Last updated: 2026-09-26

| Phase | Title | Units | Status | Depends on |
| --- | --- | --- | --- | --- |
| P0 | Environment & Scaffolding | 4 | ✅ | — |
| P1 | Core Substrates (Attention, CfC, Embedding) | 4 | ✅ | P0 |
| P2 | Auto-RL Cell & Dynamic Routing | 4 | ✅ | P1 |
| P3 | Full Assembly & Phase-1 Training | 5 | ✅ | P2 |
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

### P1.U1 — Temporal Embedding Engine ✅
- [x] `frostbite/modules/embedding.py`: linear projection (sensor-dim → 256) + learned positional encoding (`trunc_normal`, std 0.02) + dropout
- [x] Explicit shape validation (rejects 2-D input, rejects T > max_len)
- [x] Unit tests: output shape `(B, T, d_model)`; gradients reach input projection AND positional table (6 tests)

**AC:** ✅ met — shape + grad tests pass (`tests/test_embedding.py`). Param note: at d_model=256/sensor 128/max_len 1024 the embedding is ≈0.3M, not PLAN's 8.2M (that figure implies a ~32K token table); deviation to be logged with the P1.U4 budget table.

### P1.U2 — Multi-Head Self-Attention ✅
- [x] `frostbite/modules/attention.py`: 4 heads, d=256, pre-norm + residual, causal mask via `scaled_dot_product_attention(is_causal=True)` (flash-attention path on Ampere)
- [x] Unit tests: shape; residual identity check; causal leak-check (perturbing future leaves past outputs unchanged); divibility guard; full gradient flow; exact param formula (6 tests)

**AC:** ✅ met — at d=256: 263K params/block × 6 = **1.58M** vs PLAN's 1.6M ✓. Gradients verified for every parameter.

### P1.U3 — CfC Continuous Substrate ✅
- [x] `frostbite/modules/cfc_substrate.py`: wraps `ncps.torch.CfC` behind strict validation (per D6/D7) — explicit 3-D shape check, hidden-state carryover, out-projection to d_model when units differ
- [x] Chunked BPTT exactness verified: full pass == cat(chunk A, warm-started chunk B), atol 1e-5
- [x] Unit tests: shape+hidden, carryover, gradient flow to cell weights, 2-D rejection, param scaling (5 tests)

**AC:** ✅ met — forward+backward on GPU verified in P0.U1 audit; hidden-state persistence exact; **param note:** at backbone 128 the substrate is ~198K/block (1.19M over 6 blocks) vs PLAN's 12.5M — calibrating backbone width is P1.U4's job.

### P1.U4 — Parameter budget harness ✅
- [x] `tests/test_param_count.py`: hard 30M cap + per-module budgets read from `arch_30m.yaml` (`budgets:` section)
- [x] Calibrated `cfc.backbone_units: 1344` → CfC ≈ 2.07M/block ≈ 12.4M total (PLAN intent preserved, D11)
- [x] **Milestone 1 measured:** embedding 295K + 6×(263,680 att + 2,066,752 cfc) = **14,277,760 params** static stack, forward verified

**AC:** ✅ met — PLAN Milestone 1 (static stack within 30M budget) achieved with measured evidence; ~15.7M headroom remains for router + cortex + norms.

---

## PHASE 2 — Auto-RL Cell & Dynamic Routing (Sprint 2, part 1)
*Goal: the MicroRouter and differentiable Route/Skip/Halt control flow.*

### P2.U1 — MicroRouter (Auto-RL cell) ✅
- [x] `frostbite/modules/auto_rl_cell.py`: `AutoRLCell` (2-layer MLP → 3 logits) + `RoutingAction` enum (`ROUTE=0/SKIP=1/HALT=2`)
- [x] Returns raw logits only — sampling strategy injected by caller (train vs eval paths stay separate)
- [x] Unit tests: logit shape `(B, T, 3)`, determinism, gradient flow, exact param formula, **prod router stack ≤ 0.5M** (6 tests)

**AC:** ✅ met — logit shape `(B, T, 3)`; per-block params within budget; no sampling inside the module.

### P2.U2 — Gumbel-Softmax straight-through estimator ✅
- [x] `frostbite/modules/sampling.py`: `gumbel_sample(logits, tau, hard=True)` wrapping `F.gumbel_softmax` — one-hot forward, soft backward; τ validated
- [x] Property tests: one-hot forward, finite nonzero logits grads, ST Jacobian fingerprint (row-sums = 0, the softmax-Jacobian signature), τ→0 argmax convergence, τ≤0 rejection (5 tests)

**AC:** ✅ met — finite-gradient assertion on logits through the masked path; ST behavior mathematically verified.

### P2.U3 — Masked control flow (train) & branchy execution (eval) ✅
- [x] `frostbite/modules/branch_executor.py`: graph-safe one-hot mixing (`mask*route + (1-mask)*z`); eval-time true conditional skip when nobody routes; train mode always computes all branches
- [x] `frostbite/modules/routing_state.py`: per-token HALT bookkeeping — `fresh/update/resolve/all_halted`, frozen reps survive subsequent blocks
- [x] Tests: all-ROUTE == substrate, all-SKIP identity, eval substrate-untouched (spy), train substrate-always-run, **train/eval parity with argmax actions**, token freeze across blocks, all-halted short-circuit (7 tests)

**AC:** ✅ met — train and eval modes produce identical outputs under argmax-deterministic routing. PLAN Milestone 2 precondition in place.

### P2.U4 — Gradient-flow verification ✅
- [x] `tests/test_gradients.py`: full chain (embed → attend → route → ST Gumbel → masked mix → head) — grads reach embedding, attention, router, substrate, head
- [x] Regression test: all-SKIP mask does NOT zero router gradients (the `z*0` failure mode)
- [x] High-τ (5.0) still delivers router grads; deterministic double-precision `torch.autograd.gradcheck` on the masked mixing (4 tests)

**AC:** ✅ met — gradient tests green incl. finite-difference spot check via gradcheck; the SKIP-mask regression case is covered.

---

## PHASE 3 — Full Assembly & Phase-1 Training (Sprint 2, part 2)
*Goal: the complete Frostbite-30M model + supervised Gumbel training loop running end to end.*

### P3.U1 — Reinforced Liquid Block assembly ✅
- [x] `frostbite/blocks/reinforced_layer.py`: attention → router → masked mix → LayerNorm, threading `RoutingState`; returns `BlockOutput(output, state, logits)` for stack + RL credit
- [x] Train: ST-Gumbel sampling / Eval: deterministic argmax one-hot (parity per P2.U3)
- [x] Tests: shape+state, eval determinism, forced-HALT state capture, ROUTE-biased full gradient flow, all-SKIP case (substrate 0 grads by design, router/attention still learn) (5 tests)

**AC:** ✅ met — forward/backward within budget; routing behavior observable via `BlockOutput.logits`. Test lesson: never use `output.sum()` as a loss on LayerNorm outputs (per-token sum ≡ 0 → fake zero-grad).

### P3.U2 — Global Laya-Style Cortex ✅
- [x] `frostbite/heads/laya_cortex.py`: `LayaCortex` — shared GELU trunk over mean-pooled sequence; Choice head (softmax logits), Score head (scalar), Noul head (sigmoid); returns `CortexOutput`
- [x] Tests: output shapes/types, Noul ∈ [0,1], head independence, full gradient flow, **prod cortex ≤ 7.0M PLAN allocation** (5 tests)

**AC:** ✅ met — heads independently testable; combined params well within budget (compact design, headroom preserved for P4).

### P3.U3 — FrostbiteModel full assembly ✅
- [x] `frostbite/model.py`: Embedding → 6 blocks → cortex; per-token HALT freeze via `RoutingState`, eval-time stack short-circuit when all tokens halted; returns `ModelOutput(cortex, halt_layer, route_probs)`
- [x] **Full production model: 14,848,412 params ≤ 30M cap** (test-enforced)
- [x] Latency baseline recorded: ~250 ms BS=1 seq=256 on CPU (GPU + edge comparison in P5)
- [x] Tests: shapes, eval determinism, forced-HALT short-circuit (stack exits after block 0), end-to-end grads, cap, latency (6 tests)

**AC:** ✅ met — full model within cap; halt short-circuit verified. Total 14.85M leaves ~15.1M headroom under the cap (D11 rationale: compact heads, CfC-dominant).

### P3.U4 — Data pipeline & toy tasks ✅
- [x] `training/data.py`: `SyntheticTelemetry` (sinusoid-mixture + noise, next-state targets), deterministic per-split seeds (train 1000 / val 2000), `make_loaders` helper
- [x] Tests: determinism per split, split separation, window/target shapes `(T,128)/(128,)`, bad-split rejection, loader batch shapes (5 tests)

**AC:** ✅ met — deterministic splits verified; batches of `(B, T, sensor_dim)` produced. Loader performance (pinned memory < 50 ms) deferred to first GPU run in P3.U5 — synthetic data is generated once, not streamed.

### P3.U5 — Phase-1 supervised training loop ✅
- [x] `training/phase1_supervised.py`: `Phase1Trainer` — warmup+cosine LR, linear τ anneal (2.0→0.5), bf16 autocast on CUDA, grad clipping, checkpoint save/resume, per-epoch routing diagnostics
- [x] `training/smoke_gpu.py`: one-shot GPU evidence run
- [x] Tests: loss decreases over epochs, τ/LR schedules exact, checkpoint roundtrip, **full prod model smoke-train on CPU** (5 tests)

**AC:** ✅ met — **PLAN Milestone 2 achieved**: GPU run (batch 128, seq 256) completes without gradient breaks; **peak VRAM 3.46 GiB / 12 GB** (PLAN predicted ~3.5 GB); loss decreases; routing stats logged (route_frac 0.34 at τ=2).

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
