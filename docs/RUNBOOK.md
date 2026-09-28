# Frostbite-30M — Manual Runbook

Everything a human needs to run this project end to end, without reading code.
All commands run from the repo root. Prerequisite: [uv](https://docs.astral.sh/uv/) installed, NVIDIA GPU with CUDA (RTX 3060 12 GB is the reference machine).

## 0. Environment

```bash
uv sync                                   # install everything from uv.lock
uv run python -m frostbite info           # sanity: package + GPU visible
uv run pytest                             # full test suite (CPU, ~1 min)
```

Expect: tests green, `device: cuda`, `gpu: NVIDIA GeForce RTX 3060`.

## 1. Training pipelines (in order)

Each stage consumes the previous stage's checkpoint. Checkpoints live in `runs/` (git-ignored).

### 1.1 Phase-1 — supervised substrate pre-training (~40 min on GPU)

Teaches attention + CfC representations before routing is learned.

```bash
uv run python training/run_phase1.py --epochs 40 --n-series 512 --out runs/phase1/final.pt
```

Watch for: `loss` falling steadily (reference run: 0.0045 → 0.0007). Output: `runs/phase1/final.pt`.

### 1.2 Phase-2 — REINFORCE routing (~10 min on GPU)

Freezes the trunk; routers + cortex learn Route/Skip/Halt on simulated telemetry.

```bash
uv run python training/run_phase2.py --updates 150
```

Watch for: the final routing table — `halt@` depth and `substrates` must increase with difficulty (static < oscillatory < chaotic), ending with `MILESTONE 3` printed. Output: `runs/phase2/final.pt`.

### 1.3 P6 control — warm-start (~10 min on GPU)

Collects random LunarLander trajectories and adapts the trunk to lander dynamics.

```bash
uv run python training/control/warm_start.py --episodes 300 --epochs 8
```

Watch for: loss dropping ≥ 50% (reference: 0.0164 → 0.0048, −70%). Output: `runs/p6/warm.pt`.

### 1.4 P6 control — fine-tuning (~8–10 h on GPU)

The long one: actor-critic control learning on LunarLander-v3.

```bash
# fresh run from warm-start
uv run python training/control/run_control.py --updates 250

# continuation: resume from a control checkpoint, 8 episodes/update
uv run python training/control/run_control.py --resume runs/p6/control.pt \
    --updates 200 --episodes-per-update 8 --save runs/p6/control_v2.pt

# stage-2: additionally unfreeze embedding+attention at low LR (D20)
uv run python training/control/run_control.py --resume runs/p6/control.pt \
    --updates 200 --episodes-per-update 8 --trunk-lr 1e-5 \
    --save runs/p6/control_v3.pt
```

Watch for: `return` climbing away from −200 (random baseline) over the first ~30 updates, and `entropy` (max 1.386) drifting down as the policy commits — a healthy run shows both.
If returns stay flat through ~update 100 with entropy pinned near max, see `docs/DECISIONS.md` D19/D20 for the diagnosis trail.
Reference figures will be recorded in `docs/PROGRESS.md` §P6.U3 when a successful run completes.
Output: `runs/p6/control*.pt`.

### 1.5 P6 control — PPO with trunk action head (D21/D22)

The current best path: trunk-fed action head + PPO (GAE, clipped surrogate).

```bash
uv run python training/control/run_ppo.py --resume runs/p6/control.pt \
    --updates 200 --episodes-per-update 8 --trunk-lr 1e-5 \
    --save runs/p6/control_ppo.pt
```

Note: the action head changed shape (choice-bottleneck → trunk, D21); resuming from a pre-D21 checkpoint starts a fresh head — the message `action_head shape changed` is expected.
Watch for: `return` climbing AND `clip` staying below ~0.3 (high clip-fraction means the policy is moving too far per batch).
Output: `runs/p6/control_ppo.pt`.

## 2. Evaluation

### 1.6 P6 control — Double DQN + frame-skip (Tier 2, D23)

The landing attempt: off-policy replay over the frozen trunk.

```bash
uv run python training/control/run_dqn.py --resume runs/p6/control.pt \
    --episodes 600 --action-repeat 3 --save runs/p6/control_dqn.pt
```

Watch for: `trailing20` climbing steadily once `eps` falls below ~0.3; the run
stops itself and prints `SOLVED` when the trailing-20 mean reaches +200.
Frame-skip 3 means each model decision covers 3 physics steps — early episodes
are random (warm-up: first 1,000 decisions), then replay learning kicks in.
Output: `runs/p6/control_dqn.pt` (includes `solved` flag + history).

### 2.1 Routing behavior on telemetry (Milestone-3 table)

```bash
uv run python training/run_phase2.py --updates 0   # loads ckpt, prints pre/post tables
```

### 2.2 Control agent — returns + per-phase routing

```bash
uv run python training/control/evaluate.py --checkpoint runs/p6/control.pt --episodes 50
```

### 2.3 Ablations — learned router vs SKIP-only vs ROUTE-all

```python
from training.control.ablate import ablation_suite
results = ablation_suite("configs/arch_30m.yaml", "runs/p6/control.pt", episodes=30)
for name, r in results.items():
    print(name, f"{r.mean_return:+.1f} +/- {r.std_return:.1f}")
```

## 3. Utilities

| Command | Purpose |
| --- | --- |
| `uv run python training/verify_env.py` | R9 check: LunarLander-v3 installs/works on this machine |
| `uv run python training/smoke_gpu.py` | 2-step GPU training smoke: VRAM + latency figures |
| `uv run python -m frostbite info` | package/version/GPU summary |
| `uv run pytest -k param` | just the 30M-cap tests |

## 4. Troubleshooting

- **`RuntimeError: ... tensors to be on the same device`** — a CPU tensor reached a CUDA model. All known spots are fixed; if new, move inputs with `.to(next(model.parameters()).device)`.
- **Control run dies mid-training (OOM)** — fixed in `fix(P6.U3)` via per-episode backward; if it recurs, lower `--episodes-per-update` to 2.
- **box2d import errors / SwigPyObject warnings** — harmless deprecation warnings on Windows; a hard failure means `uv sync` didn't install `gymnasium[box2d]` cleanly → `uv sync --reinstall-package box2d`.
- **Slow control training** — expected (~2.5 min/update): batch-1 windowed inference through 6 blocks, R7. Vectorized envs are the documented upgrade path.
- **Reproducibility** — every entrypoint takes `--seed`; defaults match the recorded runs.

## 5. Work conventions (see AGENTS.md)

One commit per unit (`feat(P2.U3): ...`), update `docs/PROGRESS.md` in the same commit, log decisions in `docs/DECISIONS.md` the same day.
