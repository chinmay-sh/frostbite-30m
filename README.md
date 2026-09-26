# Frostbite-30M

Hybrid 29.8M-parameter edge model: Self-Attention + CfC liquid dynamics + internal RL routing (Route / Skip / Halt), trainable on a single RTX 3060 12 GB.

## Documentation

- `docs/PLAN.md` — architecture specification & implementation plan (the *what*)
- `docs/PROGRESS.md` — phase/unit execution tracker with acceptance criteria (the *how far*)
- `docs/DECISIONS.md` — append-only log of every project decision (the *why*)
- `AGENTS.md` — working conventions for agents & contributors (read before writing code)

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
uv sync                       # install deps (P0.U1)
uv run pytest                 # run test suite
uv run python -m frostbite    # CLI entry (once wired in P0.U3)
```
