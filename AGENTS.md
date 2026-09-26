# AGENTS.md — Working Conventions for Frostbite-30M

Guide for any agent or human contributing to this repo. Read this before writing code.

## Project context

Frostbite-30M is a 29.8M-parameter hybrid model (Self-Attention + CfC liquid dynamics + internal RL routing: Route / Skip / Halt), trained on a single RTX 3060 12 GB and deployed to edge devices.

- Spec: `docs/PLAN.md` (the *what* — do not edit casually)
- Execution status: `docs/PROGRESS.md` (phases P0–P5, unit IDs like `P2.U3`)
- Decisions: `docs/DECISIONS.md` (append-only log)

## Core principles (non-negotiable)

1. **Keep it simple.** Prefer the boring, obvious solution. No clever tricks, no premature abstraction, no dead code. If a piece of code needs a paragraph to explain, simplify it.
2. **Easy to understand.** Clear names (`ReinforcedLiquidBlock`, not `RLB`). Short functions. Docstrings on every class and public method — one line saying what it does is enough.
3. **OOP architecture.** Each concept is a class: a module owns its weights and its `forward`, config objects are typed dataclasses, training loops are small orchestrator classes. No loose functions juggling ten arguments, no god-objects.

Example shape of a module:

```python
class CfCSubstrate(nn.Module):
    """Liquid time-dynamics layer wrapped around an ncps CfC cell."""

    def __init__(self, config: SubstrateConfig) -> None:
        ...

    def forward(self, x: Tensor, hidden: Tensor | None = None) -> Tensor:
        ...
```

## Commands

```bash
uv sync                 # install / update deps
uv run pytest           # run tests
uv run python -m frostbite   # CLI entry point
```

## Workflow rules

- Work unit by unit, following `docs/PROGRESS.md`. A unit is done only when its acceptance criteria pass.
- **One commit per unit, no exceptions.** As soon as a unit's acceptance criteria pass, update `docs/PROGRESS.md` and commit in the same session. Reference the unit ID in commits: `feat(P2.U3): masked halt control flow` (`feat`/`fix`/`test`/`chore`/`docs`). If a unit needs multiple sessions, commit WIP with `wip(P2.U3): ...` and squash or follow up with the final `feat(P2.U3)` commit when done.
- **At the end of every work session:** update `docs/PROGRESS.md` statuses (⬜/🟨/✅) — the status flip and the unit's code go into the same commit.
- **Every meaningful decision** (library choice, API change, design trade-off) gets a row in `docs/DECISIONS.md` the same day, included in that unit's commit.
- Never exceed the 30M parameter cap — `tests/test_param_count.py` enforces it; keep it green.

## Code conventions

- Python 3.13, PyTorch, type hints on all signatures.
- Config comes from typed dataclasses loaded from `configs/*.yaml` — no magic numbers in code.
- One class per file where practical; file name matches the class name (`auto_rl_cell.py` → `AutoRLCell`).
- Tests live in `tests/` and must run on CPU (GPU optional) so they're fast and portable.
