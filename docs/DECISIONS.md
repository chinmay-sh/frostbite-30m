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

## Template for new entries

```text
| YYYY-MM-DD | D<n> | <what was decided> | <why, one line> |
```
