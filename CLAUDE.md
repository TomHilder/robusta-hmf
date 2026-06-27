# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**robusta-hmf** — Robust heteroskedastic matrix factorisation in JAX. A library for decomposing data matrices Y ≈ A @ G.T with per-pixel noise weights and Student-t robust downweighting of outliers. Built on Equinox and Optax.

The library is presented in a paper with two example applications:
1. **Toy** (`examples_paper/toy/`): Synthetic validation with known ground truth — complete
2. **Gaia RVS** (`examples_paper/gaia_rvs/`): Outlier identification in Gaia DR3 stellar spectra — complete

(A third example was planned but scrapped for scope. The paper and both analyses are effectively done.)

## Code Style

- Always place import statements at the top of files.
- Before writing a new function or utility, search the existing codebase first. Do not reimplement something that already exists.
- Line length: 99 characters (configured in pyproject.toml via ruff).

## Package Management (uv)

- Always use `uv run` to execute scripts: `uv run python script.py`
- Never use bare `python` — always go through `uv run`.
- Add packages: `uv add <package>`
- Remove packages: `uv remove <package>`
- Sync environment: `uv sync`
- Dev dependencies: `uv add --group dev <package>`
- Examples dependencies: `uv add --group examples <package>`
- Run tests: `uv run pytest`

## Git

Commit frequently to checkpoint progress — after completing any meaningful step, not just at the end of a task. Use small, incremental commits. If you notice changes are getting large (many files or hundreds of lines), stop and commit what you have before continuing.

Commits must be authored as Tom Hilder, not as Claude:
```
git commit --author="Tom Hilder <tom.hilder.dlhp@gmail.com>"
```

Do not include `Co-Authored-By: Claude` in commit messages.

Never chain `git add` and `git commit` with `&&`. Run them as separate sequential tool calls. Chained commands don't match the individual permission patterns in `.claude/settings.json` and will prompt the user for approval every time.

## Project Management

`PLAN.md` (top-level) is a lightweight status + outstanding-work doc for the project. Read it at the start of a session for context. It has three short sections:
- **Status** — one-line summary of where the project is (currently: library + paper complete, maintenance mode).
- **Outstanding** — forward-looking work and parked ideas that have nowhere else to live (releases, paper revisions, deferred designs).
- **Log** — thin, append-only; add an entry when you do something notable.

It's a running record, not a gate — small changes don't need a formal task entry, and git history is the detailed record. Commit conventions are in the Git section above. If `Outstanding` stays empty indefinitely, PLAN.md has outlived its usefulness and can be retired.

## Reference

- [codebase-report.md](codebase-report.md) — Codebase overview: library architecture, public API, toy example pipeline, Gaia RVS pipeline, module map.
