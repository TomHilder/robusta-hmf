# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**robusta-hmf** — Robust heteroskedastic matrix factorisation in JAX. A library for decomposing data matrices Y ≈ A @ G.T with per-pixel noise weights and Student-t robust downweighting of outliers. Built on Equinox (see equinox-report.md) and Optax.

The library is being used in a paper with three example applications:
1. **Toy** (`examples_paper/toy/`): Synthetic validation with known ground truth — complete
2. **Gaia RVS** (`examples_paper/gaia_rvs/`): Outlier identification in Gaia DR3 stellar spectra — WIP
3. **Third example** (TBD): Demonstrating how robustness improves the delivered basis

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

### PLAN.md is the master document

**CRITICAL — you MUST use PLAN.md for every task.** Before starting any task, read PLAN.md. After completing any task, update PLAN.md (Tasks table, Log, Session State) and commit. This is not optional. A task is not done until PLAN.md reflects it.

`PLAN.md` (top-level) is the single source of truth for planning, tracking, and execution. It contains:
- The overall project plan and milestones
- Current task breakdown with status (pending / in-progress / done)
- Assignment of tasks to sub-agents
- A decisions log recording key choices and their reasoning
- A session state section with resume instructions and next steps
- Notes on blockers encountered

### Workflow

1. **Claude (you) is the coordinator.** You manage PLAN.md and delegate work to Task tool sub-agents.
2. **Sub-agents** are spawned via the Task tool for specific, well-scoped pieces of work. When delegating to a sub-agent, always provide:
   - The specific file paths it needs to read or modify
   - The relevant section of PLAN.md (copy it into the prompt — sub-agents can't read your context)
   - Clear expected outputs (e.g. "write file X that passes test Y")
3. **After a sub-agent completes**, review its output, commit the work, and update PLAN.md to reflect the new status.
4. **Parallelise when possible** — launch independent sub-agents concurrently.
5. **Use sub-agents to protect context** — delegate research, exploration, and large file reads to sub-agents so the main conversation stays lean. Only the summary comes back.

### Keeping things in sync

- Read PLAN.md at the start of every new conversation to pick up where we left off.
- Update PLAN.md immediately when: a task starts, a task completes, a task is blocked, the plan changes, or a new task is identified.
- Never let PLAN.md drift from reality. If you did something, it must be reflected there.

### Error handling

When you hit an error during a task:
- If it's a code/logic/type error: fix it yourself and continue.
- If it's an environment/access/network error or something you can't resolve after one attempt: stop, report it clearly to the user as a blocker, and update PLAN.md. Do not loop on errors you cannot fix.

### Context management

- **Compaction**: When the conversation is compacted (automatic or via `/compact`), always preserve: the current state of PLAN.md, the list of files modified in this session, and any pending tasks or blockers.
- **Compact between phases**: After completing a major milestone or phase of work, run `/compact` to keep the conversation lean before starting the next phase.
- **End of session**: Before the conversation ends, update the "Session State" section of PLAN.md with what was accomplished, what's in progress, and what to do next.

## Reference

- [equinox-report.md](equinox-report.md) — Equinox dependency report: key APIs, how robusta-hmf uses Equinox, design patterns.
- [codebase-report.md](codebase-report.md) — Codebase overview: library architecture, public API, toy example pipeline, Gaia RVS pipeline, module map.
