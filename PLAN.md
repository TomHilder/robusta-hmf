# PLAN.md

## Goal

Develop `robusta-hmf`, a robust heteroskedastic matrix factorisation library in JAX. The library is being used in a paper with three example applications: a toy validation/demo, a Gaia RVS outlier identification pipeline, and a third (TBD) example demonstrating how robustness improves the delivered basis.

## Milestones

To be defined with user.

## Tasks

| ID | Task | Status | Assigned | Notes |
|----|------|--------|----------|-------|
| 1 | Initial project setup (bootstrap Phase 2) | done | — | Repo, uv, .gitignore already existed. Created PLAN.md and bootstrap state. |
| 2 | Explore Equinox dependency | done | sub-agent | Report: equinox-report.md |
| 3 | Explore robusta-hmf source code | done | sub-agent | Included in codebase-report.md |
| 4 | Explore toy example | done | sub-agent | Included in codebase-report.md |
| 5 | Explore Gaia RVS example | done | sub-agent | Included in codebase-report.md |
| 6 | Create CLAUDE.md | done | — | Project conventions, package management, git, project management workflow. |
| 7 | Set up permissions | done | — | .claude/settings.json with uv, git, WebFetch for docs sites. git push requires confirmation. |

## Decisions

Record key decisions here as they are made. Append only — do not delete previous entries.

| Date | Decision | Options Considered | Choice | Reasoning |
|------|----------|--------------------|--------|-----------|
| 2026-02-10 | Project bootstrapped | — | — | — |
| 2026-02-10 | Package manager | uv, npm, cargo | uv | Already in use; Python project |
| 2026-02-10 | Git author | — | Tom Hilder <tom.hilder.dlhp@gmail.com> | From pyproject.toml |
| 2026-02-10 | git push permission | Auto-allow, require confirmation | Require confirmation | Safer default for shared repo |
| 2026-02-10 | WebFetch domains | None, standard set, custom | Standard set | github.com, pypi.org, arxiv.org, docs.kidger.site, jax.readthedocs.io |

## Session State

_Updated at the end of each session or major phase._

**Last updated**: 2026-02-10
**Status**: Bootstrap complete.
**Next steps**: Define project plan with user.
**Resume instructions**: Read this file top-to-bottom to pick up context. See CLAUDE.md for project conventions. See equinox-report.md and codebase-report.md for technical reference.

## Log

| Date | Event |
|------|-------|
| 2026-02-10 | Bootstrap started. Phase 0: detected existing repo, uv, .gitignore. Phase 1: gathered project info. Phase 2: created PLAN.md and bootstrap state. |
| 2026-02-10 | Phase 3: Explored Equinox, library source, toy example, Gaia RVS example. Wrote equinox-report.md and codebase-report.md. |
| 2026-02-10 | Phase 4: Created CLAUDE.md with project conventions, uv usage, git author, project management workflow, and references to reports. |
| 2026-02-10 | Phase 5: Created .claude/settings.json. git push requires confirmation; WebFetch allowed for github.com, pypi.org, arxiv.org, docs.kidger.site, jax.readthedocs.io. |
| 2026-02-10 | Phase 6: Finalised PLAN.md. All bootstrap tasks verified complete. |
| 2026-02-10 | Bootstrap complete. |
