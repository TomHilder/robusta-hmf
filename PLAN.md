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
| 8 | Fix double inference in Gaia CV scoring | done | — | Defer all-data inference to best model only. N-1 wasted inferences eliminated. |
| 9 | Consolidate bin construction across Gaia scripts | done | — | Removed duplicate build_bins() from train_bins.py and plot_bins.py; both now use build_bins_from_config(). |
| 10 | Batch inference for large bins (OOM fix) | done | — | Added batched_infer() to analysis_funcs.py (50k chunk size). Also gc/cache cleanup between bins in analyse_bins.py. |
| 11 | Separate per-bin analysis from cross-bin summary | done | — | Split analyse_bins.py into 3 scripts. Added inference caching, outlier data saving, summarise_bins.py, replot_outliers.py. |
| 12 | Save all per-object outlier scores + fix empty CSV crash | done | — | Added all_outlier_scores.npy saving. Fixed summarise_bins.py crash on bins with 0 outliers. Updated plot_hist_stack.py to use saved scores. |
| 13 | UMAP residuals: metadata coloring + interactive click-to-open | done | — | Rewrote umap_residuals.py: loads bp_rp, abs_mag_G, score per outlier from metadata CSV + saved npz. Interactive mode opens residual PDF on click. |

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

**Last updated**: 2026-02-17
**Status**: Tasks 11–13 complete. Pipeline split into analyse_bins / summarise_bins / replot_outliers. All 14 bins analysed. UMAP residuals script supports metadata coloring and interactive click-to-open.
**Next steps**: Define broader project plan with user. Potential: third example, paper figures, library improvements.
**Resume instructions**: Read this file top-to-bottom to pick up context. See CLAUDE.md for project conventions. Run scripts from `examples_paper/gaia_rvs/` using `builtin cd <path> && uv run python <script>`. NEVER use `builtin uv`.

## Log

| Date | Event |
|------|-------|
| 2026-02-10 | Bootstrap started. Phase 0: detected existing repo, uv, .gitignore. Phase 1: gathered project info. Phase 2: created PLAN.md and bootstrap state. |
| 2026-02-10 | Phase 3: Explored Equinox, library source, toy example, Gaia RVS example. Wrote equinox-report.md and codebase-report.md. |
| 2026-02-10 | Phase 4: Created CLAUDE.md with project conventions, uv usage, git author, project management workflow, and references to reports. |
| 2026-02-10 | Phase 5: Created .claude/settings.json. git push requires confirmation; WebFetch allowed for github.com, pypi.org, arxiv.org, docs.kidger.site, jax.readthedocs.io. |
| 2026-02-10 | Phase 6: Finalised PLAN.md. All bootstrap tasks verified complete. |
| 2026-02-10 | Bootstrap complete. |
| 2026-02-10 | Task 8: Fixed double inference in Gaia CV scoring — compute_all_cv_scores() now only infers on test set; best model inferred on all data once in compute_bin_analysis(). |
| 2026-02-10 | Task 9: Consolidated bin construction — removed duplicate build_bins() from train_bins.py and plot_bins.py; both now import build_bins_from_config() from analysis_funcs.py. |
| 2026-02-10 | Trained bins 9–13 (K=10, Q=5). Analysed bins 9–13 (0 outliers at threshold 0.9). Bin 0 training/analysis pending. |
| 2026-02-11 | Task 10: Added batched_infer() (50k chunks) to fix OOM on bin 4 (192k spectra). Added gc/cache/plt cleanup between bins. |
| 2026-02-12 | Task 11: Separated per-bin analysis from cross-bin summary. analyse_bins.py now saves per-bin results (outliers.csv, summary.json, outlier_data.npz, inferred state). New summarise_bins.py generates cross-bin plots. New replot_outliers.py re-plots outlier spectra from saved data (zero HDF5). Added inference caching to compute_bin_analysis(). |
| 2026-02-17 | Task 12: Added all_outlier_scores.npy (per-object scores for ALL spectra). Fixed summarise_bins.py EmptyDataError on bins with 0 outliers. Updated plot_hist_stack.py to load from saved scores. |
| 2026-02-17 | Task 13: Rewrote umap_residuals.py — loads metadata (bp_rp, abs_mag_G, score, etc.) from CSV + saved npz. Supports coloring by any property. Added interactive mode: click point to open residual PDF in Preview. |
