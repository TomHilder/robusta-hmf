# PLAN.md

A lightweight status + outstanding-work doc for `robusta-hmf`. See CLAUDE.md for how this
file is used. Git history is the detailed record; this is for orientation and forward intent.

## Status

**Active development** (as of 2026-06-27). The library and both paper example applications
(toy validation + Gaia RVS outlier identification) are done, and the manuscript is
effectively done. Ongoing work is software engineering — hardening and polishing the
library — plus possible feature extensions. Releases are automated: push a `vX.Y.Z` tag and
`.github/workflows/release.yml` builds and publishes to PyPI via Trusted Publishers
(hatch-vcs derives the version from the tag). Latest release is `v0.0.2`.

## Outstanding

Forward-looking work. Empty this list as items are done; if it stays empty indefinitely,
retire this file.

**Engineering / hardening**
- **Fix NLL scaling** — `likelihoods.py` `loss()` multiplies the log-sum by an ad-hoc
  prefactor $Q^2 = \nu s^2$ (`StudentTLikelihood`) / $s^2$ (`CauchyLikelihood`). This is
  neither the paper's Eq. (170) objective (no prefactor) nor the true Student-t NLL prefactor
  $(\nu+1)/2$ (which is $1$ for Cauchy, $\nu=1$). Harmless for the ALS argmin, but misleading
  as a "negative log-likelihood" and it rescales the SGD gradient. Pick one convention — drop
  the prefactor (match Eq. 170) or use $(\nu+1)/2$ — and apply it consistently across
  `StudentTLikelihood.loss` and `CauchyLikelihood.loss`.
- **Docs** — user-facing documentation (README, usage/API docs, docstrings).
- **Type-checking everywhere** — comprehensive type annotations + a type checker run
  across the codebase (the package already ships a `py.typed` marker).
- **Code clean-up** — refactors and dead-code removal.

**Possible extensions** (ideas, not commitments)
- Regularisation (no implementation yet; the empty `regularisers.py` placeholder was removed 2026-07-30).
- Non-pixel bases.
- Others TBD.

## Log

Append-only. Add an entry when you do something notable.

| Date | Event |
|------|-------|
| 2026-08-02 | Packed the final full-RVS fit (K=16, Q=7.5) into a single HDF5 file — flux, ivar, model flux, per-pixel robust weights and the whole metadata CSV, in catalogue row order (`export_model_hdf5.py`, `gaia_rvs_results/full_rvs_K16_Q7.50_model.hdf5`, 36 GiB). Added `make_paper_figs.py outlier_taxonomy`: one paper-style exemplar per outlier group named in Section 5, titled "Gaia DR3 <id>: <group>", with Ce/Nd/Zr abundance windows on the neutron-capture example. |
| 2026-06-27 | Slimmed the project-management scaffolding. The previous PLAN.md (full task/decision/log history through Task 17) is preserved in git history. Reduced PLAN.md to status + outstanding-work; trimmed CLAUDE.md project-management ceremony; removed equinox-report.md (also in history). |
| 2026-07-30 | Expanded the test suite (`test-suite` branch): integration tests for the public Robusta API (ground-truth recovery, robust downweighting, infer, missing data, serialization, warnings/errors), resurrected `test_hmf.py`, ridge normal-equation and npz round-trip tests. Suite is 207 tests, ~16 s. Fixed a bug where `Robusta.fit(method="sgd")` crashed (`step_sgd` did not accept `skip_G`). Closes the "Better tests" outstanding item. |
