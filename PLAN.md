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
- **Switch the Gaia pipeline onto `robusta_hmf.sharded`** — `examples_paper/gaia_rvs`
  (on the `peer-review` branch) still carries `distributed_robusta.py` and
  `20260731_oom_tests.py`, written before the library API existed. Point them at
  `Robusta(override_frame=ShardedFrame(...))` and delete the fork. Same for the metric
  copies in `analysis_funcs.py` and the inline KL in `toy/analyse_toy.py`, now that
  `robusta_hmf.metrics` exists.
- **`loss_cadence`** — the loss is a second full sweep over `Y` and `W` every iteration, and
  under the default `conv_strategy="max_frac_G"` it is not used for convergence at all.
  Computing it only on check iterations would cut a real fraction of the runtime (fp64
  `log1p` and division dominate; K=10 runs at ~0.07 TFLOP/s, nowhere near flop-bound).
  Cheap to add, but it changes what `loss_history` means, so it needs a decision not just a
  patch.
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
| 2026-06-27 | Slimmed the project-management scaffolding. The previous PLAN.md (full task/decision/log history through Task 17) is preserved in git history. Reduced PLAN.md to status + outstanding-work; trimmed CLAUDE.md project-management ceremony; removed equinox-report.md (also in history). |
| 2026-09-24 | Fixed the two `test_als.py` failures on jax >= 0.8. With $K$ greater than the fixed factor's row count the normal equations are singular by construction, and `jnp.linalg.solve`'s output there changed between jax versions. `solve_normal_equations` now takes `n_obs` and, in that case (no ridge), returns the minimum-norm solution via `eigh`, matching `lstsq` on the whitened problem as PR #5 intended, without the per-row whitening PR #5 reintroduced. Chosen from static shapes, so full-rank fits are unchanged. Suite green on jax 0.8.0 and 0.10.2. |
| 2026-07-31 | Moved the row-sharded ALS scheduler into the library as `robusta_hmf.sharded` (`ShardedFrame`, `GramInitialiser`), shipped as a frame rather than a second model class: `Robusta` gained one hook, `override_frame`, and `OptFrame` gained the seams it needs. Rebuilt the ALS kernels around explicit normal equations and fixed a real bug they surfaced — a fully-masked row or pixel NaN'd the whole batched solve. Added `robusta_hmf.metrics` (KL score of Eq. 20, plus `std_z`/`chi2_red`/`rmse`/`mad_z`) so the two examples can drop their inline copies. `Robusta.infer` now runs through the model's own frame, so sharded and dense inference agree exactly rather than to 2e-7. Lowered the declared jax floor to 0.4.28 to match what the GPU deployments run, with a shim for the `shard_map` move; the suite passes on both 0.4.28 and 0.8.0. 130 new tests. |
| 2026-07-30 | Expanded the test suite (`test-suite` branch): integration tests for the public Robusta API (ground-truth recovery, robust downweighting, infer, missing data, serialization, warnings/errors), resurrected `test_hmf.py`, ridge normal-equation and npz round-trip tests. Suite is 207 tests, ~16 s. Fixed a bug where `Robusta.fit(method="sgd")` crashed (`step_sgd` did not accept `skip_G`). Closes the "Better tests" outstanding item. |
