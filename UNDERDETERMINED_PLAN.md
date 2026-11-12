# Under-determined and non-converging fits: implementation plan

Implements the behaviour agreed in PR #13 (issues #2 and #10). The proposal itself is in the
PR comments; this file turns it into a spec, a test list, an order of work and a list of risks.
Delete this file once the work is merged and documented.

## Notation

The model is `Y ≈ A Gᵀ`, where:
- `Y` and `W` are `(N, M)`: N rows (e.g. spectra) by M columns (e.g. pixels);
- `A` is `(N, K)`; `G` is `(M, K)`.

The **coverage** of a row or column is its number of entries with non-zero weight.

The **effective coverage** during a fit counts only entries with `W > 0` *and* a robust (IRLS)
weight of at least `min_robust_weight`.

## 1. Spec

### 1.1 Public API changes

**`Robusta(...)` gains two keyword arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `nan_error` | `False` | With `False`, non-finite entries in `Y` or `W` are treated as missing (both set to 0 there) and one `DataWarning` is issued. With `True`, raise `ValueError` instead. |
| `min_robust_weight` | `1e-2` | During the fit, an entry counts towards effective coverage only if its robust weight is at least this. Ignored for Gaussian (non-robust) fits, whose robust weights are all 1. |

They go on the constructor, not on `fit()`, so that `fit`, `infer` and `score` behave the same
way for one model.

**New warning classes in `robusta_hmf.warnings`**, exported from the package root:
- `RobustaWarning(UserWarning)`, the base class;
- `DataWarning(RobustaWarning)`, for input patching;
- `UnderdeterminedWarning(RobustaWarning)`, for coverage below K, whether or not anything is excluded;
- `ConvergenceWarning(RobustaWarning)`, for a loss rise or reaching `max_iter`.

**New `FitInfo` dataclass**, stored as `model.fit_info` by `fit()` only:

| Field | Type | Meaning |
|---|---|---|
| `converged` | `bool` | The convergence test passed. |
| `stop_reason` | `Literal["converged", "max_iter"]` | Why the loop ended. A loss rise warns but does not stop the fit, so it is not a stop reason. |
| `n_iter` | `int` | Iterations run. |
| `final_loss` | `float` | Last entry of `loss_history`. |
| `loss_rises` | `tuple[int, ...]` | Every iteration that broke the rising-loss rule (§1.6). |
| `exclusions` | `tuple[ExclusionEvent, ...]` | Every exclusion, in order. |

`ExclusionEvent` has these fields:
- `iteration`: `-1` means before the first iteration, i.e. from the input weights;
- `axis`: `"row"` or `"col"`;
- `indices`: the excluded indices;
- `cause`: `"empty"`, `"low_coverage"`, `"robust"` or `"cascade"`.

**`RHMFState` gains two optional fields:**
- `row_excluded`: bool array of shape `(N,)`, or `None`;
- `col_excluded`: bool array of shape `(M,)`, or `None`.

`None` means nothing is excluded, which keeps every existing constructor call working.
`update_state`, `save_state_to_npz` and `load_state_from_npz` carry them. Loading an old
`.npz` without them gives `None`.

**`ZMoments` gains three counts:**
- `n_excluded_rows`;
- `n_excluded_cols`;
- `n_excluded_entries` (entries with `W > 0` that fall in an excluded row or column).

They also appear in `as_dict()`.

`fit()` still returns `(state, loss_history)`, and `infer()` still returns
`(state, loss_history)`. The Robusta API changes only by addition.

### 1.2 Input validation (`fit`, `infer`, `score`)

Done once per call, before the data reach the frame's loop, in this order:

1. **Shapes.** `Y` and `W` must have the same 2-D shape (already checked for the sharded path; add it for the dense path).
2. **Non-finite entries.** Find every entry where `Y` or `W` is NaN or ±inf.
   - With `nan_error=True`: raise `ValueError`, giving the count and the first few `(row, col)` positions.
   - Otherwise set both `Y` and `W` to 0 there and issue one `DataWarning`, for example:
     `"Found 1,204 non-finite entries in Y or W (first at row 41, column 7); treated them as missing (Y = W = 0). Pass nan_error=True to raise instead."`
3. **Negative weights.** Any `W < 0` raises `ValueError`. An inverse variance cannot be negative. This item is not in the PR proposal, but it is in its spirit.
4. **Shape versus rank** (`fit` only):
   - `K > N` or `K > M` raises `ValueError`, saying that this is probably a shape mistake (e.g. `Y` transposed) or a rank larger than the data can support.
   - `infer` is exempt for `N`: inferring a single spectrum (N=1) at K=10 is normal, because each row of `A` is solved against the `M` columns. `infer` still needs `K ≤ M`.

For a prepared `ShardedData` handle, steps 2 and 3 have already happened in `ShardedFrame.prepare`
(see §1.8). Step 4 uses the handle's `n_rows` and `n_pixels`.

### 1.3 Coverage and exclusion

**Mechanism.** Exclusion masks the weights. The loop carries two "active" masks, `row_active`
of shape `(N,)` and `col_active` of shape `(M,)`, and every step uses

`W_eff = W * row_active[:, None] * col_active[None, :]`

in place of `W`, both for the updates and for the loss. An excluded row or column then has
coverage 0. The existing empty-system guard in `als.solve_normal_equations` returns an exact
zero factor for it, and its entries contribute nothing to the loss. Nothing is excluded by
editing `A` or `G` directly during the loop.

**At the start of `fit`, with no ridge:**
1. Compute the row and column coverage from `W > 0`.
2. Exclude every row and column with coverage `< K`. This includes coverage 0.
3. **Cascade.** Recompute coverage on the masked weights, exclude again, and repeat until nothing changes.
4. If no rows or no columns remain, or fewer than K of either, raise `ValueError`.
5. Issue one `UnderdeterminedWarning` for the whole cascade, for example:
   `"K=8: 3 rows and 352 columns have fewer than 8 non-zero weights (rows 12, 40, 41, …; columns 1000, 1001, …). Excluded them from the fit; their entries in A and G are returned as NaN. See model.fit_info.exclusions."`

**During the fit (ALS, robust likelihood only), at every iteration:**
1. Compute the effective coverage from `W_eff > 0` and `w_irls ≥ min_robust_weight`, using the robust weights the step has already computed.
2. Exclude rows and columns whose effective coverage is `< K`, then cascade as above.
3. Exclusion is permanent: an excluded row or column never comes back.
4. The masks are updated on device inside the jitted step, so no per-iteration host synchronisation is needed. An `excluded_at` integer array (initially `-1`) records the iteration of each exclusion.
5. At each convergence check, the host reads `excluded_at`. If anything new was excluded since the last check, it issues one `UnderdeterminedWarning` for that batch.

**With `als_ridge` set:** coverage is computed and the same `UnderdeterminedWarning` is issued,
worded "…have fewer than K non-zero weights; the ridge determines their factors". Nothing is
excluded and no NaN is returned, because the ridge is the user's chosen prior.

**`infer`:**
- Columns excluded in the training state (`state.col_excluded`) stay excluded.
- New rows with coverage `< K` against the active columns are excluded, with the same warning.
- The new rows' robust-weight exclusions follow the same rules as in `fit`.
- There is no cascade into columns, because `G` is fixed in `infer`.

### 1.4 What is returned

At the end of `run`:
- `A[row_excluded] = NaN`;
- `G[col_excluded] = NaN`;
- `row_excluded` and `col_excluded` are set on the state.

As a result, `synthesize()`, `residuals()`, `coefficients()`, `basis_vectors()` and
`robust_weights()` are NaN wherever the model is undefined, with no change to those methods.

**State passed in** (`init_state=` in `fit`, `state=` in `infer`, `score` and `synthesize`):
1. Entries covered by the masks: NaN becomes 0, and the masks seed `row_active` and `col_active`.
2. Any NaN outside the masks, or any NaN in a state with no masks, raises `ValueError` ("state contains NaN outside its exclusion masks").

**`score()`:**
- Excluded entries are treated as `W = 0`. Under the existing `ZMoments` convention they still count in `n`, as other zero-weight entries do.
- The three exclusion counts are filled in.
- Their factors are taken as 0 before computing `r`, so nothing becomes NaN.
- The dense `metrics.z_moments` path and the sharded `_make_moments` path must agree, as they do now.

### 1.5 Non-finite results

The existing checks stay:
- the dense loop raises on a non-finite loss at each convergence check and at the end;
- the sharded loop raises at each convergence check.

Three additions:
- **The sharded loop gets the end-of-run check** the dense loop has.
- **At each convergence check, both loops check that `A` and `G` are finite outside the exclusion masks.** This is cheap: `(N+M)·K` elements.
- **`nonfinite_loss_message`** gains a line about exclusions, because NaN in the *input* can no longer be a cause once §1.2 has run.

### 1.6 Rising-loss rule (ALS only)

At iteration t, with loss `L_t`, the iteration breaks the rule if

```
L_t − L_{t−1} > max( 2·D̄_t , 100·ε·|L_{t−1}| ),   D̄_t = (L_{t−1−w} − L_{t−1}) / w,   w = min(5, t−1)
```

with `D̄_1 = 0`, and `ε` the machine epsilon of the model's dtype. The window restarts at
any iteration where an exclusion happened: `w` counts only iterations since the last exclusion.

**Evaluation.** On the host, at each convergence check, over the loss values since the last
check. Those are already in `loss_history`; converting them to floats is the only cost.

**Warning.** One `ConvergenceWarning` per fit, at the first breaking iteration. For example:

`"K=32, Q=1.0: loss rose at iteration 3 (1.25e+06 → 7.39e+06), which the ALS iteration should never do; the fit is numerically unreliable. See model.fit_info.loss_rises."`

Every breaking iteration goes into `fit_info.loss_rises`. The fit does not stop. Users who want
it to stop can use `warnings.simplefilter("error", ConvergenceWarning)`.

**SGD fits are exempt.** SGD has no monotonicity guarantee.

### 1.7 Reaching `max_iter`

If the loop ends without the convergence test passing, issue one `ConvergenceWarning`:
- from `fit`: `"K=8, Q=3.0: stopped at max_iter=1000 without converging (final loss 1.2456e+06); see model.fit_info"`;
- from `infer`: `"infer on 5,000 rows (K=8, Q=3.0): stopped at max_iter=100 without converging (final loss 3.1e+04)"`.

With `conv_strategy="none"` the user has asked for exactly `max_iter` iterations, so there is no warning.

`Q` is `robust_scale`. For Gaussian fits the `Q=…` part is left out.

### 1.8 Sharded frame

Everything in §1.2–1.7 applies to `ShardedFrame`. Differences:
- **Input sanitising and negative-weight checks happen in `shard_data`**, block by block on the host as each block is transferred, so the full `(N, M)` array is never copied. The `DataWarning` is issued once, after all blocks.
- **Coverage:**
  - row coverage is computed per chunk;
  - column coverage is accumulated over chunks and `psum`'d across devices, like the G-step accumulators;
  - effective coverage is computed in `_make_pass_one` from the `Wt` it already forms;
  - one more `(M,)` int accumulator is added to the carry.
- **Padding rows** (zero weight) count as coverage 0. They must not be reported as exclusions. Report only indices `< n_rows`.
- **The masks** live on device: `row_active` sharded with the rows, `col_active` replicated.

### 1.9 What does not change

- `als.solve_normal_equations` and `_min_norm_solve` stay as they are. The minimum-norm path is still used when a kernel is called directly with `n_obs < K`. It is no longer reachable from `fit`, because §1.2 rejects `K > N` and `K > M` first.
- `fit()` and `infer()` keep their return types.
- Fits with no non-finite inputs, every coverage ≥ K, and robust weights never below `min_robust_weight` on a row or column's last K points are unchanged. That covers almost every fit today, and the results must be identical (§2.9).

## 2. Tests

All new tests go in `tests/test_underdetermined.py`, except where noted. Each mechanism gets
at least one test that fails if that mechanism is removed. Following the precision work, this is
verified by deliberately breaking each mechanism once (mutation check) before merging.

### 2.1 Input validation
- NaN in `Y` where `W > 0`: one `DataWarning`, count in the message, and the fit result equals a fit with those entries set to `Y = W = 0` by hand.
- NaN in `W`, and ±inf in `Y` and in `W`: same.
- NaN in `Y` where `W = 0` (the literal #2 case): no NaN anywhere in the result.
- `nan_error=True`: `ValueError` naming the count and the first position; no fit runs.
- `W < 0`: `ValueError`.
- `K > N` and `K > M` in `fit`: `ValueError` mentioning shape; `infer` with N=1 < K works.
- Mismatched `Y`/`W` shapes on the dense path: `ValueError`.

### 2.2 Static coverage and the cascade
- A row and a column of all-zero weights: excluded, one warning, NaN in the matching row of `A` and `G`, all other entries finite.
- A column with coverage 0 < c < K: excluded with cause `"low_coverage"`.
- A cascade: a column exclusion that drops a row below K, which then drops another column. One warning, all three in `fit_info.exclusions`, causes `"low_coverage"` then `"cascade"`.
- A cascade that removes everything: `ValueError`.
- Result equivalence: a fit with exclusions equals a fit where the user removed those rows and columns by hand, to round-off on the shared entries.

### 2.3 Effective coverage during the fit
- Construct a column whose few points become strong outliers, so their robust weights fall below 1e-2 after a few iterations. It is excluded at that iteration, with cause `"robust"`, and the event's `iteration` is > −1.
- Exclusion is permanent: once excluded, the column stays excluded at convergence, even when its robust weights would recover against the zero factor.
- `min_robust_weight=0` disables the check (nothing excluded); a large value excludes earlier.
- Gaussian likelihood: no robust-weight exclusions.
- One warning per convergence-check batch, not one per iteration.

### 2.4 Ridge
- Coverage < K with `als_ridge` set: one `UnderdeterminedWarning`; nothing excluded; no NaN; result equals the current ridge result.

### 2.5 Returned state and round trip
- The masks are on the state; `synthesize()`, `residuals()`, `coefficients()` and `basis_vectors()` are NaN exactly on the excluded rows and columns.
- Save and reload through `.npz`: the masks survive, and `infer(state=loaded)` matches `infer(state=original)`.
- Loading an old `.npz` without masks works (`None`).
- `fit(init_state=returned_state)` continues without error, and the excluded rows and columns stay excluded.
- A state with NaN outside its masks, or with NaN and no masks, raises `ValueError` in `fit`, `infer` and `score`.

### 2.6 `infer` and `score`
- `infer` keeps training-time column exclusions; new rows below K are excluded and warned about; `infer` stores nothing on the model (no `infer_info` attribute; `fit_info` unchanged).
- `score` on a state with exclusions: finite scores, the exclusion counts filled in, and moments equal to scoring with those entries' weights set to 0 by hand.
- `score` dense and sharded agree with exclusions present (in `test_sharded.py`).

### 2.7 Convergence reporting
- Rising-loss rule as a pure function, on fixed loss sequences:
  - a monotone sequence: no flags;
  - a sequence with round-off rises below `100·ε·|L|`: no flags;
  - a jump at t=3: flagged at 3, i.e. the short early window works;
  - a jump right after an exclusion iteration: the window restarts.
- Integration: a scaled-down #10 reproduction (§2.8) in float32 with `als_ridge=1e-6` gives one `ConvergenceWarning`, with the first rise and `fit_info.loss_rises` non-empty.
- Healthy float32 and float64 fits (the existing ground-truth recovery fixture) give no `ConvergenceWarning`.
- `max_iter` reached: one `ConvergenceWarning`; message contains K, Q, `max_iter` and the final loss; `fit_info.converged is False` and `stop_reason == "max_iter"`.
- Converged fit: `fit_info.converged is True`; no warning.
- `conv_strategy="none"`: no `max_iter` warning.
- Two non-converging fits with different K in one loop produce two warnings: the message text differs, so Python's once-per-location filter does not hide the second.
- `warnings.simplefilter("error", ConvergenceWarning)` turns the warning into an exception.
- SGD fits never get a rising-loss warning.
- `infer` reaching `max_iter`: the warning includes the number of rows.

### 2.8 Regression for #10
- A scaled-down version of the #10 data (e.g. N=200, M=400, K=8, true rank 4, a block of columns kept by only 3 rows). It must run in a few seconds.
- With no ridge, in float32 and float64, the fit:
  - excludes exactly the under-covered columns;
  - converges;
  - keeps the reconstruction error on the well-covered columns at the noise level;
  - gives no `ConvergenceWarning`.
- On `main` this setup goes NaN, or diverges without a warning. The test must fail there (checked once by hand).

### 2.9 No behaviour change for ordinary fits
- On the existing ground-truth fixtures (full coverage, no NaN), `fit`, `infer` and `score` give bit-identical results before and after: compare against values recorded from `main`.
- Existing tests that use all-zero rows or columns and expect finite `A`/`G` (e.g. in `test_robusta.py`'s missing-data tests) are updated to expect NaN plus the mask. The kernel-level tests in `test_als.py` (`test_ridge_makes_empty_rows_zero_too`, `test_rank_deficient_steps_give_min_norm_solution`) are unchanged, because the kernels are unchanged.

## 3. Implementation order

One commit per step, each with its tests, and the full suite passing after every step.

1. **Scaffolding, no behaviour change.**
   - `warnings.py` with the four classes; exports.
   - `FitInfo` and `ExclusionEvent`.
   - Optional mask fields on `RHMFState`, `update_state`, `.npz` save and load.
   - `OptFrame.run` returns a third value internally (`FitInfo`); `Robusta.fit` stores it and still returns two.
2. **Input validation** (§1.2), dense path, with `nan_error`. Then `shard_data` sanitising (§1.8).
3. **Static coverage, cascade and the masked loop** (§1.3, start of fit), dense path, including the NaN return and the state round trip (§1.4). Update the affected existing tests.
4. **Effective coverage during the fit** (§1.3), dense path, with `min_robust_weight`.
5. **Convergence reporting** (§1.6, §1.7), dense path. The rising-loss rule goes in `convergence.py` as a pure function.
6. **`infer` and `score`** (§1.3 `infer`, §1.4 `score`), dense path.
7. **Sharded parity** (§1.8): coverage accumulators, device masks, padding handling, end-of-run finite check, `z_moments`.
8. **#10 regression test** (§2.8) and the bit-identical check (§2.9).
9. **Documentation:**
   - a README subsection on missing data, coverage and warnings;
   - the "diverging loss: what to do" page #10 asks for;
   - `PLAN.md` log entry;
   - delete this file.
10. **Paper scripts check** (§4, risk 1). Run the toy pipeline end to end and one Gaia bin if the data are available, and confirm the published numbers are unchanged or explain the differences.

## 4. Risks

1. **The paper scripts may change behaviour.**
   - The Gaia pipeline sets NaN to 0 before fitting and clips edge pixels. If any bin has fully masked columns, `G` now comes back with NaN rows there, where it used to come back with zeros. The analysis code (`analysis_funcs.py`, residual plots, outlier scores) may not expect NaN. Example: `np.percentile(w, 1, axis=1)` over robust weights.
   - Mitigation: step 10. Where needed, use `nanpercentile` and the masks in the scripts. The published fits used inputs with no NaN and very likely coverage ≥ K everywhere, so the fitted values should be unchanged. That needs checking, not assuming.
2. **The effective-coverage check changes results** for fits where a column's points are nearly all down-weighted. Those fits used to keep the column, which was ill-determined, and will now exclude it. This is intended, but it is a behaviour change in robust fits with aggressive `Q`. The `1e-2` default (≈14σ at ν=2, s=1) keeps it rare. Release notes should say so.
3. **Performance.**
   - Coverage counts add one elementwise comparison and two reductions over `(N, M)` per iteration. On the dense path they fuse with the robust-weight computation; on the sharded path they add one `(M,)` `psum`.
   - Expected cost is a few percent, against a normal-equations step that is `O(NMK²)`.
   - The host still synchronises only at convergence checks.
   - Measure on the timing setup used in the PR (N=20k, M=2400, K=10 and 32) before and after.
4. **Cascade cost.** The cascade is a loop to a fixed point. At the start it runs on the host, once. During the fit it must run on device: use `lax.while_loop` with a bound of `N + M` iterations, though in practice it ends in 1–2. A pathological cascade could cost many coverage passes in one iteration. That is acceptable, since it can happen at most `N + M` times per fit in total.
5. **Interaction with the convergence test.** `max_frac_G` and `max_frac_A` compare matrices between checks. With masks, the comparison must skip excluded rows: their value goes from finite to 0 at exclusion, which would look like a big change and delay convergence once, or from 0 to NaN at the end. Compare only active rows.
6. **Rotation.** Excluded rows of `G` are exactly zero during the fit, so they contribute nothing to `FastAffine`'s `GᵀG`. `SlowAffine` uses QR of `A` and `G`: zero rows are fine. Check that no rotation divides by a per-row norm.
7. **`ZMoments.n` convention.** Excluded entries count in `n` like any other zero-weight entry. That matches the paper's convention but slightly lowers `std_z` when much is excluded. Reporting the exclusion counts next to the scores is the mitigation. Changing the convention is out of scope.
8. **Warning noise in tests and in the paper scripts.** Fits that hit `max_iter` on purpose (many existing tests use small `max_iter`) will now warn. Tests should assert the warning where it is the point, and otherwise filter it with a module-level `pytestmark`. Don't silence it globally in `conftest.py`, which would hide regressions.
9. **SGD.** Input validation and static exclusions apply. Effective-coverage exclusions and the rising-loss rule do not (no robust-weight step to read, no monotonicity). The SGD path is weakly tested already, and `infer()` on SGD-fitted models crashes (a separate known bug). Keep SGD changes minimal, and test only that SGD fits run and honour static exclusions.
10. **API naming.** `nan_error`, `min_robust_weight`, `fit_info`, the warning class names and the `ExclusionEvent` fields will be public. Settle the names in review before step 1.

## 5. Out of scope

- A warning when `als_ridge` is negligible at the working precision (raised in the PR discussion, not decided).
- Changing `als_ridge` to be relative to the scale of each least-squares system.
- Changing the `ZMoments.n` convention.
- The `infer()` crash on SGD-fitted models.
- The NLL prefactor convention (`PLAN.md`).
