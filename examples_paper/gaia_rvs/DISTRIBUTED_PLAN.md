# Plan: promoting `distributed_robusta.py` into `robusta_hmf`

Written 2026-07-30. Status: proposal, nothing implemented yet.

## Goal

Move the row-chunked / row-sharded ALS scheduler out of
`examples_paper/gaia_rvs/distributed_robusta.py` and into the library, so that
fitting a matrix larger than one GPU's memory is a supported mode of
`Robusta` rather than an example-local fork of it.

The numerics are already validated and are **not** what this is about. What
needs work is where the seams go, what the public surface looks like, and what
contracts become other people's problems once it ships.

## Where things stand

| | |
|---|---|
| Source | `examples_paper/gaia_rvs/distributed_robusta.py` (~700 lines) |
| Driver | `examples_paper/gaia_rvs/20260731_oom_tests.py` |
| Fit equivalence to `Robusta` | max abs ΔG 3e-14, Δloss 4e-12 (1 and 4 devices, padded and exact row counts, row_block 64/127/4096) |
| Held-out score equivalence | ΔKL 2e-7, Δstd_z 1e-7 vs `Robusta.infer` + `analysis_funcs` metrics |
| Measured full-sample peak memory | 9.0 GiB (K=10), 9.2 GiB (K=30), of a 35.7 GiB pool |
| Measured throughput | 3.2 s/iter (K=10), ~10 s/iter (K=30), one A6000, fp64, N=496_955 |

The residual 2e-7 in KL is the per-chunk vs global stopping rule in `infer`,
not an arithmetic difference. See "Open decisions".

## The one architectural decision

**Ship a frame, not a second model class.**

`robusta_hmf` already separates model from execution: `HMF` owns the
likelihood, rotation and ALS steps; `OptFrame` owns the loop; `Initialiser`
seeds it. `Robusta` already accepts `override_initialiser` and
`override_conv_tester`.

`DistributedRobusta` bypasses all of that and re-declares `Robusta.__init__`'s
parameter list. The two lists will drift the first time a parameter is added
to one and not the other. Instead:

- `distributed.py` exports **`ShardedFrame`**, an `OptFrame` whose stepper is
  the chunked/sharded one. It reads `likelihood`, `rotation` and
  `a_step.ridge` off the `HMF` handed to it.
- `distributed.py` exports **`GramInitialiser`**, which already plugs into the
  existing `override_initialiser` hook with no library change at all.
- `main.py` gains **one** new hook, `override_frame`, mirroring the two that
  exist.

Everything else — `fit`, `infer`, `synthesize`, `robust_weights`,
`coefficients`, `residuals` — then works unchanged, and a model is configured
in exactly one place.

```python
hmf_kwargs = dict(rank=10, robust_scale=5.0)
model = Robusta(
    **hmf_kwargs,
    override_frame=ShardedFrame(row_block=4096),      # devices default to all visible
    override_initialiser=GramInitialiser(),
)
state, loss = model.fit(Y, W)                          # unchanged call
```

---

## Phase 1 — `als.py`: deduplicate, and fix a real bug

Independent of everything else, and worth doing first because it is a library
bug fix that the distributed path happens to have surfaced.

1. **Singular-solve guard.** `als._solve` has none, so a row whose weights are
   all zero — an all-NaN spectrum after masking, a pixel masked in every
   spectrum — gives `M == 0` exactly and `jnp.linalg.solve` returns NaN,
   poisoning the whole batched solve. Confirmed: `Robusta.infer` NaNs on a
   test set containing one all-masked spectrum, where the chunked path returns
   zero factors. Port `_solve_batch`'s guard (substitute the identity where
   `M` is exactly zero, zero the corresponding `b`) into `als._solve`.
2. **Hoist the shared kernels.** `als._normal_equations` already builds the
   `K*K` outer products that `distributed_robusta._outer` re-derives. Export
   one implementation and have both call it.
3. Extend `tests/test_als.py` with the degenerate row/pixel case.

**Acceptance:** existing tests pass; a fit and an `infer` on data containing an
all-zero-weight row and an all-zero-weight column return finite A, G and loss.

## Phase 2 — `src/robusta_hmf/distributed.py`

Port, restructured around the frame:

- `RowPlan`, `plan_rows`, `build_mesh`, `shard_rows`, `shard_data` — as-is,
  modulo naming (below).
- `ShardedFrame(OptFrame)` — owns `_make_pass_one` / `_make_pass_two` /
  `_make_step` / `_make_infer` / `_make_moments`, built from the `HMF` it is
  given rather than from a duplicated parameter list.
- `GramInitialiser(Initialiser)` — the `Y.T @ Y` route, exact for the top-K
  subspace and avoiding the `(N, min(N, M))` `U` that `jnp.linalg.svd` wants.
- `main.py`: add `override_frame` to `Robusta.__init__` **and** make
  `Robusta.infer` reuse it (it currently constructs a second `OptFrame`
  inline).
- Raise a clear error on `method="sgd"` rather than silently ignoring it;
  `RHMFState.opt_state` is likewise not carried.

**Acceptance:** `compare_to_reference` equivalence thresholds above, driven
through `Robusta(..., override_frame=...)` rather than a second class.

## Phase 3 — API surface

1. **`infer` must match `Robusta.infer`**: same argument names (`Y_infer`,
   `W_infer`, `conv_strategy`, `conv_tol`, `conv_check_cadence`) and same
   return type `(state, loss_history)`. It currently returns
   `(state, n_iterations)`.
2. **`ShardedData`** — public, typed, better named, or hidden behind an
   explicit `prepare(Y, W)` handle. `fit()` currently dispatches on
   `isinstance(Y, ShardedData)`, which is a wart. Whatever the shape, keep the
   property it exists for: a (K, Q) grid must not re-transfer the data per
   grid point.
3. **`robusta_hmf.metrics`** — promote `score()` out of a bare dict. The KL
   score of Eq. 20 is the paper's recommended criterion and is currently
   reimplemented inline in `analyse_toy.py`, while `analysis_funcs.py` has its
   own `std_z`/`chi2_red`/`rmse`/`mad_z`. A `kl_score` / `z_moments` pair in
   the library lets both examples drop their copies. This is the most reusable
   thing in the file and the strongest argument for shipping any of it.
4. numpydoc docstrings and jaxtyping annotations, matching the rest of `src/`.
5. Delete the `__main__` block; `compare_to_reference` becomes a test.
6. Decide on printing. `OptFrame.run` prints unconditionally today, so the
   port is consistent — but that is a decision worth making once rather than
   inheriting twice.

## Phase 4 — portability

The one that will actually bite.

1. **`shard_map` moved.** `jax.experimental.shard_map.shard_map` →
   `jax.shard_map`. The GPU jax here is **0.4.28** (experimental path only);
   jax 0.8 has both, with the experimental one on the way out. Needs a
   try/except shim and a tested floor.
2. **The declared floor is currently fiction.** `pyproject.toml` says
   `jax>=0.6.0`, but everything in `examples_paper/gaia_rvs` runs on 0.4.28
   from system site-packages, because the venv deliberately excludes jax
   (see `run_full_ms.sh`). Pick a floor, test against it, make the two agree.
3. **`check_rep=False`** on the fit pass is a workaround for
   `custom_linear_solve` having no replication rule in 0.4.28. Re-check on the
   pinned version; a Cholesky-based solve may let the check stay on.
4. Declare `numpy` as a direct dependency (present transitively via jax, but
   the module imports it).

## Phase 5 — contracts

Implicit today; they become other people's problems on release. Document in
the relevant docstring, and enforce where cheap.

- **Zero-weight rows and pixels must be inert.** Row padding depends on it.
  True for `GaussianLikelihood` and `StudentTLikelihood`; it becomes a
  contract on the `Likelihood` ABC the moment this ships.
- **Not bitwise reproducible across `row_block` or device count** — summation
  order changes. ~1e-12 in the loss, but say so.
- **`infer` uses a per-chunk stopping rule**, not the global one. This is the
  entire 2e-7 disagreement with the reference. Either implement the global
  rule (costs an extra pass per iteration) or document it.

## Phase 6 — tests

`tests/test_distributed.py`, following the one-file-per-module convention.
Parameterise over:

- `row_block` — chunk-invariance (results were identical at 64 vs 400)
- device count 1 and 4. `XLA_FLAGS=--xla_force_host_platform_device_count=4`
  must be set **before** jax initialises, so this needs a subprocess or a
  `conftest.py` that sets it at collection time
- padded vs exact row counts (N = 499, 501, 512, 4097 all exercised the
  padding path)
- robust vs Gaussian likelihood; ridge vs no ridge
- fp32 and fp64
- degenerate rows and pixels (Phase 1)
- `infer` + `score` equivalence to `Robusta.infer` + `analysis_funcs` metrics

Thresholds: use the measured values in the table above, with an order of
magnitude of headroom.

## Phase 7 — packaging and docs

- Export from `__init__.py`.
- `codebase-report.md` — new module in the map and the architecture section.
- `README.md` — a short "fitting data larger than one GPU" section. The
  headline is that **chunking alone** fixed the OOM (43 GiB of scratch → 0.4
  GiB); sharding is a speed knob on top.
- `PLAN.md` — Log entry, and an Outstanding entry for whatever this defers.
- Once the library API exists, switch `20260731_oom_tests.py` to it and delete
  `examples_paper/gaia_rvs/distributed_robusta.py`.

---

## Open decisions

**Module name.** If it is called `distributed.py`, the person with one GPU and
an OOM will not look in it — and chunking alone is what fixed this. Either
name it for the scheduling (`sharded.py`, `execution.py`) or expose
`row_block` on the ordinary path too, so the single-device fix is reachable
without opting into something called "distributed".

**jax floor.** 0.4.28 (what actually runs on the GPUs here) or ≥0.6 (what
`pyproject.toml` claims). Affects the `shard_map` shim and `check_rep`.

**`loss_cadence`.** The loss is a second full sweep over Y and W every
iteration, and with the default `conv_strategy="max_frac_G"` it is not used
for convergence at all. Computing it only on check iterations would cut a real
fraction of the runtime — fp64 `log1p` and division are the expensive part
(K=10 runs at ~0.07 measured TFLOP/s, nowhere near flop-bound). Cheap to add,
but it changes what `loss_history` means.

**Global vs per-chunk `infer` convergence.** Correctness is unaffected — same
fixed point — but strict reference-matching would want the global rule.

## Risks

- **Silent divergence between the two code paths.** The whole point of Phase 1
  and 2 is that there is one ALS kernel and one parameter list. If the port
  ends up copying `als.py` instead of calling it, this will rot.
- **jax version churn.** `shard_map` has already moved once. The shim needs a
  test that actually runs on both sides of the move, or it is decoration.
- **Padding contract.** A future `Likelihood` where zero weight is not inert
  would break row padding silently — wrong numbers, not an exception. The
  alternative is an explicit row mask carried through the state, which costs
  an argument on every kernel; worth reconsidering if a third likelihood
  appears.

## Sequencing

Phases 1 and 2 decide the shape of everything else and should land first.
3 and 5 can proceed in parallel. 4 blocks release. 6 blocks release. 7 last.

Rough size: 1–2 a day, 3–5 half a day plus the version decision, 6 a day.
