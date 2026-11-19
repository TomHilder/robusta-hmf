# FunctionalRobusta: roadmap

Design: [SPEC.md](SPEC.md). Each phase ends with an exit criterion; tick items as they land.
Ordering principle: a second, sparse basis arrives *before* any Fourier-specific speed work,
so the abstractions are tested against two basis families before anything is optimised.

## Phase 0: spec and roadmap

- [x] `functional-robusta` branch off `main`
- [x] Spec and roadmap agreed

## Phase 1: generic engine, discrete bases, equivalence

- [x] `robusta_hmf.functional` package: `Measurements` (+ `from_matrix`), `Basis` interface,
      `Discrete` basis, state for (α, γ)
- [x] Generic dense step solver (reference path), rank-revealing (minimum norm with an
      eigenvalue cut-off), with a test on a deliberately singular system
- [x] w-step, loss and rotation on flat measurements, reusing `likelihoods.py` and the
      rotation maths where possible
- [x] `FunctionalRobusta.fit` with `Robusta`'s convergence, warnings and precision rules
- [x] `Discrete` fast path: batched K × K solves via `als.py`, keeping the dense (N, M)
      layout when both axes are `Discrete` and the data are rectangular
- [x] Equivalence tests against `Robusta` on small problems (missing data, outliers,
      several K and Q)

**Exit:** discrete × discrete matches `Robusta` to float64 round-off, through both the
generic and the fast path.

## Phase 2: Fourier basis (dense), ragged toy

- [x] `Basis.bind(points)` → bound operator (`evaluate`, `adjoint`, `solve`), built once per
      fit, modelled on FINUFFT's plan / setpts / execute; `Discrete` moved onto it
- [x] `Fourier` basis with dense `evaluate` / `adjoint` (explicit cos/sin design), padded
      period, and the Gram matrix over the data domain
- [x] Convergence on γ measured on function values over the data domain, not raw
      coefficients
- [x] `predict` at arbitrary coordinates
- [x] Initialisation for non-discrete bases
- [ ] Ragged toy: per-object coordinates (shifts, different sampling, different lengths);
      recovery of the true functions, compared with "resample to a common grid, then
      `Robusta`"
- [ ] Plots of recovered functions, residuals and robust weights
- [ ] Edge-mode check: noise near the ends of the data domain vs the padding fraction

**Exit:** the toy recovers the true basis and flags injected outliers at least as well as
the resample-then-`Robusta` baseline.

## Phase 3: a sparse basis (genericity check)

- [ ] `BSpline` basis through the same interface, generic solver only
- [ ] Same toy as Phase 2, unchanged apart from the basis argument
- [ ] Review: did anything in the loop, state or API need to know the basis? Fix it if so.

**Exit:** two basis families run through identical engine code.

## Phase 4: feature parity

- [ ] `infer` (column coefficients fixed), `score`, `robust_weights`, `fit_info`
- [ ] Coverage and exclusion rules generalised to basis support
- [ ] Save and load of state

**Exit:** every `Robusta` feature that has a meaning for `FunctionalRobusta` exists and is
tested.

## Phase 5: scaling

- [ ] Fourier NUFFT backend via jax-finufft, matching the dense backend to its tolerance
- [ ] Reusable NUFFT plans in `bind`, once jax-finufft exposes them (separate jax-finufft PR)
- [ ] Fourier fast path: Toeplitz normal-matrix blocks from type-1 NUFFTs; dense or
      `jax.scipy.sparse.linalg.cg`
- [ ] B-spline fast path: block-banded direct solve
- [ ] Benchmarks against the generic path, on problems of increasing size

**Exit:** fast paths match the generic solver, and ragged problems at survey scale
(N ~ 10⁵–10⁶) are practical.

## Phase 6: backend swap

- [ ] `Robusta`'s dense backend delegates to the generic engine with discrete bases
- [ ] Full existing test suite passes unchanged; equivalence tests kept as guards
- [ ] Remove the duplicated code; `ShardedFrame` stays as it is for now

**Exit:** one engine, no change visible to `Robusta` users.

## Phase 7 (sketch): fully functional

A continuous row basis (for example 2-D Fourier on the sky), data without object rows, and
both steps coupled. The phases above should already allow it through `row_basis`; scope,
data and scaling to be decided after Phase 5.

## Not yet scheduled

Real-data application: SPHEREx spectral imaging (see below), regularisation / GP priors,
non-negativity,
line-spread functions, sharding of `FunctionalRobusta`.

## Real data: SPHEREx (target for the wavelength-continuous case)

SPHEREx measurements arrive as a flat list (sky position, wavelength, flux, variance), so
they need no binning into channels or sky pixels. Example access pattern: the
`katieclogston/rosette-nebula` notebook, which queries a `talltable` service for one HEALPix
patch, then bins to a 512 × 512 × 102 cube and runs `Robusta`. Points to settle: data
access (`talltable`), SPHEREx's resolution changing across the band (a resolution-uniform
column coordinate, or knot spacing for B-splines), and per-measurement bandpasses
(fit the spectrum at instrument resolution first). Scheduled after Phase 2.
