# FunctionalRobusta: design spec

Status: draft, under discussion. Companion: [ROADMAP.md](ROADMAP.md).

## Goal

Generalise RHMF from a pixel basis to a function basis, so that data need not be rectangular:
every measurement can sit at its own coordinate, and each object can have its own number of
measurements. The learned basis vectors become continuous functions that can be evaluated
anywhere. Robustness, heteroskedasticity and missing data carry over unchanged.

The design must not be specialised to one basis family: Fourier is the expected default, but sparse bases
(B-splines, wavelets) must fit without restructuring the library.

## Model

Each measurement j is a tuple (row coordinate rⱼ, column coordinate cⱼ, value yⱼ, weight wⱼ).
The model is

```
yⱼ = ∑[k=1..K] a_k(rⱼ) g_k(cⱼ) + sⱼ
a_k(r) = ∑[ℓ] α_ℓk B_ℓ(r)          (row basis, L functions)
g_k(c) = ∑[m] γ_mk H_m(c)          (column basis, M functions)
```

with coefficient matrices α (L × K) and γ (M × K), oriented like `Robusta`'s A (N × K) and
G (M × K).

Three cases of one model:

| Case | row basis | column basis |
|---|---|---|
| RHMF today (`Robusta`) | `Discrete` (one per object) | `Discrete` (one per pixel) |
| Ragged spectra | `Discrete` | `Fourier` (or another continuous basis) |
| Fully functional | continuous, e.g. 2-D `Fourier` on the sky | continuous |

A `Discrete` basis gives each category (object, pixel) its own free coefficient: each basis
function is 1 on one category and 0 on all others. With discrete bases on both axes the
model is exactly today's Y ≈ A Gᵀ.

## Algorithm

Unchanged from the paper; only the solves generalise. One iteration:

1. **w-step**, per measurement: `wⱼ ← wⱼ^data · Q² / (Q² + wⱼ^data sⱼ²)`.
2. **row step**: weighted least squares for α with γ fixed.
3. **column step**: weighted least squares for γ with α fixed.
4. **rotation**: put the factors in a canonical frame (below).

The paper's Appendix A argument is per measurement and never uses rectangular data, so the
loss is non-increasing provided steps 2 and 3 exactly minimise (or, if iterative, are
warm-started and descend on) their quadratic objectives.

**Rotation.** The column functions are made orthonormal under the basis inner product
⟨g_k, g_k′⟩ = (γᵀ S γ)_kk′, where S is the column basis's Gram matrix. The fast rotation
whitens γᵀSγ exactly as `FastAffine` whitens GᵀG today. An SVD-based canonical frame,
ordered by singular value, is the analogue of `SlowAffine`. For a continuous row basis the
row-side Gram enters the same way.

The inner product is taken over the **data domain** D (the range the column coordinates
cover, after any user transforms), not over the basis's full period:
S_mm′ = ∫[D] H_m(c) H_m′(c) dc. S = I for `Discrete`. For a padded `Fourier` basis S is
not diagonal, but it is only n_coeffs × n_coeffs, has a closed form, and is computed once.
Over the full period it would be diagonal, but it would then include the padding, where
the functions are set by the solver's tie-break rather than by data (see Padding below).
(Even over the full period a real cos/sin basis gives S = L for the constant mode and L/2
for the others, not L·I.)

**Padding.** A `Fourier` basis is periodic, so the period must exceed the span of D by a
gap; otherwise the fit wraps one end of the data onto the other. With correlation length s
of the structure being fitted (for spectra, the longest one, e.g. the continuum, not the
line width), the gap should be several s. The cost is modes: at fixed resolution
n_coeffs ∝ period. Padding makes the column step rank-deficient **regardless of the number
of measurements**: with the data covering a fraction f of the period, only about f·n_coeffs
directions are determined. The other ≈ (1 − f)·n_coeffs directions are functions that
vanish on D and live only in the padding (in a numerical check, 100× more data did not move
the cut-off, and the normal-matrix condition number was ≈ 10³⁰⁰ for f ≤ 0.5). This does not affect the
fit on D, but it fixes two requirements: the column solve must be rank-revealing (Step
solvers below), and anything that looks at the coefficients (the Gram above, a `max_frac`
convergence test on γ) must not see the padding directions. Convergence on γ is therefore
tested on function values over D, or on the coefficients projected onto the determined
subspace.

## Architecture

The iteration loop never knows which basis it is using. Three layers:

### 1. Data: `Measurements`

A flat list of measurements: `row`, `col` (integer ids for a `Discrete` axis, coordinates
otherwise; `row` may be 2-D for a sky position), `y`, `w`. Rectangular data are one layout
of this, via `Measurements.from_matrix(Y, W)` (row i, column j ↦ ids i, j). Zero weights
and NaN handling follow `Robusta`'s rules. Known per-measurement coordinate transforms (for
example a Doppler shift into the rest frame) are applied by the user to `col`; no special
support is needed.

The flat list is the general representation, not necessarily the storage. When both axes
are `Discrete` and the data are rectangular, the engine keeps Y and W as dense (N, M)
matrices (see the `Discrete` fast path below).

### 2. Basis (one per axis)

The interface follows FINUFFT's plan structure: what does not depend on the data is fixed
when the basis is constructed (as `makeplan`), what depends on the measurement coordinates
is computed once per fit by `bind` (as `setpts`), and every iteration only applies the
bound operator (as `execute`).

| Step | Meaning | `Discrete` | `Fourier` (dense) | `Fourier` (NUFFT) |
|---|---|---|---|---|
| constructor | modes, domain, tolerance | number of categories | modes, period, offset | modes, period, offset, ε |
| `bind(points)` | precompute for these coordinates; once per fit | nothing | design matrix Φ | points rescaled to [−π, π); a NUFFT plan once jax-finufft exposes one |
| `op.evaluate(coeffs)` | function values at the measurements | gather | Φ c | type-2 NUFFT |
| `op.adjoint(values)` | transpose of `evaluate` | scatter-add | Φᵀ v | type-1 NUFFT |
| `n_coeffs` | number of basis functions | number of categories | number of modes | number of modes |
| `gram(domain)` | inner products over the data domain | I | closed form, not diagonal when padded | same |

The K functions are transformed together as one batch (FINUFFT's `n_trans`). `evaluate` is
linear in `coeffs`, so a dense design matrix, when needed, is available generically. A
bound operator may additionally provide a **fast-path solver** for its axis's step (below).

The dense and NUFFT backends of `Fourier` share one coefficient convention: real cos/sin
coefficients with a fixed, documented map to FINUFFT's complex modes (which carries the
real-valuedness constraint c₋ₖ = conj(cₖ)), and the coordinate rescaling FINUFFT needs.
The dense backend is the exact reference; the NUFFT backend must match it to its tolerance
ε. jax-finufft currently plans and sets points inside every call, so until it exposes
reusable plans the NUFFT `bind` stores only the rescaled points.

### 3. Step solvers

- **Generic (baseline)**: build the dense normal equations from the design and solve
  them with a **rank-revealing** solve: the minimum-norm solution via `eigh` with a relative
  eigenvalue cut-off, as `als._min_norm_solve` already does for K > n_obs. A plain `solve`
  is not enough, because a padded basis makes the system singular however many
  measurements there are (see Padding). Any least-squares minimiser keeps the Appendix A
  argument intact; minimum norm is the tie-break, and for `Fourier` it sends the function
  towards zero in the padding, which is never evaluated. The cut-off must sit above the
  rising-loss check's round-off floor. Correct for any basis, limited to small
  `n_coeffs × K` (a few thousand). It is the reference every fast path is tested against.
- **Fast paths** (optional, chosen by the basis):
  - `Discrete`: the step separates into batched K × K solves (today's a- and g-steps,
    `als.normal_equations` / `solve_normal_equations`). On flat measurements this costs
    index arrays and gather/scatter where today's code uses dense matrix products. So when
    both axes are `Discrete` and the data are rectangular, the fast path keeps the dense
    (N, M) layout and runs the existing kernels unchanged. This is what makes the backend
    swap free for `Robusta` users. (The generic path on a `Discrete` axis builds a dense
    design with one column per category, almost all zeros: tests on small problems only.)
  - Banded (B-splines): block-banded direct solve.
  - `Fourier`: normal-matrix blocks are Toeplitz and come from type-1 NUFFTs; solved densely
    (rank-revealing, as above) or with JAX's conjugate gradients
    (`jax.scipy.sparse.linalg.cg`) when too large. CG on the singular but consistent system
    never changes the null-space part of its starting point, so a warm start carries its
    padding component forward: project it out, or accept it, since it does not change the
    fit on D.

Anything specific to one basis family lives in that basis's fast path, never in the loop,
the state or the public API.

## Public API

```python
from robusta_hmf import Robusta                        # unchanged
from robusta_hmf.functional import FunctionalRobusta, Measurements, Discrete, Fourier

data = Measurements(row=row_id, col=lam, y=flux, w=ivar)
model = FunctionalRobusta(
    rank=10,
    robust_scale=5.0,
    row_basis=Discrete(n=n_objects),
    col_basis=Fourier(n_modes=801, period=L),
)
state, loss_history = model.fit(data)
model.predict(rows, lam_new)          # evaluate the model anywhere
model.robust_weights(data)            # per measurement
```

Method names and arguments mirror `Robusta` where the meaning carries over: `fit`, `infer`
(column coefficients fixed), `score`, `robust_weights`, `fit_info`; `robust_scale`,
`robust_nu`, `conv_strategy`, `conv_tol`, `dtype`, `nan_error`. `synthesize` (a matrix) is
replaced by `predict` (at given points). Same warnings, exceptions and precision rules.

## Equivalence and the backend swap

`Robusta` keeps its API and behaviour. `FunctionalRobusta` is built alongside it, then tested
against it:

```python
data = Measurements.from_matrix(Y, W)
ref = Robusta(rank=5, robust_scale=5.0)
new = FunctionalRobusta(rank=5, robust_scale=5.0,
                        row_basis=Discrete(n=N), col_basis=Discrete(n=M))
```

With the same initial state and settings, A, G, robust weights and loss history must agree
to float64 round-off (about 1e-10 relative). Once the discrete path matches across the
existing test suite, `Robusta`'s dense backend is swapped to delegate to the generic engine,
removing the duplicate code; the equivalence tests then guard against regressions.
`ShardedFrame` is out of scope for the first swap.

## Out of scope for v1

Regularisation (GP priors), non-negativity, line-spread functions, fitting unknown
per-object shifts, sharding. The design must not preclude them:

- A quadratic (GP) penalty on one factor fights the rotation step through the scale
  degeneracy (a → s·a, g → g/s); adding one later means penalising both factors and making
  the rotation balance them.
- Non-negativity is a convex constraint on the solves and removes the rotation step.
- Line-spread functions fit as a per-measurement linear operator on the column basis.

## Open questions (decide when they matter)

- Column coordinate for spectra (ln λ?), band limit, and the padding gap (rule in Padding
  above; the number of correlation lengths is still to be chosen).
- Edge modes: directions partly on D and partly in the padding have small but non-zero
  eigenvalues and may amplify noise near the ends of D. The eigenvalue drop is steep, so
  this band is narrow, and pooling many objects should average it down; check on the
  Phase 2 toy.
- Initialisation when a basis is not `Discrete` (the SVD start needs rectangular data).
- Coverage and exclusion rules for continuous bases (a basis function with no data in its
  support; poorly constrained Fourier modes).
- NUFFT tolerance relative to the rising-loss check's round-off floor.
- Loss prefactor convention (`PLAN.md` "Fix NLL scaling"); match `Robusta` until decided.
