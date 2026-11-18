# FunctionalRobusta: design spec

Status: draft, under discussion. Companion: [ROADMAP.md](ROADMAP.md).

## Goal

Generalise RHMF from a pixel basis to a function basis, so that data need not be rectangular:
every measurement can sit at its own coordinate, and each object can have its own number of
measurements. The learned basis vectors become continuous functions that can be evaluated
anywhere. Robustness, heteroskedasticity and missing data carry over unchanged.

This follows Hogg's two notes ("ragged" and "fully functional" RHMF). The design must not be
specialised to one basis family: Fourier is the expected default, but sparse bases
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
| RHMF today (`Robusta`) | `Indicator` (one per object) | `Indicator` (one per pixel) |
| Ragged spectra (Hogg note 1) | `Indicator` | `Fourier` (or another continuous basis) |
| Fully functional (Hogg note 2) | continuous, e.g. 2-D `Fourier` on the sky | continuous |

An `Indicator` basis gives each category (object, pixel) its own free coefficient, so with
indicators on both axes the model is exactly today's Y ≈ A Gᵀ.

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
whitens γᵀSγ exactly as `FastAffine` whitens GᵀG today (S = I for `Indicator`, and S = L·I
for `Fourier` over its period). An SVD-based canonical frame, ordered by singular value, is
the analogue of `SlowAffine`. For a continuous row basis the row-side Gram enters the same
way.

## Architecture

The iteration loop never knows which basis it is using. Three layers:

### 1. Data: `Measurements`

A flat list of measurements: `row`, `col` (integer ids for an `Indicator` axis, coordinates
otherwise; `row` may be 2-D for a sky position), `y`, `w`. Rectangular data are one layout
of this, via `Measurements.from_matrix(Y, W)` (row i, column j ↦ ids i, j). Zero weights
and NaN handling follow `Robusta`'s rules. Known per-measurement coordinate transforms (for
example a Doppler shift into the rest frame) are applied by the user to `col`; no special
support is needed.

### 2. Basis (one per axis)

Required interface:

| Method | Meaning | `Indicator` | `Fourier` |
|---|---|---|---|
| `n_coeffs` | number of basis functions | number of categories | number of modes |
| `evaluate(coeffs, points)` | function values at the measurements | gather | type-2 NUFFT (or dense) |
| `adjoint(values, points)` | transpose of `evaluate` | scatter-add | type-1 NUFFT (or dense) |
| `gram()` | inner products of basis functions | I | L·I |

`evaluate` is linear in `coeffs`, so a dense design matrix, when needed, is available
generically (for example by `jax.jacfwd`). A basis may additionally provide a **fast-path
solver** for its axis's step (below).

### 3. Step solvers

- **Generic (baseline)**: build the dense normal equations from the design and solve
  directly. Correct for any basis, limited to small `n_coeffs × K` (a few thousand). It is
  the reference every fast path is tested against.
- **Fast paths** (optional, chosen by the basis):
  - `Indicator`: the step separates into batched K × K solves (today's a- and g-steps,
    `als.normal_equations` / `solve_normal_equations`).
  - Banded (B-splines): block-banded direct solve.
  - `Fourier`: normal-matrix blocks are Toeplitz and come from type-1 NUFFTs; solved densely
    or with JAX's conjugate gradients (`jax.scipy.sparse.linalg.cg`) when too large.

Anything specific to one basis family lives in that basis's fast path, never in the loop,
the state or the public API.

## Public API

```python
from robusta_hmf import Robusta                        # unchanged
from robusta_hmf.functional import FunctionalRobusta, Measurements, Indicator, Fourier

data = Measurements(row=row_id, col=lam, y=flux, w=ivar)
model = FunctionalRobusta(
    rank=10,
    robust_scale=5.0,
    row_basis=Indicator(n=n_objects),
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
                        row_basis=Indicator(n=N), col_basis=Indicator(n=M))
```

With the same initial state and settings, A, G, robust weights and loss history must agree
to float64 round-off (about 1e-10 relative). Once the indicator path matches across the
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

- Column coordinate for spectra (ln λ?), band limit, and how to handle the periodic edges.
- Initialisation when a basis is not `Indicator` (the SVD start needs rectangular data).
- Coverage and exclusion rules for continuous bases (a basis function with no data in its
  support; poorly constrained Fourier modes).
- NUFFT tolerance relative to the rising-loss check's round-off floor.
- Loss prefactor convention (`PLAN.md` "Fix NLL scaling"); match `Robusta` until decided.
