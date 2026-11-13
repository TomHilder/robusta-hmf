# Robusta-HMF

`jax` implementation of robust heteroskedastic matrix factorisation. Robusta like the coffee bean, get it?

## Installation

Easiest is from PyPI either with `pip`

```sh
pip install robusta-hmf
```

or `uv` (recommended)

```sh
uv add robusta-hmf
```

Or, you can clone and build from source

```sh
git clone git@github.com:TomHilder/robusta-hmf.git
cd robusta-hmf
pip install -e .
```

## Usage

Given a data matrix `Y` of shape `(N, M)` and a matching matrix of inverse-variance
weights `W`, fit a rank-`K` factorisation `Y ≈ A @ G.T`:

```python
import jax
jax.config.update("jax_enable_x64", True)  # robusta-hmf fits in float64 by default

from robusta_hmf import Robusta

# rank-5 robust fit; robust_scale controls how aggressively outliers are downweighted
model = Robusta(rank=5, robust=True, robust_scale=2.0)
state, loss_history = model.fit(Y, W, max_iter=1000)

basis = state.G          # (M, K) shared basis vectors
coeffs = state.A         # (N, K) per-row coefficients
reconstruction = model.synthesize()   # A @ G.T, shape (N, M)
```

Missing or masked data is handled by setting the corresponding entries of `W` to `0` (see
below). See `examples_paper/` for complete, worked pipelines (synthetic validation in `toy/`
and Gaia RVS spectra in `gaia_rvs/`).

### Missing data and coverage

Mark missing entries with `W = 0`; their values in `Y` are ignored. NaN or infinite values in
`Y` or `W` are treated the same way: they are set to `Y = W = 0` and a `DataWarning` says how
many there were. Pass `Robusta(..., nan_error=True)` to raise instead. Negative weights always
raise.

The **coverage** of a row or column is its number of non-zero weights. A row of `A` (one
row of `Y`) or of `G` (one column of `Y`) has `rank` unknowns, so it is determined by the data
only if its coverage is at least `rank`. Rows and columns with less are **excluded**:

- Excluding one can leave another with too little coverage, so exclusion repeats until
  nothing changes.
- Excluded rows and columns take no part in the fit. Their rows of `A` and `G` come back as
  NaN, and `state.row_excluded` / `state.col_excluded` mark them, so `synthesize()` and
  `residuals()` are NaN exactly where the model is undefined.
- One `UnderdeterminedWarning` lists them, and `model.fit_info.exclusions` records each one
  and why.
- The returned state can be saved, passed back to `fit(init_state=...)`, or used for
  `infer()` and `score()`. `infer()` keeps the columns excluded in training and excludes new
  rows with too little coverage. `score()` treats excluded entries as zero-weight and reports
  how many there were (`n_excluded_rows`, `n_excluded_cols`, `n_excluded_entries`).

If you would rather keep every row and column, set `als_ridge`. The ridge then decides the
under-covered factors (it shrinks them towards zero), nothing is excluded, and the
`UnderdeterminedWarning` says which factors the ridge, not the data, determines. A ridge too
small to register next to the least-squares system at the working precision
(`ridge ≤ 10·ε·d`, with `d` the system's largest diagonal entry) is no prior at all, and a
warning gives the value it would need to exceed.

### Convergence and warnings

Every warning robusta-hmf issues derives from `RobustaWarning`:

| Warning | Means |
|---|---|
| `DataWarning` | Non-finite input entries were treated as missing. |
| `UnderdeterminedWarning` | Some rows or columns have fewer non-zero weights than `rank`. |
| `ConvergenceWarning` | The fit stopped at `max_iter` without converging, or its loss rose. |

`model.fit_info` records how the last `fit()` went: `converged`, `stop_reason`, `n_iter`,
`final_loss`, `loss_rises` and `exclusions`. `infer()` stores nothing on the model, since one
model is often applied to many datasets; its warnings name the number of rows instead.

To make any of them an error:

```python
import warnings
from robusta_hmf import ConvergenceWarning
warnings.simplefilter("error", ConvergenceWarning)
```

### If the loss rises or diverges

ALS never increases the loss in exact arithmetic. So when a fit's loss rises by more than
round-off, the computation itself has gone wrong, and a `ConvergenceWarning` says at which
iteration. The usual causes, in order:

1. **float32.** Fit in float64 (the default) first. If that behaves, the float32 fit was
   running out of precision in badly conditioned solves.
2. **A ridge too small to matter.** If you set `als_ridge`, check for the warning above, and
   either raise the ridge or remove it so under-covered rows and columns are excluded.
3. **A rank too high for the data.** Rows or columns with coverage only just above `rank`
   are determined, but poorly. Try a lower rank.

If the loss becomes NaN or infinite, the fit stops with a `FloatingPointError` whose message
names the likely cause.

### Precision

Fits run in float64 by default. JAX only provides float64 once
`jax.config.update("jax_enable_x64", True)` has been called, before any JAX array is
created. Without it, `fit()`, `infer()` and `score()` raise `PrecisionError` instead of
silently computing in float32.

float32 is supported as an explicit choice, `Robusta(..., dtype="float32")`, and is much
faster on GPUs. It is also less forgiving of badly conditioned solves; see
[If the loss rises or diverges](#if-the-loss-rises-or-diverges).

### Model selection

`score()` returns the held-out statistics used in the paper. The headline number is the KL
divergence of the standardised residuals from `N(0, 1)` (Eq. 20) — zero is perfect, lower is
better — which is what to minimise over a `(K, Q)` grid:

```python
infer_state, _ = model.infer(Y_test, W_test, state=state)
scores = model.score(Y_test, W_test, state=infer_state)

scores.kl        # KL divergence from N(0, 1); the model-selection criterion
scores.std_z     # sigma_z, target 1
scores.chi2_red  # reduced chi-squared, target 1
```

### Fitting data larger than one GPU

Each ALS step over the whole `(N, M)` matrix materialises about five full-size
intermediates on top of `Y` and `W`. For 496,955 × 2321 spectra in float64 that is ~43 GiB
of scratch, which will not fit in a 48 GiB card.

**Chunking the row axis is what fixes this**, and it needs no second device — peak scratch
drops to ~0.4 GiB, set by `row_block` rather than by `N`. Sharding across devices is a
speed knob layered on top. Both come from swapping in one frame:

```python
from robusta_hmf import Robusta, ShardedFrame

model = Robusta(
    rank=10,
    robust_scale=5.0,
    override_frame=ShardedFrame(row_block=4096),  # devices default to all visible
)
state, loss_history = model.fit(Y, W)   # plain numpy arrays; never fully copied to host
```

Everything else — `infer`, `synthesize`, `score`, `coefficients`, `residuals` — works
unchanged, and the results match the ordinary path to round-off. `Y` and `W` are stored on
device in float32 by default while all arithmetic happens in the model's `dtype` (float64
unless you opt in to float32), halving resident memory.

To fit several models to one sample (say a `(K, Q)` grid) without re-transferring the data,
place it once and reuse the handle:

```python
frame = ShardedFrame(row_block=4096)
data, _ = frame.prepare(Y, W)          # one transfer to device
for rank in ranks:
    model = Robusta(rank=rank, override_frame=frame)   # bind() copies, so reuse is safe
    state, _ = model.fit(data)
```

## Citation

If you use `robusta-hmf` in your research, please cite the accompanying paper
([arXiv:2607.08081](https://arxiv.org/abs/2607.08081)):

```bibtex
@article{hilder2026robusta,
    title         = {Robust Heteroskedastic Matrix Factorization: A Generalization
                     of PCA that Flags Outliers and Handles Missing Data},
    author        = {Hilder, Thomas and Hogg, David W. and Casey, Andrew R.
                     and Rix, Hans-Walter},
    year          = {2026},
    eprint        = {2607.08081},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.IM},
    url           = {https://arxiv.org/abs/2607.08081},
}
```

## Help

Found a bug, or have a question or feature request? Please
[open an issue](https://github.com/TomHilder/robusta-hmf/issues) on GitHub.

## TODOs

- [ ] More interpretable loss, maybe normalised in some sensible way (maybe proper NLL)
- [ ] Type checking with `mypy`
