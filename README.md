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

# JAX defaults to float32, but you should use float64 with robusta
jax.config.update("jax_enable_x64", True)

from robusta_hmf import Robusta

# Instantiate a model. Here, of rank 5, with a robust_scale of 2.0
# The rank is the number of basis vectors in the model
# The robust_scale parameter controls how aggressively outliers are downweighted
model = Robusta(rank=5, robust=True, robust_scale=2.0)

# Perform the fit using data matrix Y and weights matrix W
# Both of shape (N, M), and weights should contain inverse variances
state, loss_history = model.fit(Y, W, max_iter=1000)

basis = state.G          # (M, K) shared basis vectors
coeffs = state.A         # (N, K) per-row coefficients

# The model's (robust) prediction of Y
reconstruction = model.synthesize()   # A @ G.T, shape (N, M)
```

Please refer to the additional documentation below to understand our implementation further.
For reference regarding the method itself, see [Hilder+2026](https://arxiv.org/abs/2607.08081).

### Missing data, coverage and regularisation

One should handle missing or masked data entry `Y_ij` by setting the corresponding
entry `W_ij` to `0`. `.fit()` will ignore any of these values.

Any `NaN` or `inf` values in `Y` or `W` will be handled automatically by setting them to
zero, i.e.: `Y_ij = W_ij = 0`. In this case, a `DataWarning` is issued that informs you
of how many of these cases there were. Passing `Robusta(..., nan_error=True)` will escalate
this warning to a raised Error, which may be desirable for large-scale or pipeline uses
of `robusta-hmf`.

The **coverage** of a row or column is its number of non-zero weights. A row of `A` (one
row of `Y`) or of `G` (one column of `Y`) has `rank` unknowns, so it is determined by the data
only if its coverage is at least `rank`. Rows and columns with less are automatically **excluded**:

- Excluded rows and columns do not inform the fit. Their rows of `A` and `G` will be set
  to `NaN` in the returned fit, to signpost their exclusion. `state.row_excluded` / `state.col_excluded`
  also indicate exclusions, such that `.synthesize()` and `.residuals()` will give `NaN`
  where the model is undefined. `.robust_weights()` instead treats excluded entries like
  missing ones (weight 1), so per-row outlier scores stay finite.
- When exclusion occurs, an `UnderdeterminedWarning` will list them. `model.fit_info.exclusions`
  also records instances of exclusion as well as why.
- The returned state can be saved, passed back to `fit(init_state=...)`, or used for
  `infer()` and `score()`. Exclusion will cause the following behaviour for these:
  - `fit(init_state=...)` recomputes exclusions from the data it is given, so rows and
    columns that are now covered come back.
  - `infer()` keeps the columns excluded in training and excludes new rows with too
    little coverage.
  - `score()` treats excluded entries as zero-weight and reports how many there were
    (`n_excluded_rows`, `n_excluded_cols`, `n_excluded_entries`).

If you want to prevent exclusion from occurring (i.e. keep all rows and columns in all cases),
you can keep the model defined with **regularisation** by setting `als_ridge`. The undetermined
entries of `A` and `G` are then shrunk towards zero. The larger the ridge is, the more this
also impacts other entries of `A` and `G`. An `UnderdeterminedWarning` will tell you which
parts of the model are determined only by the ridge and not the data. If the ridge is too small
to register at the level of precision, a warning will give the minimum usable value. For example,
if you are working with float32 (firstly, probably don't), the ridge will need to be larger than
that at float64. With a ridge, the loss being minimised includes the ridge penalty, so the
rising-loss check described below is not applied.

As a final note for the nerds, you can easily mix this behaviour by setting entries of `Y` to zero and the corresponding entries of `W` to small numbers where you want regularisation. `W` can then be zero where you want exclusion. In this case, don't use `als_ridge`.

### Convergence and warnings

`robusta-hmf` may issue any of the following warnings:

| Warning | Means |
|---|---|
| `DataWarning` | `NaN` or `inf` input entries were treated as missing. |
| `UnderdeterminedWarning` | Some rows or columns have fewer non-zero weights than `rank`. |
| `ConvergenceWarning` | The fit stopped at `max_iter` without converging, or the loss function started increasing. |

`model.fit_info` records how the last `fit()` went: `converged`, `stop_reason`, `n_iter`,
`final_loss`, `loss_rises` and `exclusions`. `infer()` stores nothing on the model, since one
model is often applied to many datasets; its warnings name the number of rows instead.

If you wish for these warnings to instead raise Errors, you can do that simply as follows:

```python
import warnings
from robusta_hmf import ConvergenceWarning
warnings.simplefilter("error", ConvergenceWarning)
```

### If the loss rises or diverges

RHMF is an iterative algorithm, and each step involves solving linear systems. In exact
arithmetic the loss always decreases. Unfortunately, reality is harsher and matrix solves 
in floating point arithmetic suffer from round-off.

When the loss rises by more than the expected jitter caused by round-off, it indicates that
the matrix solve in one of the iteration steps has lost accuracy (often because of a
growing condition number), and a `ConvergenceWarning` is issued.

Some possible causes of convergence issues are:

1. **float32.** Try the fit in float64. If that behaves, the float32 fit was
   running out of precision in badly conditioned solves.
2. **Regularisation too small.** If you set `als_ridge`, check for the warning above, and
   either raise the ridge or remove it so under-covered rows and columns are excluded. Note
   that if you are manually regularising by using tiny weights, `robusta` will not perform
   any check for you that the values are large enough for the working precision.
3. **The rank is too large.** Rows or columns with coverage only just above `rank`
   are determined, but poorly. Try a lower rank.
4. **Your dataset is crazy.** In some cases, the initial guess provided by `robusta` will
   be very bad, causing very badly conditioned matrix solves. This is problem specific,
   and so you may have to explore other initialisation strategies.

If the loss becomes NaN or infinite, the fit stops with a `FloatingPointError` that will
state the likely cause.

### Precision

You should use float64 by default. JAX only provides float64 once
`jax.config.update("jax_enable_x64", True)` has been called, before any JAX array is
created. Without it, `fit()`, `infer()` and `score()` raise `PrecisionError` instead of
silently computing in float32.

float32 is supported as an explicit choice, `Robusta(..., dtype="float32")`, and is much
faster on GPUs. It is also less forgiving of badly conditioned solves as explained above.

### Model selection / cross validation

`score()` returns the statistics used in the paper. There, we recommended using the KL
divergence of the standardised residuals from a standard Normal distribution `N(0, 1)`.
For this score, lower is better, and so you should choose a `rank` (K in the paper) and
`robust_scale` (Q in the paper) that performs well in terms of this score. `score()`
also provides some other potentially useful diagnostics although we **do not** recommend
using them for model selection. Excluded entries count as zero residuals in these scores, so
compare scores across ranks only when `n_excluded_entries` is 0 (or the same) for each.

```python
infer_state, _ = model.infer(Y_test, W_test, state=state)
scores = model.score(Y_test, W_test, state=infer_state)

scores.kl        # KL divergence from N(0, 1); the model-selection criterion
scores.std_z     # sigma_z, target 1
scores.chi2_red  # reduced chi-squared, target 1
```

In practice, you may get quite good performance just picking the `rank` from some heuristic
arguments (how much diversity do you expect?), and a `robust_scale` between approx. 2 and 5.

### Fitting data larger than one GPU

Each iteration in the optimisation over the whole `(N, M)` data materialises about five full-size
intermediates on top of `Y` and `W`. As an example, for a problem of size approx. 500,000 x 2500 in
float64, this will be about 50 GB of memory. This can be (naively) too much for a single GPU.

**Chunking allows you to still run on one GPU**. On top of this, one can **shard** across GPUs
to increase the speed of the solve. Both of these are implemented via `ShardedFrame`:

```python
from robusta_hmf import Robusta, ShardedFrame

model = Robusta(
    rank=10,
    robust_scale=5.0,
    override_frame=ShardedFrame(row_block=4096),  # devices default to all visible
)
state, loss_history = model.fit(Y, W)
```

Everything else (`infer`, `synthesize`, `score`, `coefficients`, `residuals`) works
the same as usual. `Y` and `W` are stored on device in float32 by default while all
arithmetic happens in the model's `dtype` (float64 unless you opt in to float32),
halving local memory used by `Y` and `W`.

To fit several models to one sample (say a `(K, Q)` grid) without re-transferring the data,
use the following pattern:

```python
frame = ShardedFrame(row_block=4096)
data, _ = frame.prepare(Y, W)          # one transfer to device
for rank in ranks:
    model = Robusta(rank=rank, override_frame=frame)
    state, _ = model.fit(data)
```

The data are checked for `NaN` and `inf` when `prepare()` places them, so pass
`frame.prepare(Y, W, nan_error=True)` there if you want an error rather than a warning.

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
