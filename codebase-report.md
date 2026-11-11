# Codebase Report: robusta-hmf

## Library Architecture (src/robusta_hmf/)

### Public API

The main user-facing class is `Robusta` (in `main.py`), which wraps the internal `HMF` engine:

```python
from robusta_hmf import Robusta

model = Robusta(
    rank=5,                    # Number of latent factors (K)
    robust=True,               # Use Student-t likelihood (vs Gaussian)
    robust_scale=3.0,          # Student-t scale parameter (Q)
    robust_nu=1.0,             # Degrees of freedom (1.0 = Cauchy)
    method="als",              # "als" or "sgd"
    init_strategy="svd",       # Initialization method
    conv_strategy="max_frac_G",# Convergence criterion
    conv_tol=1e-4,
    rotation="fast",           # Rotation method for identifiability
)
```

Key methods:
- `model.fit(Y, W, max_iter=1000)` -> `(RHMFState, loss_history)` — train on data
- `model.infer(Y_infer, W_infer, state=...)` -> `(RHMFState, loss_history)` — infer coefficients with fixed basis
- `model.score(Y, W, state=...)` -> `ZMoments` — held-out KL / std_z / chi2_red / rmse
- `model.synthesize(state=...)` -> `A @ G.T` — reconstruct data
- `model.residuals(Y, state=...)` -> `Y - A @ G.T`
- `model.robust_weights(Y, W, state=...)` -> per-pixel weights in [0, 1]
- `model.basis_vectors(state=...)` -> G matrix (M x K)
- `model.coefficients(state=...)` -> A matrix (N x K)

Three `override_*` hooks swap out a component without touching the rest of the
configuration: `override_initialiser`, `override_conv_tester`, `override_frame`.

### Model / execution split

`Robusta` owns configuration only. It builds an `HMF` (the likelihood, rotation and ALS
steps — *what* an iteration computes) and an `OptFrame` (*how* the loop is scheduled), and
delegates. `OptFrame` exposes five seams a scheduling subclass can override — `prepare()`,
`make_initialiser()`, `initial_coefficients()`, `z_moments()` and `run()` — plus `bind()`,
which wires a user-constructed frame to the model so neither holds a duplicate parameter
list.

`ShardedFrame` (in `sharded.py`) is the one alternative frame. It chunks the row axis and
shards it across a device mesh, so a matrix larger than one GPU's memory fits:

```python
model = Robusta(rank=10, override_frame=ShardedFrame(row_block=4096))
state, loss = model.fit(Y, W)   # numpy arrays; everything else unchanged
```

Chunking alone is the memory fix (peak scratch goes from `O(N*M)` to `O(row_block*M)`);
sharding is a speed knob on top, and a one-device mesh is the default. Because the ALS
kernels in `als.py` build the normal equations as matmuls against `K*K` outer products,
both the A step and the G-step accumulation are row-local, and one `psum` of the small
`(M, K, K)` and `(M, K)` blocks per iteration is the only communication. Results match the
dense path to round-off; see `tests/test_sharded.py`.

### Mathematical Model

**Matrix factorization**: Y ≈ A @ G.T where Y is (N x M), A is (N x K), G is (M x K).

**Heteroskedastic**: Each pixel (i,j) has its own noise variance, encoded as weight W_ij = 1/σ²_ij.

**Robust**: Student-t likelihood downweights outliers via IRLS. Robust weight for pixel (i,j):
```
w_ij = 1 / (1 + |r_ij|² / Q)
```
where r_ij is the residual and Q is the scale parameter. Total weight = W_ij * w_ij.

**Optimization**:
- ALS (default): Alternating weighted least squares on A and G. The batched normal
  equations are built as two matmuls against the `K*K` outer products of the fixed factor
  (`als.normal_equations`), not by whitening per row — same flops, but the peak
  intermediate is `batch*K*K` rather than `batch*M*K`, and the row axis stays shardable.
  `als.solve_normal_equations` substitutes the identity where the weights are all zero, so
  a fully-masked row or pixel returns a zero factor instead of NaN-ing the whole batch.
- SGD: Gradient descent via optax (adafactor by default). Uses `eqx.filter_value_and_grad`.
  Not supported by `ShardedFrame`.

**Rotation**: After each ALS step, apply orthogonal rotation for identifiability (FastAffine uses eigendecomposition, SlowAffine uses SVD).

### Module Map

```
src/robusta_hmf/
├── __init__.py          # Public exports
├── main.py              # Robusta class (user-facing wrapper)
├── hmf.py               # HMF class (what an iteration computes)
├── frame.py             # OptFrame: the loop, plus the seams a scheduler overrides
├── sharded.py           # ShardedFrame, GramInitialiser, RowPlan, ShardedData
├── metrics.py           # ZMoments, kl_from_moments, z_scores, z_moments, mad_z
├── state.py             # RHMFState dataclass, update/refresh helpers
├── als.py               # normal_equations, solve_normal_equations, WeightedAStep/GStep
├── likelihoods.py       # Likelihood, GaussianLikelihood, StudentTLikelihood, CauchyLikelihood
├── rotations.py         # Rotation, Identity, FastAffine, SlowAffine
├── initialisation.py    # Initialization strategies (SVD, random, custom)
└── py.typed             # PEP 561 marker
```

### Tests

Located in `tests/`, one file per module. `conftest.py` sets `XLA_FLAGS` before jax
initialises so multi-device tests run on forced host CPU devices.

- `test_als.py`: ALS steps satisfy the normal equations (parametric over shapes); degenerate
  zero-weight rows and pixels
- `test_sharded.py`: mostly equivalence against the dense path, parameterised over device
  count, `row_block`, padded/exact row counts, likelihood, ridge and dtype
- `test_metrics.py`: metric definitions pinned against the inline versions in the paper's
  analysis scripts
- `test_robusta.py`: integration tests for the public API
- `test_rotations.py`, `test_hmf.py`, `test_convergence.py`, `test_initialisation.py`,
  `test_likelihoods.py`, `test_state.py`

`test_als.py` deliberately includes shapes with `K > min(N, M)`, where the normal equations
are singular by construction. `solve_normal_equations` returns the minimum-norm solution there
(it is told the fixed factor's row count via `n_obs`), rather than relying on
`jnp.linalg.solve`, whose output for singular systems differs across jax versions.

---

## Toy Example (examples_paper/toy/)

**Purpose**: Synthetic validation demonstrating RHMF's ability to handle outliers and heteroskedastic noise.

### Pipeline

1. **Data generation** (`gen_data.py`):
   - 8000 spectra, 1200 pixels, 5 true basis functions (3 Legendre polynomials + absorption line template + sinusoidal)
   - Heteroskedastic noise: per-spectrum, per-wavelength, and per-pixel variance components
   - 5 outlier types: spectrum outliers (40), pixel outliers (0.4%), column outliers (3 columns), absorption line outliers (10 spectra), missing data segments (50% of spectra)
   - 50/50 train/test split

2. **Model fitting** (`run_toy_gen_and_fits.py`):
   - Grid search over K ∈ {3,4,5,6,7} and Q ∈ {0.5,1,2,3,4,5,10} = 35 models
   - Each model: `Robusta(rank=K, robust_scale=Q, method="als", init_strategy="svd")`
   - Train on training set, save states as .npz

3. **Analysis** (`analyse_toy.py`):
   - Compares RHMF against PCA and Robust PCA (RPCA)
   - Evaluates: CV z-score metric, per-pixel F1, per-object F1
   - Generates ~15 publication-quality plots: reconstructions, basis functions, weight histograms, heatmaps, absorption line diagnostics, hyperparameter selection grids

### Key Result

RHMF correctly identifies and downweights all outlier types via robust weights, while maintaining accurate reconstructions for clean spectra. The CV metric enables data-driven selection of optimal (K, Q).

---

## Gaia RVS Example (examples_paper/gaia_rvs/)

**Purpose**: Real-data outlier identification in Gaia DR3 RVS stellar spectra.

### Data

- ~3M+ Gaia DR3 RVS spectra (846-870 nm, calcium triplet region)
- 2401 pixels per spectrum, clipped to 2321 (removing 40 edge pixels)
- Metadata: parallax, BP-RP color, G magnitude -> absolute magnitude for HR diagram placement

### Pipeline

1. **Data loading** (`collect.py`): `MatchedData` class joins HDF5 spectra with CSV metadata via Polars
2. **Binning** (`bins.py`): 14 overlapping elliptical bins along the main sequence in the HR diagram
3. **Training** (`train_bins.py`): Train Robusta models per bin over (K, Q) grid
4. **Analysis** (`analyse_bins.py` -> `analysis_funcs.py`):
   - `compute_bin_analysis()`: CV scoring, model selection, outlier detection
   - `plot_bin_analysis()`: Generate all diagnostic plots (no recomputation)
   - `BinAnalysis` dataclass holds all results per bin

### Compute/Plot Split Pattern

The key architectural pattern:
- `BinAnalysis` dataclass holds ALL computed results (data, CV scores, best model, reconstructions, outlier scores, weights)
- `compute_bin_analysis()` does expensive work (inference, CV, outlier detection), returns `BinAnalysis`
- `plot_bin_analysis()` takes `BinAnalysis` + paths, generates plots with zero recomputation
- `analyse_bin()` is a thin wrapper: compute then plot

### Indexing Invariant

All per-spectrum arrays share ordering: `all_Y[i]`, `all_W[i]`, `source_ids[i]`, `outlier_scores[i]`, etc. Train/test indices index into these arrays.

### CV Metrics

Four metrics computed on test set: `std_z` (std of z-scores, target 1.0), `chi2_red` (reduced chi-squared, target 1.0), `rmse` (weighted RMSE, lower better), `mad_z` (median |z|, target 0.6745).

These now live in `robusta_hmf.metrics`, alongside the paper's KL score (Eq. 20) that
`examples_paper/toy/analyse_toy.py` computed inline. `analysis_funcs.py` still carries its
own copies; they are pinned by `tests/test_metrics.py` and can be deleted in favour of the
library versions.

### Current State

- Pipeline is complete; analyses for the paper are final.
- Former WIP items are now resolved or moot (analyses are finalised): the model grid was expanded as needed for the published analysis, cross-bin outlier overlap was handled, and `w_quantiles.py` is not used in the final pipeline.
- No open Gaia tasks remain.

### Supporting Scripts

- `plot_bins.py`: Visualize bin geometry on HR diagram
- `compare_metrics.py`: Compare outlier detection across CV metrics
- `collect_outlier_spectra.py`: Gather outlier plots for inspection
- `umap_residuals.py`: UMAP on outlier residuals
- `rvs_plot_utils.py`: Spectral line markers for plots

---

## Key Dependencies

| Package | Role |
|---------|------|
| jax | Array computation, JIT, autodiff, vmap |
| equinox | Module system, filtered transforms |
| optax | Gradient-based optimizers (adafactor for SGD mode) |
| pandas | DataFrame operations (outlier summaries) |
| numpy | Array operations in examples |
| matplotlib / mpl_drip | Plotting |
| h5py | Reading Gaia HDF5 data |
| polars | Efficient data joining in Gaia example |
| scipy | Statistical functions in examples |
