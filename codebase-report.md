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
- `model.synthesize(state=...)` -> `A @ G.T` — reconstruct data
- `model.residuals(Y, state=...)` -> `Y - A @ G.T`
- `model.robust_weights(Y, W, state=...)` -> per-pixel weights in [0, 1]
- `model.basis_vectors(state=...)` -> G matrix (M x K)
- `model.coefficients(state=...)` -> A matrix (N x K)

### Mathematical Model

**Matrix factorization**: Y ≈ A @ G.T where Y is (N x M), A is (N x K), G is (M x K).

**Heteroskedastic**: Each pixel (i,j) has its own noise variance, encoded as weight W_ij = 1/σ²_ij.

**Robust**: Student-t likelihood downweights outliers via IRLS. Robust weight for pixel (i,j):
```
w_ij = 1 / (1 + |r_ij|² / Q)
```
where r_ij is the residual and Q is the scale parameter. Total weight = W_ij * w_ij.

**Optimization**:
- ALS (default): Alternating weighted least squares on A and G. Row/column-wise linear solves via `jax.vmap`.
- SGD: Gradient descent via optax (adafactor by default). Uses `eqx.filter_value_and_grad`.

**Rotation**: After each ALS step, apply orthogonal rotation for identifiability (FastAffine uses eigendecomposition, SlowAffine uses SVD).

### Module Map

```
src/robusta_hmf/
├── __init__.py          # Public exports: Robusta, RHMFState, save/load
├── main.py              # Robusta class (user-facing wrapper)
├── hmf.py               # HMF class (internal optimization engine)
├── state.py             # RHMFState dataclass, update/refresh helpers
├── als.py               # WeightedAStep, WeightedGStep (ALS solvers)
├── likelihoods.py       # Likelihood, GaussianLikelihood, StudentTLikelihood, CauchyLikelihood
├── rotations.py         # Rotation, Identity, FastAffine, SlowAffine
├── regularisers.py      # Placeholder (unimplemented)
├── init.py              # Initialization strategies (SVD, random)
└── py.typed             # PEP 561 marker
```

### Tests

Located in `tests/`:
- `test_als.py`: Verifies ALS step solutions satisfy normal equations (parametric over shapes)
- `test_rotation.py`: Verifies rotation preserves A @ G.T product
- `test_hmf.py`: End-to-end step tests (currently commented out)

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

### Current State

- Pipeline is functional but somewhat WIP
- Default model grid is narrow (K=[5], Q=[3.0]) — needs expanding for full analysis
- May need cleanup/refactoring to match toy example's patterns
- `w_quantiles.py` is known to be incorrect — ignore it
- Overlapping bins don't deduplicate outlier detections across bins

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
| equinox | Module system, filtered transforms (see equinox-report.md) |
| optax | Gradient-based optimizers (adafactor for SGD mode) |
| pandas | DataFrame operations (outlier summaries) |
| numpy | Array operations in examples |
| matplotlib / mpl_drip | Plotting |
| h5py | Reading Gaia HDF5 data |
| polars | Efficient data joining in Gaia example |
| scipy | Statistical functions in examples |
