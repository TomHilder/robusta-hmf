# Tasks 8 & 9: Full Gaia Dataset Analysis

This directory contains scripts to complete the peer-review tasks for applying RHMF to the full Gaia main sequence dataset.

## Overview

**Task 8**: Apply RHMF to full Gaia main sequence (not binned)
- Trains RHMF models across a grid of (K, Q) values
- Evaluates on held-out test set
- Identifies outliers via robust weights
- **Estimated runtime**: 4–8 GPU hours for full grid

**Task 9**: Cross-validation for optimal (K, Q) selection
- Runs K-fold cross-validation on full dataset
- Computes KL divergence scores for model selection
- Identifies optimal hyperparameters
- **Estimated runtime**: 2–4 additional GPU hours (can run in parallel with Task 8)

## Prerequisites

1. **Data**: Must have run `collect.py` to create `gaia_rvs_results/collected_spectra.npz`
2. **Environment**: `uv run python` with GPU access (JAX + CUDA)
3. **Dependencies**: robusta_hmf, jax, numpy, matplotlib, scikit-learn, tqdm

## Quick Start

### Task 8: Full Dataset Training

```bash
cd examples_paper/gaia_rvs

# Default grid (recommended)
uv run python task_8_gaia_full_dataset.py

# Custom rank and Q values
uv run python task_8_gaia_full_dataset.py --ranks 5 6 7 8 9 10 --q-vals 3.0 4.0 5.0

# Run overnight (recommended for full grid)
nohup uv run python task_8_gaia_full_dataset.py > task_8.log 2>&1 &
```

**Outputs:**
- `gaia_rvs_results/full_dataset_rhmf_K{rank}_Q{Q}.npz` — Model states
- `gaia_rvs_results/full_dataset_analysis_{rank}_{Q}.npz` — Outlier lists & weights
- `plots_analysis/full_dataset_weights_K{rank}_Q{Q}.pdf` — Weight distributions

### Task 9: Cross-Validation

```bash
# Focused grid (3–4 hours, good for initial exploration)
uv run python task_9_gaia_cross_validation.py

# Full grid (comprehensive, ~4 GPU hours)
uv run python task_9_gaia_cross_validation.py --full-grid

# Run in parallel with Task 8 (separate GPU recommended)
uv run python task_8_gaia_full_dataset.py > task_8.log 2>&1 &
sleep 60  # Let Task 8 start
uv run python task_9_gaia_cross_validation.py --full-grid > task_9.log 2>&1 &
```

**Outputs:**
- `gaia_rvs_results/cross_validation_results.npz` — CV scores, optimal params
- `plots_analysis/cv_score_heatmap.pdf` — KL divergence grid
- `plots_analysis/outlier_count_heatmap.pdf` — Outlier count grid

## Arguments

### task_8_gaia_full_dataset.py

```
--ranks RANKS [RANKS ...]
    Rank values to test (default: 5 6 7 8 9 10)

--q-vals Q_VALS [Q_VALS ...]
    Robust scale values to test (default: 3.0 4.0 5.0)

--results-dir PATH
    Directory for model states (default: ./gaia_rvs_results)

--plots-dir PATH
    Directory for diagnostic plots (default: ./plots_analysis)
```

### task_9_gaia_cross_validation.py

```
--full-grid
    Use larger parameter grid (K=3-10, Q=2-6) [~4 GPU hours]
    Default grid: K=5-10, Q=3-5 [~2-3 GPU hours]

--ranks RANKS [RANKS ...]
    Rank values to test (overrides default/full-grid)

--q-vals Q_VALS [Q_VALS ...]
    Robust scale values to test (overrides default/full-grid)

--n-folds N
    Number of CV folds (default: 5)

--results-dir PATH
    Directory for results (default: ./gaia_rvs_results)

--plots-dir PATH
    Directory for plots (default: ./plots_analysis)
```

## Data Requirements

### Input: `collected_spectra.npz`

Expected structure (from `collect.py`):
```
spectra:   (N, M) float32  — N=~300k spectra, M=2361 pixels
ivar:      (N, M) float32  — inverse-variance weights
source_id: (N,) int64      — Gaia source identifiers
```

### Memory & Compute

| Grid Size | K Values | Q Values | Est. Time | Est. Memory |
|-----------|---------|---------|-----------|-------------|
| Small     | 1       | 1       | 30 min    | 4 GB        |
| Medium    | 6       | 3       | 3–4 hours | 8 GB        |
| Large     | 8       | 5       | 6–8 hours | 12 GB       |

## Monitoring Progress

### Task 8
```bash
# Watch output
tail -f task_8.log

# Expected output:
# Training RHMF: K=5, Q=3.0
# Training data shape: (150000, 2361)
# Fitting model...
# Converged after 42 iterations
# Final loss: 1.234567e+08
```

### Task 9
```bash
# Watch progress
tail -f task_9.log

# Shows CV fold progress bar and KL scores as computed
```

## Output Interpretation

### Task 8 Results

1. **Weight distribution** (`full_dataset_weights_K{K}_Q{Q}.pdf`)
   - Histogram of per-object median robust weights
   - Outliers: weight < 0.5 (red dashed line)
   - Outlier count and percentage printed to stdout

2. **State files** (`full_dataset_rhmf_K{K}_Q{Q}.npz`)
   - Can be loaded with `RHMFState.load(filename)`
   - Contains: A (coefficients), G (basis), and all model parameters

3. **Analysis files** (`full_dataset_analysis_{K}_{Q}.npz`)
   - `cv_score`: Test set RMSE
   - `outlier_indices`: Indices of low-weight spectra
   - `outlier_source_ids`: Gaia source IDs of outliers
   - `per_object_weights`: Median weight per spectrum

### Task 9 Results

1. **CV heatmap** (`cv_score_heatmap.pdf`)
   - Each cell: mean KL divergence across folds
   - Lower is better (darker = better)
   - Best (K, Q) marked by minimum

2. **Outlier heatmap** (`outlier_count_heatmap.pdf`)
   - Mean number of outliers per (K, Q)
   - Helps understand sensitivity to hyperparameter choices

3. **CV results** (`cross_validation_results.npz`)
   - `kl_scores`: (n_ranks, n_q, n_folds) raw scores
   - `kl_mean` / `kl_std`: Aggregated scores
   - `best_rank_kl`, `best_q_kl`: Optimal hyperparameters

## Typical Workflow

```bash
# Step 1: Ensure data is collected
cd examples_paper/gaia_rvs
ls -lh gaia_rvs_results/collected_spectra.npz

# Step 2: Run Task 8 (full grid, ~6–8 hours)
uv run python task_8_gaia_full_dataset.py \
    --ranks 5 6 7 8 9 10 \
    --q-vals 3.0 4.0 5.0

# Step 3: Check results
ls -lh gaia_rvs_results/full_dataset_*.npz
ls -lh plots_analysis/full_dataset_weights*.pdf

# Step 4: Run Task 9 (cross-validation)
uv run python task_9_gaia_cross_validation.py --full-grid

# Step 5: Check optimal parameters
python -c "
import numpy as np
cv = np.load('gaia_rvs_results/cross_validation_results.npz')
print(f'Best K: {cv[\"best_rank_kl\"]}')
print(f'Best Q: {cv[\"best_q_kl\"]:.2f}')
print(f'Best KL: {cv[\"best_kl\"]:.6f}')
"

# Step 6: Generate paper figures
uv run python make_paper_figs.py
```

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `CUDA out of memory` or `RuntimeError: Unable to allocate`

**Solutions**:
1. Reduce dataset size (use earlier bins from `analyse_bins.py`)
2. Reduce rank: `--ranks 5 6 7` instead of 5–10
3. Reduce Q values: `--q-vals 4.0 5.0` instead of 3–6
4. Use smaller folds: `--n-folds 3` instead of 5
5. Run one (K, Q) pair at a time manually

### Model not converging

**Symptom**: Loss doesn't decrease after first few iterations

**Solutions**:
1. Increase `max_iter`: edit scripts to use `max_iter=500`
2. Decrease learning rate (if available in Robusta)
3. Try different Q values (lower Q = more robust weighting)

### Data file not found

**Symptom**: `FileNotFoundError: collected_spectra.npz`

**Solution**: Run `collect.py` first
```bash
uv run python collect.py
```

## Extending to New Datasets

### Using these scripts with other data

1. Replace `collected_spectra.npz` with your own data
2. Ensure columns are `(spectra, ivar, source_id)`
3. Adjust `N_CLIP_PIX` in `gaia_config.py` if needed
4. Run tasks as above

### Custom hyperparameters

Edit default grids in script `main()` function:
```python
ranks = [3, 4, 5, 6, 7, 8, 9, 10]
q_vals = [2.0, 3.0, 4.0, 5.0, 6.0]
```

## Integration with Paper

After completing Tasks 8 & 9:

1. Update `paper/main.tex` section 3 (Experiments → Gaia Analysis) with:
   - Optimal (K, Q) values from Task 9
   - Outlier counts and percentages from Task 8
   - CV score plots from Task 9

2. Regenerate paper figures:
   ```bash
   uv run python make_paper_figs.py
   ```

3. Update referee response (`paper/referee-response.txt`) to note:
   - Full dataset analysis completed
   - Optimal hyperparameters selected via cross-validation
   - Outlier identification results on full main sequence

## References

- Hilder et al. (in prep): "Robust Heteroskedastic Matrix Factorization"
- Original toy example analysis: `examples_paper/toy/analyse_toy.py`
- Gaia RVS data: Gaia Collaboration, https://www.gaia.ac.uk/

## Questions?

Check script docstrings:
```bash
uv run python task_8_gaia_full_dataset.py --help
uv run python task_9_gaia_cross_validation.py --help
```

Or see inline comments in task_*.py files.
