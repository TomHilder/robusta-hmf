# Full Main-Sequence Analysis (referee items 9 & 12)

Applies RHMF to the **entire** Gaia RVS main-sequence sample (union of the 14
bins, deduplicated — not binned) with cross-validation over a (K, Q) grid,
then identifies outliers with the best model. Everything reuses the proven
per-bin machinery in `analysis_funcs.py`: same edge clipping, same train/test
seed, same CV metrics, same 1st-percentile outlier score.

## Requirements

- The machine with the Gaia RVS data that `collect.py`/`MatchedData` reads
  (the same machine used for `train_bins.py`).
- One or more CUDA GPUs with JAX installed in the uv environment.

## Run everything

```bash
cd examples_paper/gaia_rvs
./run_full_ms.sh                # uses every GPU nvidia-smi reports
N_GPUS=1 ./run_full_ms.sh       # force single GPU
```

The script trains the (K, Q) grid — sharded round-robin across GPUs, one
process per GPU — waits for all shards, then runs the analysis stage.
**Safe to re-run after any interruption**: completed models are detected by
their state files and skipped.

Default grid: `K ∈ {5, 10, 15, 20, 25, 30}`, `Q ∈ {2, 3, 5, 7.5}` (24
models). Shrink it for a first pass:

```bash
./run_full_ms.sh --ranks 10 20 30 --q-vals 3.0 5.0
```

## Or run the stages manually

```bash
# Training (single GPU)
uv run python train_full_ms.py

# Training (2 GPUs by hand)
CUDA_VISIBLE_DEVICES=0 uv run python train_full_ms.py --shard 0 --n-shards 2 &
CUDA_VISIBLE_DEVICES=1 uv run python train_full_ms.py --shard 1 --n-shards 2 &
wait

# CV + best model + outliers (needs all grid models present)
uv run python analyse_full_ms.py
```

## Outputs

| File | Contents |
|---|---|
| `gaia_rvs_results/converged_state_R{K}_Q{Q}_bin_full_ms.npz` | trained models |
| `gaia_rvs_results/full_ms_cv_scores.npz` | std_z / chi2_red / rmse / mad_z over the grid |
| `gaia_rvs_results/inferred_all_data_R{K}_Q{Q}_bin_full_ms.npz` | best-model inference on all spectra |
| `gaia_rvs_results/full_ms_outliers.csv` | source_id + score per outlier, sorted worst-first |
| `plots_full_ms/cv_heatmaps.pdf` | CV metrics across (K, Q) |
| `plots_full_ms/weights_hist.pdf` | outlier-score distribution |
| `plots_full_ms/basis_vectors.pdf` | best-model eigenspectra |

## Monitoring

```bash
tail -f full_ms_shard0.log      # training progress (one log per GPU shard)
tail -f full_ms_analyse.log     # analysis stage
```

## After it finishes

1. Compare `full_ms_cv_scores.npz` best (K, Q) with the per-bin choice (K=10,
   Q=5) — this answers referee item 12.
2. Cross-match `full_ms_outliers.csv` against the per-bin outlier lists —
   this answers referee item 9 (does the binned analysis miss anything?).
3. `plots_full_ms/basis_vectors.pdf` gives the Gaia eigenspectra requested in
   major comment 2.
4. While on this machine, also regenerate the Fig 6 label fix:
   `uv run python plot_hist_stack.py` and copy `stacked_hist.pdf` to
   `paper/figs/` (one minute, CPU-only).
