"""
Train RHMF models on the full Gaia RVS sample over a (K, Q) grid, for the
referee-requested full-dataset + CV analysis.

Two sample choices (--sample):
    ms   (default) union of all 14 main-sequence bins from gaia_config.py,
         deduplicated -- the paper's main-sequence selection, unbinned.
    all  every spectrum in the RVS file (999645), i.e. the whole sample with
         no colour-magnitude selection at all -- not even the finite-photometry
         and positive-parallax cuts MatchedData applies by default, which only
         exist so a star can be placed on the HR diagram for the binning.

Everything else (edge clipping, train/test split seed, NaN handling, model
settings) mirrors train_bins.py exactly. State files use the per-bin naming
convention with bin tag "full_ms" or "full_rvs", so all analysis_funcs
helpers work unchanged.

Each (K, Q) model is skipped if its state file already exists, which gives
free resume-after-interrupt AND lets multiple GPUs share one grid safely.

USAGE (single GPU):
    uv run python train_full_ms.py [--sample all]

USAGE (shard the grid across GPUs, one process per GPU):
    CUDA_VISIBLE_DEVICES=0 uv run python train_full_ms.py --shard 0 --n-shards 2 &
    CUDA_VISIBLE_DEVICES=1 uv run python train_full_ms.py --shard 1 --n-shards 2 &

or just use run_full_ms.sh, which does this for every visible GPU.

Run analyse_full_ms.py afterwards (same --sample) for CV scores, best-model
selection, and outlier identification.
"""

import argparse
from pathlib import Path

import gaia_config as cfg
import jax
import numpy as np
from analysis_funcs import (
    build_bins_from_config,
    clip_edge_pix,
    get_test_train_split_idx,
    nans_mask,
)
from tqdm import tqdm

from robusta_hmf import Robusta, save_state_to_npz

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# The full sample is far more heterogeneous than any single bin (the per-bin
# analysis uses K=10), so the grid extends to larger ranks.
RANKS = [5, 10, 15, 20, 25, 30]
Q_VALS = [2.0, 3.0, 5.0, 7.5]

MAX_ITER = 1000
TRAIN_FRAC = cfg.TRAIN_FRAC

RESULTS_DIR = Path("./gaia_rvs_results")

# Bin tag used in state filenames: converged_state_R{K}_Q{Q}_bin_{tag}.npz
SAMPLE_TAGS = {"ms": "full_ms", "all": "full_rvs", "all_filtered": "full_rvs"}

# Numeric precision. The per-bin analysis in the paper ran on CPU, where float32
# matmuls are exact; on Ampere and later GPUs jax defaults to TF32, which keeps
# only ~10 mantissa bits. The ALS steps build normal equations by summing over
# every spectrum, so at ~5e5 rows that is not enough precision and the loss can
# fail to decrease monotonically.
#   tf32  jax default on GPU -- fastest, least accurate
#   fp32  float32 storage, full-precision float32 matmuls (no memory cost)
#   fp64  float64 throughout -- most accurate, 2x memory, and much slower on
#         cards without fast fp64 (e.g. 1:32 on an A6000 vs 1:2 on an A100/H100)
PRECISIONS = ("tf32", "fp32", "fp64")
DEFAULT_PRECISION = "fp64"

# ============================================================================ #


def configure_precision(precision):
    """Apply a precision setting. Must run before any JAX array is created."""
    if precision == "fp64":
        jax.config.update("jax_enable_x64", True)
    elif precision == "fp32":
        jax.config.update("jax_default_matmul_precision", "highest")
    elif precision != "tf32":
        raise ValueError(f"Unknown precision: {precision!r} (use one of {PRECISIONS})")
    return np.float64 if precision == "fp64" else np.float32


def build_sample(sample="ms"):
    """Return (data, idx, ids, tag) for the requested sample.

    "ms":  union of all main-sequence bins, deduplicated (bins overlap since
           widths exceed spacing), idx/ids kept aligned.
    "all": every spectrum in the RVS file, with the HR-diagram metadata
           filters off (see the module docstring). 999,645 spectra.

    "all_filtered": the same, with the two metadata filters left on, which is
           993,910 spectra. This is what "all" meant before the filters were
           turned off, and it is the sample the saved full-RVS fit
           (``full_rvs_final_weights.npz`` and its converged state, both 993,910
           rows) was actually made on. Anything reading those files back has to
           rebuild the sample this way or the rows do not line up; the assertion
           in ``plot_final_outlier_spectra.load_outlier_inputs`` catches it.
           Retire this once the final fit is redone on the full 999,645.
    """
    if sample == "ms":
        data, bins, _, _ = build_bins_from_config()
        all_idx = np.concatenate([b.idx for b in bins])
        all_ids = np.concatenate([b.ids for b in bins])
        _, first = np.unique(all_idx, return_index=True)
        return data, all_idx[first], all_ids[first], SAMPLE_TAGS["ms"]
    elif sample in ("all", "all_filtered"):
        from collect import MatchedData

        on = sample == "all_filtered"
        data = MatchedData(filter_nans=on, filter_neg_parallax=on)
        idx = np.arange(len(data.spectra_indices))
        return data, idx, data["source_id"], SAMPLE_TAGS[sample]
    raise ValueError(f"Unknown sample: {sample!r} (use 'ms', 'all' or 'all_filtered')")


def load_full_ms_training_data(data, idx, train_frac=TRAIN_FRAC, dtype=np.float32):
    """Training Y, W for the full-MS sample, mirroring train_bins.train_bin."""
    train_idx, _ = get_test_train_split_idx(len(idx), train_frac=train_frac)
    train_flux, train_u_flux = clip_edge_pix(*data.get_flux_batch(idx[train_idx]))
    train_weights = 1.0 / (train_u_flux**2)

    Y, W = train_flux, train_weights
    spec_nans_mask = nans_mask([Y, W])
    Y[~spec_nans_mask] = np.nan
    W[~spec_nans_mask] = np.nan
    Y = np.nan_to_num(Y)
    W = np.nan_to_num(W)
    # The HDF5 flux is float32, and jax preserves input dtypes -- without this
    # cast, jax_enable_x64 silently has no effect and the fit stays in float32.
    return Y.astype(dtype, copy=False), W.astype(dtype, copy=False)


def main(ranks, q_vals, shard, n_shards, sample="ms", max_iter=MAX_ITER,
         results_dir=RESULTS_DIR, precision=DEFAULT_PRECISION):
    dtype = configure_precision(precision)
    print(f"Precision: {precision} (dtype {np.dtype(dtype).name}, "
          f"device {jax.devices()[0].device_kind})")

    print(f"Building sample '{sample}'...")
    data, idx, ids, tag = build_sample(sample)
    print(f"Sample '{tag}': {len(idx)} unique spectra")

    print("Loading training spectra...")
    Y, W = load_full_ms_training_data(data, idx, dtype=dtype)
    print(f"Training data: {Y.shape[0]} spectra x {Y.shape[1]} pixels, "
          f"{Y.dtype} ({(Y.nbytes + W.nbytes) / 2**30:.1f} GiB for Y+W)")

    # Grid, sharded round-robin so shards are balanced across ranks.
    Q_grid, Rank_grid = np.meshgrid(q_vals, ranks)
    grid = list(zip(Q_grid.flatten(), Rank_grid.flatten()))
    my_grid = [g for i, g in enumerate(grid) if i % n_shards == shard]
    print(f"Shard {shard}/{n_shards}: {len(my_grid)} of {len(grid)} models")

    results_dir.mkdir(parents=True, exist_ok=True)
    n_trained = n_skipped = 0
    for Q, rank in tqdm(my_grid, desc=f"{tag} shard {shard}"):
        state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_bin_{tag}.npz"
        if state_file.exists():
            n_skipped += 1
            continue

        model = Robusta(
            rank=rank,
            robust_scale=Q,
            conv_strategy="max_frac_G",
            conv_tol=1e-4,
            init_strategy="svd",
            rotation="fast",
            target="G",
            whiten=True,
        )
        state, loss = model.fit(
            Y,
            W,
            max_iter=max_iter,
            conv_check_cadence=5,
        )
        save_state_to_npz(state, state_file)
        n_trained += 1

    print(f"Done: trained {n_trained}, skipped {n_skipped} (already existed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--q-vals", type=float, nargs="+", default=Q_VALS)
    parser.add_argument("--sample", choices=["ms", "all"], default="ms",
                        help="'ms' = main-sequence bin union; 'all' = whole RVS sample")
    parser.add_argument("--shard", type=int, default=0, help="This process's shard index")
    parser.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--precision", choices=PRECISIONS, default=DEFAULT_PRECISION,
                        help="Numeric precision (default: %(default)s). tf32 is jax's "
                             "GPU default and is too coarse for normal equations over "
                             "~5e5 spectra; fp32 forces exact float32 matmuls at no "
                             "memory cost; fp64 doubles memory and is slow without "
                             "fast fp64 hardware.")
    args = parser.parse_args()

    if not (0 <= args.shard < args.n_shards):
        raise SystemExit(f"--shard must be in [0, {args.n_shards})")

    main(args.ranks, args.q_vals, args.shard, args.n_shards, sample=args.sample,
         max_iter=args.max_iter, precision=args.precision)
