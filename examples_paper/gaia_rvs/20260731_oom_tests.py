"""Fit one Robusta model to the entire Gaia RVS sample in float64.

The obstacle is scratch memory, not the model. ``Robusta.fit`` evaluates each
ALS step over the whole matrix at once, so a step holds ~5 full ``(N, M)``
intermediates. At N = 499_822 training rows and M = 2321 pixels that is 8.6 GiB
each in float64, i.e. ~43 GiB of scratch on top of 17 GiB of resident Y and W:
hence every 5th spectrum fits on a 48 GiB A6000 and every 4th does not.

:mod:`distributed_robusta` restructures the same ALS iteration so that

  * the row axis is traversed in chunks -- peak scratch becomes
    ``O(row_block * M)`` rather than ``O(N * M)``, ~0.4 GiB at row_block=4096;
  * the row axis is sharded over all visible GPUs, with only a small
    ``(M, K, K)`` all-reduce per iteration;
  * Y and W live on device as float32 (they are float32 measurements) while
    every accumulation stays in float64.

The arithmetic is unchanged: ``distributed_robusta.compare_to_reference``
reproduces the library's loss trajectory to ~1e-12.

USAGE
    # all visible GPUs, whole RVS sample, K=10, Q=2
    uv run python 20260731_oom_tests.py --sample all --rank 10 --q 2

    # reproduce the OOM with the stock library implementation
    uv run python 20260731_oom_tests.py --engine reference --subsample 4

Under Slurm ask for the GPUs and let JAX see all of them -- do NOT set
CUDA_VISIBLE_DEVICES to a single device:

    sbatch -p gpu --gpus=4 -c 16 --mem=256G -t 4:00:00 --wrap \
      "cd $PWD && UV_NO_SYNC=1 uv run python -u 20260731_oom_tests.py --sample all"

If a GPU is shared with anything else, cap the pool with
``XLA_PYTHON_CLIENT_MEM_FRACTION=0.9``.
"""

import argparse
import time
from pathlib import Path

import gaia_config as cfg
import jax
import numpy as np

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

MAX_ITER = 1000
TRAIN_FRAC = cfg.TRAIN_FRAC

RESULTS_DIR = Path("./gaia_rvs_results")

# Bin tag used in state filenames: converged_state_R{K}_Q{Q}_bin_{tag}.npz
SAMPLE_TAGS = {"ms": "full_ms", "all": "full_rvs"}

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
    "all": every spectrum in the matched RVS catalogue.
    """
    from analysis_funcs import build_bins_from_config

    if sample == "ms":
        data, bins, _, _ = build_bins_from_config()
        all_idx = np.concatenate([b.idx for b in bins])
        all_ids = np.concatenate([b.ids for b in bins])
        _, first = np.unique(all_idx, return_index=True)
        return data, all_idx[first], all_ids[first], SAMPLE_TAGS["ms"]
    elif sample == "all":
        from collect import MatchedData

        data = MatchedData()
        idx = np.arange(len(data.spectra_indices))
        return data, idx, data["source_id"], SAMPLE_TAGS["all"]
    raise ValueError(f"Unknown sample: {sample!r} (use 'ms' or 'all')")


def load_training_data(
    data, idx, ids, train_frac=TRAIN_FRAC, dtype=np.float32, block=65536, subsample=1
):
    """Training Y, W (and the matching source ids) for the requested sample.

    Same masking as ``train_bins.train_bin``, but assembled block by block
    straight into preallocated float32 output arrays. The original route --
    read all 5e5 spectra, then ``nans_mask([Y, W])`` -- stacks Y and W into one
    (2, N, M) array and then copies again through ``nan_to_num``, which peaks
    at several times the size of the data on the host.

    Blocks are read in HDF5 order (``get_flux_batch`` reads sorted contiguous
    slabs; feeding it a scattered subset would make every slab sparse) and
    scattered back into the caller's order.
    """
    from analysis_funcs import clip_edge_pix, get_test_train_split_idx

    train_idx, _ = get_test_train_split_idx(len(idx), train_frac=train_frac)
    if subsample > 1:
        train_idx = train_idx[::subsample]
    sel = idx[train_idx]
    n = len(sel)

    order = np.argsort(data.spectra_indices[sel])
    Y = W = None
    n_masked = 0
    t0 = time.time()
    for start in range(0, n, block):
        rows = order[start : start + block]
        flux, u_flux = clip_edge_pix(*data.get_flux_batch(sel[rows]))
        if Y is None:
            Y = np.empty((n, flux.shape[1]), dtype)
            W = np.empty((n, flux.shape[1]), dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / (u_flux.astype(dtype) ** 2)
        # isfinite, not isnan: u_flux == 0 gives an infinite weight, which the
        # original nan_to_num route turned into 1.8e308 rather than masking.
        bad = ~(np.isfinite(flux) & np.isfinite(w))
        n_masked += int(bad.sum())
        Y[rows] = np.where(bad, 0, flux)
        W[rows] = np.where(bad, 0, w)
        print(
            f"  read {min(start + block, n)}/{n} rows ({time.time() - t0:.0f} s)",
            end="\r",
            flush=True,
        )
    print(
        f"\n  read {n} rows in {time.time() - t0:.0f} s; "
        f"masked {n_masked / (n * Y.shape[1]):.3%} of pixels",
        flush=True,
    )
    return Y, W, ids[train_idx]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument("--rank", type=int, default=10, help="K")
    p.add_argument("--q", type=float, default=2.0, help="robust scale Q")
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--conv-tol", type=float, default=1e-4)
    p.add_argument("--conv-check-cadence", type=int, default=1)
    p.add_argument("--subsample", type=int, default=1, help="keep every Nth training spectrum")
    p.add_argument("--precision", default=DEFAULT_PRECISION, choices=PRECISIONS)
    p.add_argument(
        "--engine",
        default="distributed",
        choices=("distributed", "reference"),
        help="'reference' is the stock library Robusta -- for reproducing the OOM",
    )
    p.add_argument(
        "--row-block", type=int, default=4096, help="rows per scan chunk (peak scratch memory)"
    )
    p.add_argument(
        "--n-devices", type=int, default=None, help="use only the first N visible devices"
    )
    p.add_argument(
        "--store-fp64",
        action="store_true",
        help="keep Y and W on device in float64 (2x memory; the inputs are float32)",
    )
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    dtype = configure_precision(args.precision)
    devices = jax.devices()
    if args.n_devices is not None:
        devices = devices[: args.n_devices]
    print(f"Devices: {[str(d) for d in devices]}")
    for d in devices:
        try:
            stats = d.memory_stats() or {}
            limit = stats.get("bytes_limit")
            if limit:
                print(f"  {d}: {limit / 2**30:.1f} GiB pool ({d.device_kind})")
        except Exception:  # CPU devices have no memory_stats
            pass
    print(f"Precision: {args.precision} (compute dtype {np.dtype(dtype).name})", flush=True)

    print(f"Building sample {args.sample!r}...", flush=True)
    data, idx, ids, tag = build_sample(args.sample)
    print(f"Sample {tag!r}: {len(idx)} unique spectra", flush=True)

    # Y and W are stored in float32 unless asked otherwise: the HDF5 fluxes and
    # uncertainties are float32, so the cast to float64 happens per chunk on
    # device and nothing is lost by not paying for it on the host too.
    store_dtype = np.float64 if args.store_fp64 else np.float32
    Y, W, ids_train = load_training_data(
        data, idx, ids, dtype=store_dtype, subsample=args.subsample
    )
    N, M = Y.shape
    print(f"Training matrix: {N} x {M} ({Y.nbytes / 2**30:.2f} GiB each for Y and W)", flush=True)

    t0 = time.time()
    if args.engine == "reference":
        from robusta_hmf import Robusta

        model = Robusta(
            rank=args.rank,
            robust_scale=args.q,
            conv_strategy="max_frac_G",
            conv_tol=args.conv_tol,
            init_strategy="svd",
            rotation="fast",
            target="G",
            whiten=True,
        )
        state, loss = model.fit(
            Y.astype(dtype, copy=False),
            W.astype(dtype, copy=False),
            max_iter=args.max_iter,
            conv_check_cadence=args.conv_check_cadence,
        )
    else:
        from distributed_robusta import DistributedRobusta

        model = DistributedRobusta(
            rank=args.rank,
            robust_scale=args.q,
            conv_strategy="max_frac_G",
            conv_tol=args.conv_tol,
            rotation="fast",
            target="G",
            whiten=True,
            row_block=args.row_block,
            devices=devices,
            store_dtype=store_dtype,
            compute_dtype=dtype,
        )
        state, loss = model.fit(
            Y,
            W,
            max_iter=args.max_iter,
            conv_check_cadence=args.conv_check_cadence,
        )
    print(f"Fit finished in {time.time() - t0:.0f} s ({len(loss)} iterations)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    # Same stem as train_full_ms.py so analyse_full_ms.py can pick these up.
    stem = f"R{args.rank}_Q{args.q:.2f}_bin_{tag}"
    if args.subsample > 1:
        stem += f"_sub{args.subsample}"
    path = args.out / f"converged_state_{stem}.npz"
    np.savez(
        path,
        A=np.asarray(state.A),
        G=np.asarray(state.G),
        it=np.asarray(state.it),
        loss=np.asarray(loss),
        source_id=ids_train,
    )
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
