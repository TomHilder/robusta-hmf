"""Sequential (K, Q) grid search on the full Gaia RVS sample, in float64.

The obstacle to fitting the whole sample was scratch memory, not the model.
``Robusta.fit`` evaluates each ALS step over the whole matrix at once, so a
step holds ~5 full ``(N, M)`` intermediates. At N = 496_955 training rows and
M = 2321 pixels that is 8.6 GiB each in float64, i.e. ~43 GiB of scratch on
top of 17 GiB of resident Y and W: hence every 5th spectrum fits on a 48 GiB
A6000 and every 4th does not.

:mod:`distributed_robusta` restructures the same ALS iteration so the row axis
is traversed in chunks and sharded over every visible GPU. Measured peak
device memory for the full sample is 9.0 GiB at K=10 and 9.2 GiB at K=30, so
the grid runs on one card. See that module for the details.

MODEL SELECTION follows the paper (Section 4.3, Eq. 20) rather than the
``std_z`` shortcut in ``analyse_full_ms.py``. Each model is fit on the
training split, then used to predict the held-out split with G held fixed
(a-step and w-step only). With

    z_ij = r_ij sqrt(w_data_ij w_robust_ij)

unit normal under a well-calibrated model, the score is the KL divergence
from the empirical distribution of z to N(0, 1), Gaussian-approximated:

    S(Q, K) = -log(sigma_z) + (sigma_z^2 + mu_z^2) / 2 - 1/2

which is zero only at mu_z = 0, sigma_z = 1. Lower is better. ``std_z``,
``chi2_red`` and ``rmse`` come free from the same sums and are recorded too,
so the ``analyse_full_ms.py`` ranking can be compared against the paper's.

DATA. One 50/50 train/test split of the whole sample -- the canonical seeded
split from ``get_test_train_split_idx``, the same one ``train_full_ms.py`` and
``analyse_full_ms.py`` use. ``--subsample N`` then keeps every Nth spectrum of
*both* halves, so the grid fits on 1/N of train and scores on 1/N of test,
still disjoint. That is only to make the grid affordable: the winning (K, Q)
is refit on the full training half at the end.

USAGE
    # grid on 1/10 of each half (fast), then refit the winner on all of train
    uv run python 20260731_oom_tests.py

    # the whole split at every grid point -- expensive, watch the ETA
    uv run python 20260731_oom_tests.py --subsample 1

    # a single model, no grid
    uv run python 20260731_oom_tests.py --ranks 10 --q-vals 2 --no-refit-best

    # reproduce the OOM with the stock library implementation
    uv run python 20260731_oom_tests.py --engine reference --ranks 10 --q-vals 2 --subsample 4

Finished models are skipped, so an interrupted grid resumes where it stopped.
Under Slurm, ask for the GPUs and let JAX see all of them -- do NOT set
CUDA_VISIBLE_DEVICES to a single device:

    sbatch -p gpu --gpus=4 -c 16 --mem=256G -t 12:00:00 --wrap \
      "cd $PWD && UV_NO_SYNC=1 uv run python -u 20260731_oom_tests.py"

If a GPU is shared with anything else, cap the pool with
``XLA_PYTHON_CLIENT_MEM_FRACTION=0.9``.
"""

import argparse
import gc
import time
from pathlib import Path

import gaia_config as cfg
import jax
import numpy as np

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# The full sample is far more heterogeneous than any single bin (the per-bin
# analysis uses K=10), so the grid extends to larger ranks. Same grid as
# train_full_ms.py.
RANKS = [2, 4, 8, 16, 32, 64]
Q_VALS = [2.0, 3.0, 5.0, 7.5]

MAX_ITER = 1000
TRAIN_FRAC = cfg.TRAIN_FRAC

RESULTS_DIR = Path("./gaia_rvs_results")

# Bin tag used in state filenames: converged_state_R{K}_Q{Q}_bin_{tag}.npz
SAMPLE_TAGS = {"ms": "full_ms", "all": "full_rvs"}

# How much of the sample the grid search uses. The canonical 50/50 train/test
# split (get_test_train_split_idx, seeded with cfg.RNG_SEED, shared with
# train_full_ms.py and analyse_full_ms.py) is made first; this then keeps every
# Nth spectrum of BOTH halves, so the grid trains on 1/N of the training half
# and scores on 1/N of the held-out half, and the two stay disjoint.
# The winner is refit on the full training set afterwards, so this only has to
# rank models correctly, not produce the final basis.
GRID_SUBSAMPLE = 10

# Optional extra cap on held-out spectra, off by default now that --subsample
# sets the test size too. The scoring statistics are means over n_test * 2321
# per-pixel residuals -- already >1e8 at 50k spectra, converged far beyond the
# differences between grid points -- so capping costs nothing if a run ever
# needs it. analyse_full_ms.py caps at 50_000 for the same reason.
CV_MAX_TEST = 0

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


def build_sample(sample="all"):
    """Return (data, idx, ids, tag) for the requested sample.

    "ms":  union of all main-sequence bins, deduplicated (bins overlap since
           widths exceed spacing), idx/ids kept aligned.
    "all": every spectrum in the RVS file -- all 999645 of them. The metadata
           filters (finite BP-RP/G, positive parallax) are off here: they are
           there so a star can be placed on the HR diagram for the binning,
           which this sample does not do, and they would drop 5735 spectra
           that the fit is perfectly able to model. bp_rp and abs_mag_G are
           NaN for those rows, and the HR plots mask them out.
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

        data = MatchedData(filter_nans=False, filter_neg_parallax=False)
        idx = np.arange(len(data.spectra_indices))
        return data, idx, data["source_id"], SAMPLE_TAGS["all"]
    raise ValueError(f"Unknown sample: {sample!r} (use 'ms' or 'all')")


def load_split(
    data,
    idx,
    ids,
    which="train",
    train_frac=TRAIN_FRAC,
    subsample=1,
    max_rows=0,
    dtype=np.float32,
    block=65536,
):
    """Y, W and source ids for one side of the train/test split.

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

    train_idx, test_idx = get_test_train_split_idx(len(idx), train_frac=train_frac)
    split_idx = train_idx if which == "train" else test_idx
    if subsample > 1:
        split_idx = split_idx[::subsample]
    if max_rows and len(split_idx) > max_rows:
        # Seeded, so every model is scored on the same held-out spectra --
        # matching the --cv-max-test subsample in analyse_full_ms.py.
        split_idx = np.random.default_rng(cfg.RNG_SEED).choice(
            split_idx, size=max_rows, replace=False
        )

    sel = idx[split_idx]
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
        print(f"  reading {which}: {min(start + block, n)}/{n} rows", end="\r", flush=True)
    print(
        f"  {which}: {n} x {Y.shape[1]} in {time.time() - t0:.0f} s, "
        f"{2 * Y.nbytes / 2**30:.2f} GiB for Y+W, "
        f"{n_masked / (n * Y.shape[1]):.3%} of pixels masked",
        flush=True,
    )
    return Y, W, ids[split_idx]


def state_path(results_dir, rank, q, tag, subsample):
    """Naming as train_full_ms.py, with a suffix when only a subsample was fit."""
    stem = f"converged_state_R{rank}_Q{q:.2f}_bin_{tag}"
    if subsample > 1:
        stem += f"_sub{subsample}"
    return results_dir / f"{stem}.npz"


def make_model(rank, q, args, dtype, devices):
    from distributed_robusta import DistributedRobusta

    return DistributedRobusta(
        rank=rank,
        robust_scale=q,
        conv_strategy="max_frac_G",
        conv_tol=args.conv_tol,
        rotation="fast",
        target="G",
        whiten=True,
        row_block=args.row_block,
        devices=devices,
        store_dtype=np.float64 if args.store_fp64 else np.float32,
        compute_dtype=dtype,
    )


def fit_one(rank, q, train, args, dtype, devices):
    """Fit one (K, Q) model on ``train``. Returns an RHMFState."""
    if args.engine == "reference":
        # The stock library path, kept for reproducing the OOM. It needs Y and
        # W in the compute dtype up front, which is half the problem.
        from robusta_hmf import Robusta

        Y, W = train
        model = Robusta(
            rank=rank,
            robust_scale=q,
            conv_strategy="max_frac_G",
            conv_tol=args.conv_tol,
            init_strategy="svd",
            rotation="fast",
            target="G",
            whiten=True,
        )
        state, _ = model.fit(
            Y.astype(dtype, copy=False),
            W.astype(dtype, copy=False),
            max_iter=args.max_iter,
            conv_check_cadence=args.conv_check_cadence,
        )
        return state

    state, _ = make_model(rank, q, args, dtype, devices).fit(
        train,
        max_iter=args.max_iter,
        conv_check_cadence=args.conv_check_cadence,
        verbose=args.verbose,
    )
    return state


def run_grid(ranks, q_vals, train, test, args, dtype, devices, tag, n_train=None):
    """Fit and score every (K, Q) in turn. Returns a dict of (n_K, n_Q) arrays."""
    from robusta_hmf.state import load_state_from_npz

    shape = (len(ranks), len(q_vals))
    out = {k: np.full(shape, np.nan) for k in ("kl", "mu_z", "std_z", "chi2_red", "rmse")}
    out["seconds"] = np.full(shape, np.nan)

    n_models = len(ranks) * len(q_vals)
    done = 0
    t_start = time.time()
    print(f"\nGrid: {len(ranks)} ranks x {len(q_vals)} Q values = {n_models} models", flush=True)
    print(f"{'K':>4} {'Q':>5} {'KL':>12} {'std_z':>9} {'mu_z':>10} {'chi2_red':>10} {'s':>7}")

    for i, rank in enumerate(ranks):
        for j, q in enumerate(q_vals):
            t0 = time.time()
            path = state_path(args.out, rank, q, tag, args.subsample)
            # A cached state is only the same model if it was fit on the same
            # rows: the "all" sample changed size when the HR-diagram metadata
            # filters came off, and scoring only touches G, so a stale state
            # would otherwise be reused in silence.
            stale = False
            if path.exists() and not args.overwrite:
                state = load_state_from_npz(path)
                stale = n_train is not None and state.A.shape[0] != n_train
                if stale:
                    print(
                        f"  {path.name}: {state.A.shape[0]} rows, but the training set has "
                        f"{n_train} -- refitting.",
                        flush=True,
                    )
            if path.exists() and not args.overwrite and not stale:
                cached = "  (cached)"
            else:
                state = fit_one(rank, q, train, args, dtype, devices)
                np.savez(
                    path, A=np.asarray(state.A), G=np.asarray(state.G), it=np.asarray(state.it)
                )
                cached = ""

            # Scored with the chunked scorer whichever engine did the fitting:
            # held-out scoring only needs G.
            scorer = make_model(rank, q, args, dtype, devices)
            inferred, _ = scorer.infer(
                test, state=state, max_iter=args.infer_max_iter, tol=args.infer_tol
            )
            scores = scorer.score(test, inferred)
            for k, v in scores.items():
                out[k][i, j] = v
            out["seconds"][i, j] = time.time() - t0
            done += 1

            print(
                f"{rank:>4} {q:>5g} {scores['kl']:>12.6f} {scores['std_z']:>9.4f} "
                f"{scores['mu_z']:>10.2e} {scores['chi2_red']:>10.4f} "
                f"{out['seconds'][i, j]:>7.0f}{cached}",
                flush=True,
            )
            if done == 1 and n_models > 1 and not cached:
                print(
                    f"     ~{out['seconds'][i, j] * (n_models - 1) / 60:.0f} min remaining "
                    "at this rate",
                    flush=True,
                )
            del scorer, inferred, state
            gc.collect()

    print(f"Grid finished in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return out


def report_best(grid, ranks, q_vals):
    """Best model by the paper's KL score; also by analyse_full_ms.py's std_z."""
    kl = grid["kl"]
    i, j = np.unravel_index(np.nanargmin(kl), kl.shape)
    print(f"\nBest by KL (paper Eq. 20): K={ranks[i]}, Q={q_vals[j]:g}  (KL={kl[i, j]:.6f})")

    # analyse_full_ms.py ranks on |std_z - 1| instead. Report both, so the
    # choice of criterion stays visible rather than implicit.
    dev = np.abs(grid["std_z"] - 1.0)
    a, b = np.unravel_index(np.nanargmin(dev), dev.shape)
    print(
        f"Best by std_z (analyse_full_ms.py): K={ranks[a]}, Q={q_vals[b]:g}  "
        f"(std_z={grid['std_z'][a, b]:.4f})"
    )
    if (a, b) != (i, j):
        print("  NOTE: the two criteria disagree; the paper advocates the KL score.")
    return ranks[i], q_vals[j]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    p.add_argument("--q-vals", type=float, nargs="+", default=Q_VALS)
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--conv-tol", type=float, default=1e-4)
    p.add_argument("--conv-check-cadence", type=int, default=5)
    p.add_argument("--infer-max-iter", type=int, default=1000)
    p.add_argument("--infer-tol", type=float, default=1e-4)
    p.add_argument(
        "--subsample",
        type=int,
        default=GRID_SUBSAMPLE,
        help="keep every Nth spectrum of both halves of the split for the grid "
        "(1 = the whole sample; default: %(default)s)",
    )
    p.add_argument(
        "--cv-max-test",
        type=int,
        default=CV_MAX_TEST,
        help="optional extra cap on held-out spectra used for scoring (0 = no cap)",
    )
    p.add_argument(
        "--refit-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after the grid, refit the winning (K, Q) on the full training set",
    )
    p.add_argument("--overwrite", action="store_true", help="refit models that already have state")
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
    p.add_argument("--verbose", action="store_true", help="print every convergence check")
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    dtype = configure_precision(args.precision)
    from distributed_robusta import build_mesh, shard_data

    devices = jax.devices()
    if args.n_devices is not None:
        devices = devices[: args.n_devices]
    print(f"Devices: {[str(d) for d in devices]}")
    for d in devices:
        try:
            limit = (d.memory_stats() or {}).get("bytes_limit")
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
    mesh = build_mesh(devices)

    # One 50/50 train/test split of the whole sample (seeded, and the same one
    # train_full_ms.py and analyse_full_ms.py use), then the same stride on
    # both halves: the grid fits on 1/N of train and scores on 1/N of test.
    print("Loading data...", flush=True)
    Y_tr, W_tr, _ = load_split(
        data, idx, ids, "train", subsample=args.subsample, dtype=store_dtype
    )
    Y_te, W_te, _ = load_split(
        data,
        idx,
        ids,
        "test",
        subsample=args.subsample,
        max_rows=args.cv_max_test,
        dtype=store_dtype,
    )
    if args.subsample > 1:
        tail = (
            "the winner is refit on the full training half afterwards."
            if args.refit_best
            else "--refit-best is off, so no full-sample fit will be made."
        )
        print(
            f"Grid: fitting on {len(Y_tr)} spectra, scoring on {len(Y_te)} held out "
            f"(1/{args.subsample} of each half); {tail}",
            flush=True,
        )

    # Sharded once and reused by every grid point; re-transferring the training
    # matrices per model would otherwise dominate.
    test = shard_data(Y_te, W_te, mesh, args.row_block, store_dtype)
    train = (
        (Y_tr, W_tr)
        if args.engine == "reference"
        else shard_data(Y_tr, W_tr, mesh, args.row_block, store_dtype)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    grid = run_grid(
        args.ranks, args.q_vals, train, test, args, dtype, devices, tag, n_train=len(Y_tr)
    )
    best_K, best_Q = report_best(grid, args.ranks, args.q_vals)

    suffix = f"_sub{args.subsample}" if args.subsample > 1 else ""
    scores_path = args.out / f"{tag}_grid_scores{suffix}.npz"
    np.savez(
        scores_path,
        ranks=np.asarray(args.ranks),
        q_vals=np.asarray(args.q_vals),
        best_K=best_K,
        best_Q=best_Q,
        n_train=len(Y_tr),
        n_test=len(Y_te),
        **grid,
    )
    print(f"Wrote {scores_path}")

    final_path = state_path(args.out, best_K, best_Q, tag, 1)
    if args.subsample == 1:
        print(f"Grid used the full training set; {final_path} is the final model.")
        return
    if not args.refit_best:
        return
    if final_path.exists() and not args.overwrite:
        print(f"\nFinal model already exists: {final_path}")
        return

    # Free the subsampled training data before reloading the full split.
    del train, Y_tr, W_tr
    gc.collect()

    print(f"\nRefitting the winner (K={best_K}, Q={best_Q:g}) on the full training set...")
    Y_tr, W_tr, ids_tr = load_split(data, idx, ids, "train", dtype=store_dtype)
    full = (
        (Y_tr, W_tr)
        if args.engine == "reference"
        else shard_data(Y_tr, W_tr, mesh, args.row_block, store_dtype)
    )
    t0 = time.time()
    state = fit_one(best_K, best_Q, full, args, dtype, devices)
    print(f"Refit in {(time.time() - t0) / 60:.1f} min", flush=True)

    scorer = make_model(best_K, best_Q, args, dtype, devices)
    inferred, _ = scorer.infer(test, state=state, max_iter=args.infer_max_iter, tol=args.infer_tol)
    final = scorer.score(test, inferred)
    print("Final model, held out: " + ", ".join(f"{k}={v:.6f}" for k, v in final.items()))

    np.savez(
        final_path,
        A=np.asarray(state.A),
        G=np.asarray(state.G),
        it=np.asarray(state.it),
        source_id=ids_tr,
        **{f"cv_{k}": v for k, v in final.items()},
    )
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()
