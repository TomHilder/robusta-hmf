"""Final model: fit the chosen (K, Q) to EVERY Gaia RVS spectrum, then compute
per-spectrum robust weights and colour the HR diagram by them.

This is the step after ``20260731_oom_tests.py``. That script ranks models on a
subsample of the canonical 50/50 split and refits the winner on the training
half; this one drops the split entirely and fits the whole matched RVS
catalogue (~1e6 spectra), which is what the outlier catalogue and the HR
diagram figures need -- every spectrum has to have a weight, not just the half
that happened to land in the training set.

Because nothing is held out, the fit here is not a model-selection step: (K, Q)
comes in from the grid search, either explicitly via ``--K/--Q`` or read out of
the grid's score file (the default). The scores printed after convergence are
in-sample and are there as a sanity check, not as a selection criterion.

The engine is :mod:`distributed_robusta` with the same settings the grid used,
imported from the grid script rather than restated, so "the same (K, Q)" really
means the same model. Resident device memory is ~18.5 GiB for Y+W as float32
across all visible GPUs (twice the training half), plus ~1 GiB of scratch per
row-block, so this fits on one 48 GiB card and is comfortable on more.

OUTLIER SCORE. Per spectrum, the 1st percentile of its per-pixel robust
weights -- the same score as ``analyse_full_ms.py`` and ``analyse_bins.py``, so
the numbers are directly comparable with the per-bin analysis in the paper.
Median and mean weight per spectrum are recorded alongside it. Pixels masked by
the data (zero data weight) get a robust weight of 1, since the residual there
is zero by construction; that is the existing convention in
``analysis_funcs.compute_outlier_scores``, kept here deliberately.

OUTPUTS (in ./gaia_rvs_results and ./plots_<tag>_final):
    converged_state_R<K>_Q<Q>_bin_<tag>_allrows.npz  -- A, G for all spectra
    <tag>_final_weights.npz    -- source_id, score, median/mean weight, colour,
                                  absolute magnitude, for EVERY spectrum
    <tag>_final_outliers.csv   -- source ids and scores below the threshold
    plots_<tag>_final/hr_by_weight.pdf      -- HRD, every spectrum by weight
    plots_<tag>_final/hr_weight_hexbin.pdf  -- HRD, median weight per cell
    plots_<tag>_final/hr_outliers.pdf       -- HRD, outliers over a grey field
    plots_<tag>_final/weights_hist.pdf      -- score distribution

USAGE
    # take (K, Q) from the grid's score file (the paper's KL criterion)
    uv run python fit_final_full_rvs.py

    # or state them
    uv run python fit_final_full_rvs.py --K 32 --Q 3

    # reuse a G already fit on the training half; only infer A for all rows
    uv run python fit_final_full_rvs.py --from-state \
        gaia_rvs_results/converged_state_R32_Q3.00_bin_full_rvs.npz

    # replot from a finished run without touching the GPU
    uv run python fit_final_full_rvs.py --plots-only

Under Slurm, ask for the GPUs and let JAX see all of them:

    sbatch -p gpu --gpus=4 -c 16 --mem=256G -t 12:00:00 --wrap \
      "cd $PWD && UV_NO_SYNC=1 uv run python -u fit_final_full_rvs.py"
"""

import argparse
import importlib
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyse_full_ms import plot_weight_hist
from analysis_funcs import get_test_train_split_idx
from collect import compute_abs_mag
from train_full_ms import (
    DEFAULT_PRECISION,
    PRECISIONS,
    RESULTS_DIR,
    build_sample,
    configure_precision,
)

from robusta_hmf.state import RHMFState, load_state_from_npz

plt.style.use("mpl_drip.custom")

# The grid search lives in a date-stamped module whose name is not a valid
# Python identifier, so a plain import statement cannot reach it. Its block
# reader and model constructor are the ones every fitted model in
# gaia_rvs_results was produced with; importing them is what keeps this fit
# identical to the grid's rather than a lookalike.
_grid = importlib.import_module("20260731_oom_tests")
load_split = _grid.load_split
make_model = _grid.make_model
GRID_SUBSAMPLE = _grid.GRID_SUBSAMPLE
SAMPLE_TAGS = _grid.SAMPLE_TAGS

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Spectra scoring below this are called outliers. 0.5 as in analyse_full_ms.py;
# every score is saved, so the cut can be moved without refitting anything.
WEIGHT_THRESHOLD = 0.5

# HR diagram framing, matching plot_bins.py and analysis_funcs.py.
HR_XLIM = (-0.5, 3.5)
HR_YLIM = (15, -5)

MAX_ITER = 1000

# ============================================================================ #


def resolve_KQ(args, tag):
    """(K, Q) from the CLI if given, else the best model in the grid's scores."""
    if args.K is not None and args.Q is not None:
        print(f"Using K={args.K}, Q={args.Q:g} from the command line")
        return args.K, args.Q
    if args.K is not None or args.Q is not None:
        raise SystemExit("Pass both --K and --Q, or neither (to read the grid scores)")

    path = args.grid_scores
    if path is None:
        # The grid subsamples by default, which suffixes its score file.
        for candidate in (
            args.out / f"{tag}_grid_scores_sub{GRID_SUBSAMPLE}.npz",
            args.out / f"{tag}_grid_scores.npz",
        ):
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        raise SystemExit(
            f"No grid score file found in {args.out} -- finish 20260731_oom_tests.py, "
            "or pass --K/--Q explicitly (or --grid-scores PATH)."
        )

    d = np.load(path)
    ranks, q_vals = d["ranks"], d["q_vals"]
    if args.select == "kl":
        # The paper's criterion (Section 4.3, Eq. 20). Lower is better.
        i, j = np.unravel_index(np.nanargmin(d["kl"]), d["kl"].shape)
        why = f"KL={d['kl'][i, j]:.6f}"
    else:
        # analyse_full_ms.py ranks on |std_z - 1| instead.
        dev = np.abs(d["std_z"] - 1.0)
        i, j = np.unravel_index(np.nanargmin(dev), dev.shape)
        why = f"std_z={d['std_z'][i, j]:.4f}"
    K, Q = int(ranks[i]), float(q_vals[j])
    print(f"Best model by {args.select} in {path.name}: K={K}, Q={Q:g}  ({why})")
    return K, Q


def load_all_rows(data, idx, ids, dtype):
    """Y, W, source ids and catalogue positions for EVERY spectrum in the sample.

    ``load_split`` reads one side of the canonical train/test split block by
    block, straight into preallocated arrays -- the only route that does not
    peak at several times the size of the data on the host. With
    ``train_frac=1.0`` every spectrum lands in the "train" side, so the whole
    sample comes back through the same reader. The row order is the seeded
    shuffle from ``get_test_train_split_idx``, recomputed here so rows can be
    mapped back to the catalogue for colour and magnitude.
    """
    row_idx, _ = get_test_train_split_idx(len(idx), train_frac=1.0)
    Y, W, row_ids = load_split(data, idx, ids, "train", train_frac=1.0, dtype=dtype)
    assert np.array_equal(row_ids, ids[row_idx]), "row order does not match the recomputed split"
    return Y, W, row_ids, row_idx


def per_spectrum_weights(model, Y, W, state, row_block):
    """Per-spectrum summaries of the per-pixel robust weights.

    The weight matrix is the same shape as ``Y`` (~9 GiB here, ~18 in float64),
    so it is consumed one row-block at a time and reduced on the fly; nothing
    (N, M) is ever resident.
    """
    n = Y.shape[0]
    out = {k: np.empty(n) for k in ("score", "median", "mean")}
    start, t0 = 0, time.time()
    for block in model.robust_weights_chunked(Y, W, state=state, row_block=row_block):
        stop = start + len(block)
        out["score"][start:stop] = np.percentile(block, 1, axis=1)
        out["median"][start:stop] = np.median(block, axis=1)
        out["mean"][start:stop] = block.mean(axis=1)
        start = stop
        print(f"  weights: {stop}/{n} spectra", end="\r", flush=True)
    print(f"  weights: {n} spectra in {time.time() - t0:.0f} s" + " " * 20, flush=True)
    return out


def check_text_rendering():
    """Warn early if the LaTeX toolchain the plot style needs is unavailable.

    ``mpl_drip.custom`` sets ``text.usetex``, so on a node whose texmf tree is
    missing the Computer Modern fonts every ``savefig`` raises -- and the
    figures come last, after hours of fitting. Probing costs a second here.
    Nothing is lost either way: the weights are written before the figures, so
    ``--plots-only`` finishes the job once ``module load texlive`` is in effect.
    """
    fig = plt.figure()
    try:
        fig.text(0.5, 0.5, "probe")
        fig.canvas.draw()
        return True
    except Exception as exc:
        print(
            f"WARNING: matplotlib cannot render text ({type(exc).__name__}: {exc}).\n"
            "         Try 'module load texlive' before running. The weights are saved\n"
            "         before plotting, so '--plots-only' can make the figures later.",
            flush=True,
        )
        return False
    finally:
        plt.close(fig)


def _frame_hr(ax):
    ax.set_xlim(*HR_XLIM)
    ax.set_ylim(*HR_YLIM)
    ax.set_xlabel("Color (BP - RP)")
    ax.set_ylabel("G-Band Absolute Magnitude")


def plot_hr_by_weight(score, bp_rp, abs_mag_G, threshold, K, Q, out, label):
    """Every spectrum on the HR diagram, coloured by its per-spectrum weight.

    Same colour convention as ``analysis_funcs.plot_all_spectra_hr_by_weight``
    (viridis over [0, 1]), but drawn for ~1e6 points: rasterized, so the PDF
    stays small, and sorted by descending weight so the most anomalous spectra
    are painted last instead of being buried under the bulk of the sample.
    """
    order = np.argsort(-score)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter = ax.scatter(
        bp_rp[order],
        abs_mag_G[order],
        c=score[order],
        cmap="viridis",
        s=2,
        alpha=0.6,
        marker=".",
        linewidths=0,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    plt.colorbar(scatter, ax=ax, label="1st-percentile robust weight per spectrum")
    _frame_hr(ax)
    n_out = int(np.sum(score < threshold))
    # Titles avoid "<" and "|": the style file renders text through LaTeX,
    # where both come out as something else entirely in text mode.
    ax.set_title(
        f"{label}, K={K}, Q={Q:.2f}, N={len(score)}, outliers={n_out} (score below {threshold})"
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hr_hexbin(score, bp_rp, abs_mag_G, K, Q, out, label, gridsize=200, mincnt=5):
    """Median weight per HR cell, next to the sample density.

    At ~1e6 spectra the scatter above is dominated by whichever points happen
    to be drawn last; binning shows where in the HR diagram the model fits
    badly on average, which is the part that carries information.
    """
    extent = (HR_XLIM[0], HR_XLIM[1], min(HR_YLIM), max(HR_YLIM))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    hb = axes[0].hexbin(
        bp_rp,
        abs_mag_G,
        C=score,
        reduce_C_function=np.median,
        gridsize=gridsize,
        mincnt=mincnt,
        extent=extent,
        cmap="viridis",
    )
    # Cell medians occupy a narrow band near the top of [0, 1] -- on the fixed
    # scale the scatter uses, this panel would be one flat colour. Stretched to
    # the bulk of the cell medians instead, so the structure is visible; read
    # the colour bar, not the colour.
    cells = hb.get_array()
    if cells.size:
        hb.set_clim(np.nanpercentile(cells, 1), np.nanpercentile(cells, 99))
    plt.colorbar(hb, ax=axes[0], label="Median robust weight score")
    axes[0].set_title(f"Median score per cell (at least {mincnt} spectra)")

    hb = axes[1].hexbin(
        bp_rp,
        abs_mag_G,
        gridsize=gridsize,
        mincnt=1,
        extent=extent,
        cmap="magma",
        bins="log",
    )
    plt.colorbar(hb, ax=axes[1], label="Spectra per cell")
    axes[1].set_title("Sample density")

    for ax in axes:
        _frame_hr(ax)
    fig.suptitle(
        rf"$\textsf{{\textbf{{{label}: K={K}, Q={Q:.2f}}}}}$", fontsize="24", c="dimgrey", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hr_outliers(score, bp_rp, abs_mag_G, threshold, K, Q, out, label):
    """Outliers over a grey field of the whole sample, as in plot_outliers_on_hr."""
    mask = score < threshold
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.scatter(bp_rp, abs_mag_G, s=0.5, alpha=0.1, c="grey", zorder=0, marker=".", rasterized=True)
    if mask.any():
        scatter = ax.scatter(
            bp_rp[mask],
            abs_mag_G[mask],
            c=score[mask],
            cmap="viridis_r",
            s=6,
            alpha=0.8,
            marker=".",
            linewidths=0,
            zorder=5,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label="Outlier score (lower = more anomalous)")
    _frame_hr(ax)
    ax.set_title(
        f"{label}, K={K}, Q={Q:.2f}, {int(mask.sum())} outliers "
        f"of {len(score)} (score below {threshold})"
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def make_plots(weights_file, plots_dir, threshold, label):
    """All four figures, from the saved per-spectrum weights."""
    d = np.load(weights_file)
    score, K, Q = d["score"], int(d["best_K"]), float(d["best_Q"])
    bp_rp, abs_mag_G = d["bp_rp"], d["abs_mag_G"]

    # Sources with no usable astrometry/photometry cannot be placed on the HRD;
    # they still have weights and stay in the npz/CSV.
    finite = np.isfinite(bp_rp) & np.isfinite(abs_mag_G)
    if not finite.all():
        print(f"  {np.sum(~finite)} spectra lack colour/magnitude and are omitted from the HRD")
    score, bp_rp, abs_mag_G = score[finite], bp_rp[finite], abs_mag_G[finite]

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_hr_by_weight(
        score, bp_rp, abs_mag_G, threshold, K, Q, plots_dir / "hr_by_weight.pdf", label
    )
    plot_hr_hexbin(score, bp_rp, abs_mag_G, K, Q, plots_dir / "hr_weight_hexbin.pdf", label)
    plot_hr_outliers(
        score, bp_rp, abs_mag_G, threshold, K, Q, plots_dir / "hr_outliers.pdf", label
    )
    plot_weight_hist(score, threshold, plots_dir / "weights_hist.pdf", label)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument("--K", type=int, default=None, help="rank; default: read the grid scores")
    p.add_argument("--Q", type=float, default=None, help="robust scale; default: as --K")
    p.add_argument(
        "--select",
        default="kl",
        choices=("kl", "std_z"),
        help="criterion when reading (K, Q) from the grid scores",
    )
    p.add_argument(
        "--grid-scores", type=Path, default=None, help="grid score npz to read (K, Q) from"
    )
    p.add_argument(
        "--from-state",
        type=Path,
        default=None,
        help="reuse G from this state file and only infer A for all rows",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="replot from an existing weights file; no fitting, no GPU",
    )
    p.add_argument("--weight-threshold", type=float, default=WEIGHT_THRESHOLD)
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--conv-tol", type=float, default=1e-4)
    p.add_argument("--conv-check-cadence", type=int, default=5)
    p.add_argument("--infer-max-iter", type=int, default=1000)
    p.add_argument("--infer-tol", type=float, default=1e-4)
    p.add_argument("--precision", default=DEFAULT_PRECISION, choices=PRECISIONS)
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
    p.add_argument("--overwrite", action="store_true", help="refit even if the state file exists")
    p.add_argument("--verbose", action="store_true", help="print every convergence check")
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.add_argument("--plots-dir", type=Path, default=None)
    args = p.parse_args()

    tag = SAMPLE_TAGS[args.sample]
    label = "Full Main Sequence" if args.sample == "ms" else "Full RVS Sample"
    plots_dir = args.plots_dir or Path(f"./plots_{tag}_final")
    weights_file = args.out / f"{tag}_final_weights.npz"

    if args.plots_only:
        if not weights_file.exists():
            raise SystemExit(f"{weights_file} does not exist -- run without --plots-only first.")
        check_text_rendering()
        make_plots(weights_file, plots_dir, args.weight_threshold, label)
        return

    # Before the fit, not after it: a broken plot backend should be a warning
    # printed in the first second, not a traceback several hours in.
    check_text_rendering()

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

    args.out.mkdir(parents=True, exist_ok=True)
    K, Q = resolve_KQ(args, tag)
    # Distinct from the grid's converged_state_R*_bin_<tag>.npz, which is fit on
    # the training half only; this one covers every row.
    state_file = args.out / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"

    # Y and W stay float32 unless asked otherwise: the HDF5 fluxes and
    # uncertainties are float32, so the cast to the compute dtype happens per
    # chunk on device and nothing is lost by not paying for it on the host.
    store_dtype = np.float64 if args.store_fp64 else np.float32
    print("Loading every spectrum...", flush=True)
    Y, W, row_ids, row_idx = load_all_rows(data, idx, ids, store_dtype)

    model = make_model(K, Q, args, dtype, devices)
    mesh = build_mesh(devices)
    full = shard_data(Y, W, mesh, args.row_block, store_dtype)

    if args.from_state is not None:
        print(f"Reusing G from {args.from_state}; inferring A for all rows...", flush=True)
        t0 = time.time()
        prior = load_state_from_npz(args.from_state)
        if prior.G.shape[1] != K:
            raise SystemExit(f"{args.from_state} has rank {prior.G.shape[1]}, not K={K}")
        inferred, n_it = model.infer(
            full, state=prior, max_iter=args.infer_max_iter, tol=args.infer_tol
        )
        # infer() returns A over the padded row grid; trim to the real rows so
        # it lines up with Y for the weight pass.
        state = RHMFState(A=np.asarray(inferred.A)[: len(Y)], G=prior.G, it=0)
        print(f"Inferred in {(time.time() - t0) / 60:.1f} min ({n_it} iterations)", flush=True)
    elif state_file.exists() and not args.overwrite:
        print(f"Loading existing full-sample state {state_file}", flush=True)
        state = load_state_from_npz(state_file)
        if state.A.shape[0] != len(Y):
            raise SystemExit(
                f"{state_file} has {state.A.shape[0]} rows, but the sample has {len(Y)} -- "
                "pass --overwrite to refit."
            )
    else:
        print(f"Fitting K={K}, Q={Q:g} on all {len(Y)} spectra...", flush=True)
        t0 = time.time()
        state, _ = model.fit(
            full,
            max_iter=args.max_iter,
            conv_check_cadence=args.conv_check_cadence,
            verbose=True,
        )
        print(
            f"Converged after {int(state.it)} iterations in {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
        np.savez(
            state_file,
            A=np.asarray(state.A),
            G=np.asarray(state.G),
            it=np.asarray(state.it),
            source_id=row_ids,
        )
        print(f"Wrote {state_file}")

    # In-sample, since nothing was held out: a calibration check on the fit
    # (std_z near 1), not a model-selection number.
    scores = model.score(full, state)
    print("In-sample: " + ", ".join(f"{k}={v:.6f}" for k, v in scores.items()), flush=True)

    print("Computing per-spectrum robust weights...", flush=True)
    weights = per_spectrum_weights(model, Y, W, state, args.row_block)

    # Colour and absolute magnitude for the HR diagram, in the same row order.
    bp_rp = data["bp_rp"][idx[row_idx]]
    abs_mag_G = compute_abs_mag(data["phot_g_mean_mag"], data["parallax"])[idx[row_idx]]

    score = weights["score"]
    outliers = np.flatnonzero(score < args.weight_threshold)
    print(
        f"Found {len(outliers)} outliers ({100 * len(outliers) / len(score):.2f}% of {len(score)})"
    )

    np.savez(
        weights_file,
        source_id=row_ids,
        score=score,
        median_weight=weights["median"],
        mean_weight=weights["mean"],
        bp_rp=bp_rp,
        abs_mag_G=abs_mag_G,
        row_idx=row_idx,
        best_K=K,
        best_Q=Q,
        threshold=args.weight_threshold,
        **{f"insample_{k}": v for k, v in scores.items()},
    )
    print(f"Wrote {weights_file} ({len(score)} spectra)")

    outliers_file = args.out / f"{tag}_final_outliers.csv"
    pd.DataFrame(
        {
            "idx": row_idx[outliers],
            "source_id": row_ids[outliers],
            "score": score[outliers],
            "median_weight": weights["median"][outliers],
            "bp_rp": bp_rp[outliers],
            "abs_mag_G": abs_mag_G[outliers],
            "best_K": K,
            "best_Q": Q,
        }
    ).sort_values("score").to_csv(outliers_file, index=False)
    print(f"Wrote {outliers_file}")

    # Everything above is on disk by now, so a plotting failure costs the
    # figures and nothing else -- do not let it take the run down with it.
    try:
        make_plots(weights_file, plots_dir, args.weight_threshold, label)
    except Exception as exc:
        print(f"Plotting failed ({type(exc).__name__}: {exc})")
        print("Weights are saved; replot with 'uv run python fit_final_full_rvs.py --plots-only'")

    print("\nDone. Summary:")
    print(f"  Sample:      {tag} ({len(score)} spectra, all rows fit)")
    print(f"  Model:       K={K}, Q={Q:.2f}")
    print(f"  In-sample:   std_z={scores['std_z']:.4f}, chi2_red={scores['chi2_red']:.4f}")
    print(f"  Outliers:    {len(outliers)} (score < {args.weight_threshold})")
    print(f"  Plots:       {plots_dir}")


if __name__ == "__main__":
    main()
