"""One PNG per outlier: spectrum, best-fit reconstruction, residual, weights.

The companion to ``plot_final_full_rvs.py``. That module draws the sample-level
figures from the weights npz alone; this one goes back to the spectra and the
converged state to show, for every spectrum the final model called an outlier,
what the model actually got wrong.

Unlike ``plot_final_full_rvs.py`` this needs the real inputs -- the RVS HDF5,
the metadata CSV, the converged state, and JAX for the IRLS weights -- so it
runs where the fit ran. It is cheap all the same: only the outlier rows are
read out of the HDF5, and the weights for ~1500 x 2321 pixels are a fraction of
a second on CPU. There is no GPU requirement.

WHAT IS PLOTTED, per outlier, three stacked panels sharing a wavelength axis:

    1. observed flux (black) over the rank-K reconstruction (red), with the
       pixels the model downweighted marked;
    2. the residual, over the +/- 1 sigma band implied by the catalogue flux
       errors -- a residual inside the band is a pixel the model fits to the
       noise, one outside it is not;
    3. the per-pixel robust weight, with the spectrum's outlier score (the 1st
       percentile of exactly this curve) drawn as a horizontal line.

Pixels masked in the data (zero inverse-variance: non-finite flux or error) are
shaded grey and left out of the flux trace. Their robust weight is 1 by
construction -- the residual there is zero -- so they cannot be outlier pixels
and are never marked as such.

The title carries the identifiers to look the star up with: Gaia DR3 source id,
outlier score, RA, Dec, BP - RP, and absolute G magnitude.

Filenames lead with the score, zero-padded, so an alphabetical listing of the
output directory is ranked worst-first:

    score_0.0016_srcid_457487413730043904.png

USAGE
    # every spectrum below the threshold recorded in the weights npz
    uv run python plot_final_outlier_spectra.py

    # a quick look at the twenty worst
    uv run python plot_final_outlier_spectra.py --limit 20

    # a different cut (the npz holds every score, so nothing is refit)
    uv run python plot_final_outlier_spectra.py --threshold 0.3

Spectral line markers are drawn when ``rvs_plot_utils`` can find its line list
CSVs; they are decoration, and a missing line list is a note rather than a
failure.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis_funcs import clip_edge_pix
from collect import compute_abs_mag
from matplotlib.ticker import MultipleLocator
from plot_final_full_rvs import check_text_rendering
from train_full_ms import DEFAULT_PRECISION, RESULTS_DIR, build_sample, configure_precision

import gaia_config as cfg

try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# A pixel is called an outlier pixel when its robust weight falls below this.
# The per-spectrum score is a percentile of the same weights, so this is the
# per-pixel analogue of the sample-level cut, not an independent knob.
PIXEL_WEIGHT_THRESHOLD = 0.5

# Downweighted pixels are marked in orange rather than red: the reconstruction
# is already red, and the two reading as the same thing was the point of
# confusion in the first draft of this figure.
OUTLIER_COLOUR = "tab:orange"

# Shade contiguous runs of downweighted pixels at least this long. Isolated
# single downweighted pixels are common and are left to the per-pixel markers.
MIN_SHADED_RUN = 4

# PNG, as asked: ~1500 files, and each is a 2321-pixel spectrum drawn three
# times over. Vector output for that is neither small nor fast to open.
DPI = 130
FIGSIZE = (14, 9)

DEFAULT_WEIGHTS = RESULTS_DIR / "full_rvs_final_weights.npz"

# ============================================================================ #


def load_radec(source_ids):
    """RA and Dec for *source_ids*, straight from the metadata CSV.

    ``collect.read_meta`` selects four columns and drops rows with nulls in
    any of them, so ``MatchedData`` carries no sky positions. Adding RA/Dec to
    that select would put them inside the ``drop_nulls`` and could quietly
    change which spectra make up the sample every other script builds, so they
    are read here instead and joined on source id. Returns two arrays in the
    order of *source_ids*, NaN where the id is not in the CSV.
    """
    import polars as pl

    from collect import META

    want = pl.DataFrame({"source_id": np.asarray(source_ids, np.int64)})
    got = want.join(
        pl.scan_csv(META).select(["source_id", "ra", "dec"]).collect(),
        on="source_id",
        how="left",
    )
    ra = got["ra"].to_numpy().astype(float)
    dec = got["dec"].to_numpy().astype(float)
    n_missing = int(np.sum(~np.isfinite(ra)))
    if n_missing:
        print(f"  note: {n_missing} source ids have no RA/Dec in {META.name}")
    return ra, dec


def contiguous_runs(mask):
    """Start/stop index pairs for each run of True in a 1-D boolean mask.

    Outlier pixels come in clumps -- a bad line core, a cosmic ray, a whole
    unmodelled band -- so they are drawn as a handful of shaded spans rather
    than as one ``axvspan`` per pixel, which for a badly fit spectrum would be
    hundreds of artists per panel and a slow, heavy PNG.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _shade(ax, λ_grid, mask, min_run=1, **kwargs):
    """Shade the wavelength ranges where *mask* is True.

    Runs shorter than *min_run* pixels are skipped: the worst spectra have
    several hundred downweighted pixels scattered in ones and twos, and shading
    every one of them turns the panel pink. The short ones are still marked
    individually by the per-pixel dots.
    """
    for start, stop in contiguous_runs(mask):
        if stop - start < min_run:
            continue
        ax.axvspan(λ_grid[start], λ_grid[stop - 1], **kwargs)


def _add_lines(axes, lines, λ_grid):
    """Spectral line markers on every panel, labelled on the top one only."""
    if lines is None:
        return
    from rvs_plot_utils import add_line_markers

    wl_range = (float(λ_grid.min()), float(λ_grid.max()))
    kwargs = dict(show_strong=True, show_abundance=False, show_cn=False, show_dib=False)
    try:
        for i, ax in enumerate(axes):
            add_line_markers(
                ax=ax, lines=lines, show_labels=(i == 0), label_fontsize=10,
                wl_range=wl_range, **kwargs,
            )
    except Exception as exc:  # decoration only; never lose the figure over it
        print(f"Warning: could not add line markers: {exc}")


def plot_outlier(λ_grid, flux, ivar, recon, weights, meta, out, lines=None):
    """Write one outlier's three-panel figure to *out*.

    Parameters
    ----------
    λ_grid : (M,) array
        Wavelength grid, edge pixels already clipped.
    flux, ivar : (M,) arrays
        Observed flux and its inverse variance, both zero where masked.
    recon : (M,) array
        The rank-K reconstruction, ``A_i @ G.T``.
    weights : (M,) array
        Per-pixel IRLS robust weights in [0, 1].
    meta : dict
        source_id, score, ra, dec, bp_rp, abs_mag_G for the title.
    out : Path
    lines : RVSLineLists or None
    """
    masked = ivar <= 0
    outlier_pix = (weights < PIXEL_WEIGHT_THRESHOLD) & ~masked
    residual = flux - recon
    # Masked pixels carry flux 0 and residual 0 by construction, which would
    # draw as a spike to zero rather than as the gap it is.
    show_flux = np.where(masked, np.nan, flux)
    show_resid = np.where(masked, np.nan, residual)
    with np.errstate(divide="ignore"):
        sigma = np.where(masked, np.nan, 1.0 / np.sqrt(np.where(masked, 1.0, ivar)))

    fig, axes = plt.subplots(
        3, 1, figsize=FIGSIZE, dpi=DPI, sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.4]}
    )
    legend_kw = dict(fontsize=9, framealpha=0.92, facecolor="white", edgecolor="none")

    axes[0].plot(λ_grid, show_flux, c="k", lw=1.4, label="Data", zorder=3)
    axes[0].plot(λ_grid, recon, c="tab:red", lw=1.4, ls=(0, (5, 1)), label="Model", zorder=4)
    axes[0].scatter(
        λ_grid[outlier_pix], show_flux[outlier_pix],
        s=14, c=OUTLIER_COLOUR, marker="o", zorder=5, linewidths=0,
        label=f"Downweighted pixels ({int(outlier_pix.sum())} of {len(λ_grid)})",
    )
    axes[0].set_ylabel("Flux")
    axes[0].legend(loc="lower right", ncol=3, **legend_kw)

    axes[1].fill_between(
        λ_grid, -sigma, sigma, color="tab:blue", alpha=0.35, lw=0, zorder=1,
        label="Data 1 sigma",
    )
    axes[1].axhline(0, c="tab:blue", lw=0.8, zorder=2)
    axes[1].plot(λ_grid, show_resid, c="k", lw=1.2, zorder=3)
    axes[1].scatter(
        λ_grid[outlier_pix], show_resid[outlier_pix],
        s=14, c=OUTLIER_COLOUR, marker="o", zorder=5, linewidths=0,
    )
    axes[1].set_ylabel("Residual\nFlux")
    axes[1].legend(loc="lower right", **legend_kw)

    axes[2].plot(λ_grid, weights, c="k", lw=1.2, zorder=3)
    axes[2].axhline(
        PIXEL_WEIGHT_THRESHOLD, c=OUTLIER_COLOUR, ls=":", lw=1.2,
        label=f"Pixel cut ({PIXEL_WEIGHT_THRESHOLD})", zorder=2,
    )
    axes[2].axhline(
        meta["score"], c="tab:purple", ls="--", lw=1.2,
        label=f"Score, 1st pct ({meta['score']:.3f})", zorder=2,
    )
    # Headroom above 1 for the legend: the weight curve runs the full [0, 1],
    # so there is nowhere inside the data for it to sit.
    axes[2].set_ylim(-0.05, 1.45)
    axes[2].set_yticks([0.0, 0.5, 1.0])
    axes[2].set_ylabel("Robust\nWeight")
    axes[2].set_xlabel("Wavelength [nm]")
    axes[2].legend(loc="upper right", ncol=2, **legend_kw)

    for ax in axes:
        _shade(ax, λ_grid, outlier_pix, min_run=MIN_SHADED_RUN,
               color=OUTLIER_COLOUR, alpha=0.13, lw=0, zorder=0)
        _shade(ax, λ_grid, masked, color="grey", alpha=0.25, lw=0, zorder=0)
        ax.set_xlim(λ_grid[0], λ_grid[-1])
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.tick_params(which="major", length=6)

    _add_lines(axes, lines, λ_grid)

    # Titles avoid "<" and "|": the style file renders text through LaTeX,
    # where both come out as something else entirely in text mode.
    axes[0].set_title(
        f"Gaia DR3 {meta['source_id']}   score = {meta['score']:.4f}\n"
        f"RA = {meta['ra']:.6f}   Dec = {meta['dec']:.6f}   "
        f"BP - RP = {meta['bp_rp']:.3f}   abs G = {meta['abs_mag_G']:.2f}",
        fontsize=12,
    )
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def load_outlier_inputs(weights_file, state_file, threshold, limit=None, sample="all"):
    """Everything the figures need, for the outlier rows only.

    Returns ``(λ_grid, Y, ivar, recon, robust_weights, meta)`` where the arrays
    are (n_outliers, M) and *meta* is a list of per-row title dicts, ordered
    worst score first.
    """
    d = np.load(weights_file)
    score, row_idx = d["score"], d["row_idx"]
    K, Q = int(d["best_K"]), float(d["best_Q"])

    # Worst first, so --limit takes the most anomalous rather than an arbitrary
    # slice, and so the output directory reads in rank order.
    rows = np.flatnonzero(score < threshold)
    rows = rows[np.argsort(score[rows])]
    if limit:
        rows = rows[:limit]
    if not len(rows):
        raise SystemExit(f"No spectra score below {threshold}.")
    print(f"{len(rows)} outliers below {threshold} (K={K}, Q={Q:g})", flush=True)

    print("Building the sample...", flush=True)
    data, idx, ids, _ = build_sample(sample)
    # row_idx indexes *idx*; idx indexes the catalogue. Both hops are needed:
    # they collapse only because the "all" sample happens to take every row.
    cat = idx[row_idx[rows]]
    assert np.array_equal(ids[row_idx[rows]], d["source_id"][rows]), (
        "sample row order does not match the weights file -- was the npz written "
        "from a different --sample?"
    )

    print(f"Reading {len(cat)} spectra...", flush=True)
    flux, u_flux = clip_edge_pix(*data.get_flux_batch(cat))
    # Identical masking to load_split in 20260731_oom_tests.py, so the weights
    # here are computed on exactly the inputs the fit saw.
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = 1.0 / (u_flux.astype(np.float64) ** 2)
    bad = ~(np.isfinite(flux) & np.isfinite(ivar))
    Y = np.where(bad, 0.0, flux).astype(np.float64)
    W = np.where(bad, 0.0, ivar)

    λ_grid = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]
    assert Y.shape[1] == len(λ_grid), f"{Y.shape[1]} pixels but {len(λ_grid)} wavelengths"

    print(f"Loading {state_file}...", flush=True)
    st = np.load(state_file)
    A, G = st["A"], st["G"]
    if A.shape[0] != len(score):
        raise SystemExit(
            f"{state_file} has {A.shape[0]} rows but the weights file has {len(score)} -- "
            "these are not from the same fit."
        )
    if G.shape[1] != K:
        raise SystemExit(f"{state_file} has rank {G.shape[1]}, but the weights say K={K}")
    A = A[rows]
    recon = A @ G.T

    # The same Student-t IRLS weights the fit used. Built through the model
    # constructor rather than by restating the likelihood, so the robustness
    # parameters that make_model leaves at their defaults (robust_nu) stay
    # whatever DistributedRobusta says they are.
    print("Computing per-pixel robust weights...", flush=True)
    configure_precision(DEFAULT_PRECISION)
    from distributed_robusta import DistributedRobusta

    likelihood = DistributedRobusta(rank=K, robust_scale=Q).likelihood
    robust = np.asarray(likelihood.weights_irls(Y, W, A, G))

    ra, dec = load_radec(ids[row_idx[rows]])
    bp_rp = data["bp_rp"][cat]
    abs_mag_G = compute_abs_mag(data["phot_g_mean_mag"], data["parallax"])[cat]
    meta = [
        {
            "source_id": int(ids[row_idx[r]]),
            "score": float(score[r]),
            "ra": float(ra[i]),
            "dec": float(dec[i]),
            "bp_rp": float(bp_rp[i]),
            "abs_mag_G": float(abs_mag_G[i]),
        }
        for i, r in enumerate(rows)
    ]
    return λ_grid, Y, W, recon, robust, meta


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "weights",
        type=Path,
        nargs="?",
        default=DEFAULT_WEIGHTS,
        help="per-spectrum weights npz from fit_final_full_rvs.py (default: %(default)s)",
    )
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument(
        "--state",
        type=Path,
        default=None,
        help="converged all-rows state npz; default: derived from the weights file's (K, Q)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="outlier cut; default: the threshold recorded in the weights npz",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the PNGs (default: ./plots_<tag>_final/outlier_spectra)",
    )
    p.add_argument("--limit", type=int, default=None, help="only the N worst outliers")
    p.add_argument(
        "--no-latex",
        action="store_true",
        help="draw with matplotlib's own text rendering instead of LaTeX",
    )
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False
    if not args.weights.exists():
        raise SystemExit(f"{args.weights} does not exist -- run fit_final_full_rvs.py first.")

    d = np.load(args.weights)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    state_file = args.state or (
        args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    if not state_file.exists():
        raise SystemExit(f"{state_file} does not exist -- pass --state explicitly.")
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/outlier_spectra")

    check_text_rendering()
    λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
        args.weights, state_file, threshold, args.limit, args.sample
    )

    try:
        from rvs_plot_utils import load_linelists

        lines = load_linelists()
    except Exception as exc:
        print(f"note: no spectral line markers ({type(exc).__name__}: {exc})")
        lines = None

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(meta)} figures to {out_dir}...", flush=True)
    t0 = time.time()
    for i, m in enumerate(meta):
        # Score first in the name, so the directory listing is ranked.
        out = out_dir / f"score_{m['score']:.4f}_srcid_{m['source_id']}.png"
        plot_outlier(λ_grid, Y[i], W[i], recon[i], robust[i], m, out, lines)
        if (i + 1) % 25 == 0 or i + 1 == len(meta):
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(meta)} ({rate:.1f} fig/s)", end="\r", flush=True)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Figures in {out_dir}")


if __name__ == "__main__":
    main()
