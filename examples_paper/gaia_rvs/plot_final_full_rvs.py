"""Figures for the final full-RVS model, from the saved per-spectrum weights.

Split out of ``fit_final_full_rvs.py`` so the figures can be made somewhere
other than where the fit ran. Everything here reads one file --
``<tag>_final_weights.npz``, ~60 MB for the full sample -- and needs nothing
but numpy and matplotlib: no JAX, no GPU, no ``robusta_hmf``, and none of the
19 GB of spectra. Copy that one npz to a machine with a working LaTeX
installation and run this.

The one exception is ``hr_by_component.pdf``, which needs the per-spectrum
amplitudes ``A`` and so reads the converged state npz as well (another ~130 MB,
and still plain numpy). It is picked up automatically when it sits next to the
weights file and skipped with a note when it does not, so the one-file usage
above is unaffected.

That constraint is why the weight histogram is written out here rather than
imported from ``analyse_full_ms.py``: importing it drags in ``analysis_funcs``
and ``collect``, which assert the HDF5 and the metadata CSV are present. The
figure it draws is the same one.

The npz carries every spectrum's score, so ``--threshold`` moves the outlier
cut and replots without refitting anything; ``<tag>_final_outliers.csv`` is
just the rows below the threshold that the fit script happened to use.

USAGE
    # defaults: gaia_rvs_results/full_rvs_final_weights.npz -> plots_full_rvs_final/
    uv run python plot_final_full_rvs.py

    # elsewhere, with the npz copied over
    python plot_final_full_rvs.py full_rvs_final_weights.npz --plots-dir figs

    # move the outlier cut
    python plot_final_full_rvs.py --threshold 0.3

If titles fail to render with a LaTeX error, the plot style wants a TeX
installation the machine does not have: ``module load texlive`` on Rusty.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# The house style, when it is installed. This module is meant to be runnable on
# a laptop with nothing but numpy and matplotlib, so a missing style is a note
# rather than a failure -- the figures come out in matplotlib defaults.
try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Spectra scoring below this are called outliers. 0.5 as in analyse_full_ms.py.
WEIGHT_THRESHOLD = 0.5

# A second, much tighter cut, drawn as its own figure. At 0.5 the HRD carries
# ~1500 points and reads as a population; this one keeps only the few dozen
# most extreme spectra, where individual objects can be picked out and looked
# up. Nothing downstream uses it -- the outlier catalogue is still the 0.5 cut.
EXTREME_THRESHOLD = 0.05

# HR diagram framing, matching plot_bins.py and analysis_funcs.py.
HR_XLIM = (-0.5, 3.5)
HR_YLIM = (15, -5)

# Colour limits for every weight-valued panel, as percentiles of the values
# being drawn. The scores are weights in [0, 1], but they concentrate in a band
# a few hundredths wide near 0.9 -- on the nominal [0, 1] scale every figure
# below is one flat colour. Widen these to compress the stretch.
CLIM_PERCENTILES = (1.0, 99.0)

DEFAULT_WEIGHTS = Path("./gaia_rvs_results/full_rvs_final_weights.npz")

# ============================================================================ #


def check_text_rendering(fallback=True):
    """Check that text renders, and drop LaTeX if it does not.

    ``mpl_drip.custom`` sets ``text.usetex``, so on a node whose texmf tree is
    missing the Computer Modern fonts every ``savefig`` raises -- and in
    ``fit_final_full_rvs.py`` the figures come last, after hours of fitting.
    Probing costs a second, and turning usetex off gets figures out of a
    machine with no working TeX at all. They come out in matplotlib's own
    fonts rather than the paper's, hence the warning: for anything going into
    the paper, fix TeX (``module load texlive`` on Rusty) and replot.

    Every title in this module is plain text, so nothing but the typeface
    changes. Returns True if LaTeX rendering is live.
    """
    fig = plt.figure()
    try:
        fig.text(0.5, 0.5, "probe")
        fig.canvas.draw()
        return True
    except Exception as exc:
        print(
            f"WARNING: matplotlib cannot render text ({type(exc).__name__}: {exc}).",
            flush=True,
        )
        if fallback and plt.rcParams.get("text.usetex", False):
            plt.rcParams["text.usetex"] = False
            print(
                "         Falling back to matplotlib's own text rendering: the figures\n"
                "         will not carry the paper's fonts. 'module load texlive' first,\n"
                "         or replot elsewhere, for anything going into the paper.",
                flush=True,
            )
            return check_text_rendering(fallback=False)
        return False
    finally:
        plt.close(fig)


def _frame_hr(ax):
    ax.set_xlim(*HR_XLIM)
    ax.set_ylim(*HR_YLIM)
    ax.set_xlabel("Color (BP - RP)")
    ax.set_ylabel("G-Band Absolute Magnitude")


def _robust_clim(values, lo=CLIM_PERCENTILES[0], hi=CLIM_PERCENTILES[1]):
    """Colour limits from the percentiles of the finite values, or None.

    Every score-valued panel here needs the same stretch for the same reason:
    the weights pile up in a narrow band well inside [0, 1], so any fixed or
    min-to-max scale paints the whole sample one colour.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def _hexbin_clim(hb, lo=CLIM_PERCENTILES[0], hi=CLIM_PERCENTILES[1]):
    """Stretch a hexbin's colour scale to the percentiles of the drawn cells.

    ``mincnt`` masks the sparse cells and numpy's percentiles ignore the mask,
    so the limits have to come off the compressed array -- taking them straight
    off ``get_array()`` would stretch the scale to whatever is under the mask
    rather than to the cells actually drawn.
    """
    cells = hb.get_array()
    cells = cells.compressed() if np.ma.isMaskedArray(cells) else np.asarray(cells)
    clim = _robust_clim(cells, lo, hi)
    if clim is not None and clim[0] < clim[1]:
        hb.set_clim(*clim)


def plot_hr_by_weight(score, bp_rp, abs_mag_G, threshold, K, Q, out, label):
    """Every spectrum on the HR diagram, coloured by its per-spectrum weight.

    Same colour convention as ``analysis_funcs.plot_all_spectra_hr_by_weight``
    (viridis), but drawn for ~1e6 points: rasterized, so the PDF stays small,
    and sorted by descending weight so the most anomalous spectra are painted
    last instead of being buried under the bulk of the sample.

    The colour scale is stretched to the percentiles of the score rather than
    fixed to [0, 1]. For the full RVS sample the scores span roughly 0.79 to
    0.91 between the 1st and 99th percentiles, so on a [0, 1] scale the whole
    diagram comes out one shade of green and the figure says nothing. Points
    outside the stretch saturate, which is what the arrowed colour bar means;
    the outliers get their own figure in ``hr_outliers.pdf``.
    """
    order = np.argsort(-score)
    clim = _robust_clim(score)
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
        vmin=None if clim is None else clim[0],
        vmax=None if clim is None else clim[1],
        rasterized=True,
    )
    plt.colorbar(
        scatter,
        ax=ax,
        extend="both",
        label="1st-percentile robust weight per spectrum",
    )
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


def plot_hr_hexbin(score, bp_rp, abs_mag_G, threshold, K, Q, out, label, gridsize=200, mincnt=5):
    """Per-cell summaries of the weight over the HR diagram, plus the density.

    At ~1e6 spectra a scatter is dominated by whichever points happen to be
    drawn last; binning shows where in the HR diagram the model fits badly,
    which is the part that carries information. Four panels, because the
    per-cell median alone hides the thing worth seeing:

    * **worst spectrum per cell** -- the minimum score among the spectra in the
      cell. The median is a statement about the bulk and sits in a band a few
      hundredths wide; the minimum spans the full range down to ~0.003 and is
      what picks out where the anomalies live.
    * **median per cell** -- the bulk quality of fit, stretched to its own
      percentiles.
    * **outlier fraction per cell** -- how much of each cell falls below the
      threshold, which is the quantity the outlier catalogue is built from.
    * **density** -- the denominator for all three.

    All the score panels read against a stretched colour bar; read the numbers,
    not the colour.
    """
    extent = (HR_XLIM[0], HR_XLIM[1], min(HR_YLIM), max(HR_YLIM))
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    axes = axes.ravel()
    common = dict(gridsize=gridsize, mincnt=mincnt, extent=extent)

    hb = axes[0].hexbin(
        bp_rp, abs_mag_G, C=score, reduce_C_function=np.min, cmap="viridis", **common
    )
    _hexbin_clim(hb)
    plt.colorbar(hb, ax=axes[0], extend="both", label="Minimum robust weight score in cell")
    axes[0].set_title(f"Worst spectrum per cell (at least {mincnt} spectra)")

    hb = axes[1].hexbin(
        bp_rp, abs_mag_G, C=score, reduce_C_function=np.median, cmap="viridis", **common
    )
    _hexbin_clim(hb)
    plt.colorbar(hb, ax=axes[1], extend="both", label="Median robust weight score")
    axes[1].set_title(f"Median score per cell (at least {mincnt} spectra)")

    # np.mean of a boolean-as-float C is the fraction below the threshold.
    hb = axes[2].hexbin(
        bp_rp,
        abs_mag_G,
        C=(score < threshold).astype(float),
        reduce_C_function=np.mean,
        cmap="inferno",
        **common,
    )
    # Stretched at the top end only: most cells hold no outliers at all, and a
    # vmin above zero would colour those as if they did.
    clim = _robust_clim(hb.get_array().compressed(), lo=0, hi=99.5)
    if clim is not None and clim[1] > 0:
        hb.set_clim(0, clim[1])
    plt.colorbar(hb, ax=axes[2], extend="max", label=f"Fraction with score below {threshold}")
    axes[2].set_title(f"Outlier fraction per cell (at least {mincnt} spectra)")

    hb = axes[3].hexbin(
        bp_rp, abs_mag_G, gridsize=gridsize, mincnt=1, extent=extent, cmap="magma", bins="log"
    )
    plt.colorbar(hb, ax=axes[3], label="Spectra per cell")
    axes[3].set_title("Sample density")

    for ax in axes:
        _frame_hr(ax)
    _suptitle(fig, f"{label}: K={K}, Q={Q:.2f}")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hr_outliers(score, bp_rp, abs_mag_G, threshold, K, Q, out, label, marker_size=None):
    """Outliers over a grey field of the whole sample, as in plot_outliers_on_hr.

    *marker_size* defaults to a size chosen from how many points survive the
    cut: the ~1500 outliers at the standard threshold need small dots to stay
    legible, while the few dozen at a tight cut would be all but invisible at
    the same size.
    """
    mask = score < threshold
    n = int(mask.sum())
    if marker_size is None:
        marker_size = 6 if n > 500 else (30 if n > 100 else 70)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.scatter(bp_rp, abs_mag_G, s=0.5, alpha=0.1, c="grey", zorder=0, marker=".", rasterized=True)
    if mask.any():
        scatter = ax.scatter(
            bp_rp[mask],
            abs_mag_G[mask],
            c=score[mask],
            cmap="viridis_r",
            s=marker_size,
            alpha=0.85,
            marker="o",
            edgecolors="k",
            linewidths=0.4 if n <= 500 else 0,
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


def plot_weight_hist(score, threshold, out, label):
    """Distribution of the per-spectrum outlier score.

    The same figure as ``analyse_full_ms.plot_weight_hist``, restated rather
    than imported: importing that module pulls in ``analysis_funcs`` and
    ``collect``, which require the spectra and metadata files to be present,
    and this module is meant to run without them.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(score, bins=100, color="C0", alpha=0.8)
    ax.axvline(threshold, color="grey", ls="--", label=f"Threshold ({threshold})")
    ax.set_yscale("log")
    ax.set_xlabel("1st-Percentile Robust Weight per Spectrum")
    ax.set_ylabel("Count")
    ax.legend()
    _suptitle(fig, f"{label}: Outlier Scores", y=0.96)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hr_by_component(A, bp_rp, abs_mag_G, K, Q, out, label, gridsize=150, mincnt=5):
    """One HRD panel per component, coloured by the median coefficient per cell.

    ``A`` is the (N, K) matrix of per-spectrum amplitudes: row *i* says how much
    of each of the K eigenspectra in ``G`` the model used to rebuild spectrum
    *i*. Binned onto the HR diagram and reduced by the median, each panel shows
    where in colour-magnitude space that component is being used -- which is
    how you read what a component means, since the eigenspectra themselves are
    just vectors over wavelength.

    Two things to keep in mind. The factorisation is only defined up to an
    invertible rotation of (A, G), so the individual components are a basis and
    not physical parameters; the sign of any one of them is arbitrary, and a
    panel that looks inverted from what you expect is not a bug. And component
    order here is the order in ``G``, not a variance ranking -- the fit's
    rotation targets G, so there is no guarantee component 0 dominates.

    Each panel is stretched to its own percentiles: the components differ in
    scale by orders of magnitude, so a shared colour scale would show the
    largest one and flat grey for the rest. Panels whose values straddle zero
    get a zero-centred diverging scale, so the sign is readable; the rest get
    the same sequential map as the weight figures.
    """
    extent = (HR_XLIM[0], HR_XLIM[1], min(HR_YLIM), max(HR_YLIM))
    ncol = int(np.ceil(np.sqrt(K)))
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow), dpi=130)
    axes = np.atleast_1d(axes).ravel()

    for k in range(K):
        ax = axes[k]
        hb = ax.hexbin(
            bp_rp,
            abs_mag_G,
            C=A[:, k],
            reduce_C_function=np.median,
            gridsize=gridsize,
            mincnt=mincnt,
            extent=extent,
        )
        cells = hb.get_array()
        cells = cells.compressed() if np.ma.isMaskedArray(cells) else np.asarray(cells)
        clim = _robust_clim(cells)
        if clim is not None and clim[0] < clim[1]:
            if clim[0] < 0 < clim[1]:
                # Symmetric about zero, so mid-colour means "this component is
                # unused here" and the two signs are distinguishable.
                lim = max(abs(clim[0]), abs(clim[1]))
                hb.set_cmap("RdBu_r")
                hb.set_clim(-lim, lim)
            else:
                hb.set_cmap("viridis")
                hb.set_clim(*clim)
        plt.colorbar(hb, ax=ax, extend="both")
        ax.set_title(f"Component {k}")
        _frame_hr(ax)

    for ax in axes[K:]:
        ax.set_visible(False)

    _suptitle(fig, f"{label}: median component amplitude per cell, K={K}, Q={Q:.2f}")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _suptitle(fig, text, y=1.02):
    """The folder's suptitle: LaTeX sans-serif bold when usetex is on.

    Without usetex the same string would go to mathtext, which does not know
    ``\\textsf``, so it is set as plain bold text instead.
    """
    if plt.rcParams.get("text.usetex", False):
        fig.suptitle(rf"$\textsf{{\textbf{{{text}}}}}$", fontsize="24", c="dimgrey", y=y)
    else:
        fig.suptitle(text, fontsize="24", c="dimgrey", y=y, fontweight="bold")


def default_state_file(weights_file):
    """The all-rows state npz that goes with a weights npz, by naming convention.

    ``fit_final_full_rvs.py`` writes both into the same directory from the same
    (K, Q), so the name is derivable; returns None when it is not there.
    """
    weights_file = Path(weights_file)
    d = np.load(weights_file)
    tag = weights_file.name.replace("_final_weights.npz", "")
    path = weights_file.parent / (
        f"converged_state_R{int(d['best_K'])}_Q{float(d['best_Q']):.2f}_bin_{tag}_allrows.npz"
    )
    return path if path.exists() else None


def make_plots(
    weights_file,
    plots_dir,
    threshold,
    label,
    state_file=None,
    extreme_threshold=EXTREME_THRESHOLD,
):
    """The sample-level figures, from the saved per-spectrum weights.

    *state_file* is the converged all-rows state npz. It is optional and only
    adds the per-component HRD panels: everything else comes from the weights
    file alone, which is what lets this module run anywhere.
    """
    d = np.load(weights_file)
    score, K, Q = d["score"], int(d["best_K"]), float(d["best_Q"])
    bp_rp, abs_mag_G = d["bp_rp"], d["abs_mag_G"]

    # Sources with no usable astrometry/photometry cannot be placed on the HRD;
    # they still have weights and stay in the npz/CSV.
    finite = np.isfinite(bp_rp) & np.isfinite(abs_mag_G)
    if not finite.all():
        print(f"  {np.sum(~finite)} spectra lack colour/magnitude and are omitted from the HRD")
    score, bp_rp, abs_mag_G = score[finite], bp_rp[finite], abs_mag_G[finite]

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_hr_by_weight(
        score, bp_rp, abs_mag_G, threshold, K, Q, plots_dir / "hr_by_weight.pdf", label
    )
    plot_hr_hexbin(
        score, bp_rp, abs_mag_G, threshold, K, Q, plots_dir / "hr_weight_hexbin.pdf", label
    )
    plot_hr_outliers(
        score, bp_rp, abs_mag_G, threshold, K, Q, plots_dir / "hr_outliers.pdf", label
    )
    # The same figure at a much tighter cut. Named for the value so that
    # changing --extreme-threshold writes a new file rather than silently
    # overwriting a figure drawn at a different cut.
    plot_hr_outliers(
        score,
        bp_rp,
        abs_mag_G,
        extreme_threshold,
        K,
        Q,
        plots_dir / f"hr_outliers_below_{extreme_threshold:g}.pdf",
        label,
    )
    plot_weight_hist(score, threshold, plots_dir / "weights_hist.pdf", label)

    if state_file is None:
        print("note: no state file, so the per-component HRD panels are skipped")
        return
    A = np.load(state_file)["A"]
    if A.shape[0] != len(finite):
        raise SystemExit(
            f"{state_file} has {A.shape[0]} rows but the weights file has {len(finite)} -- "
            "these are not from the same fit."
        )
    if A.shape[1] != K:
        raise SystemExit(f"{state_file} has rank {A.shape[1]}, but the weights say K={K}")
    plot_hr_by_component(
        A[finite], bp_rp, abs_mag_G, K, Q, plots_dir / "hr_by_component.pdf", label
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "weights",
        type=Path,
        nargs="?",
        default=DEFAULT_WEIGHTS,
        help="per-spectrum weights npz written by fit_final_full_rvs.py (default: %(default)s)",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="where to write the figures (default: ./plots_<sample>_final)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="outlier cut; default: the threshold recorded in the npz",
    )
    p.add_argument(
        "--extreme-threshold",
        type=float,
        default=EXTREME_THRESHOLD,
        help="second, tighter cut for its own HRD figure (default: %(default)s)",
    )
    p.add_argument("--label", default=None, help="figure title prefix")
    p.add_argument(
        "--state",
        type=Path,
        default=None,
        help="converged all-rows state npz, for the per-component HRD panels "
        "(default: the matching converged_state_*_allrows.npz next to the weights, "
        "if it is there)",
    )
    p.add_argument(
        "--no-state",
        action="store_true",
        help="skip the per-component HRD panels even if the state file is present",
    )
    p.add_argument(
        "--no-latex",
        action="store_true",
        help="draw with matplotlib's own text rendering instead of LaTeX "
        "(not the paper's fonts, but it works anywhere)",
    )
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} does not exist -- run fit_final_full_rvs.py first.")

    d = np.load(args.weights)
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    # "full_rvs_final_weights.npz" -> "full_rvs"
    tag = args.weights.name.replace("_final_weights.npz", "")
    label = args.label or ("Full Main Sequence" if tag == "full_ms" else "Full RVS Sample")
    plots_dir = args.plots_dir or Path(f"./plots_{tag}_final")

    n_out = int(np.sum(d["score"] < threshold))
    print(
        f"{args.weights}: {len(d['score'])} spectra, K={int(d['best_K'])}, Q={float(d['best_Q'])}"
    )
    print(f"Outliers at score below {threshold}: {n_out}")

    state_file = None if args.no_state else (args.state or default_state_file(args.weights))
    if args.state is not None and not args.state.exists():
        raise SystemExit(f"{args.state} does not exist.")
    if state_file is not None:
        print(f"Component panels from {state_file}")

    check_text_rendering()
    make_plots(args.weights, plots_dir, threshold, label, state_file, args.extreme_threshold)
    print(f"\nDone. Figures in {plots_dir}")


if __name__ == "__main__":
    main()
