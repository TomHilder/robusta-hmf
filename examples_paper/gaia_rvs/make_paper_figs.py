"""
Regenerate the Gaia/RVS paper figures individually, straight into paper/figs.

Each figure has its own function and can be regenerated in isolation, e.g.:

    uv run python make_paper_figs.py hr_bins
    uv run python make_paper_figs.py gaia_spec_2a
    uv run python make_paper_figs.py all

Run from this directory (examples_paper/gaia_rvs) so the relative data paths
(plots_analysis/, gaia_rvs_results/, HDF5 via build_bins_from_config) resolve.

This is additive: it reuses the existing plotting functions
(plot_bins, plot_stacked_hist, _make_residual_figure) rather than duplicating them.
Output goes to the in-repo paper figures directory, NOT the stale external path.
"""

import argparse
import json
from pathlib import Path

import gaia_config as cfg
import matplotlib.pyplot as plt
import numpy as np
from analysis_funcs import (
    LINE_SET_VARIANTS,
    _make_residual_figure,
    build_bins_from_config,
    load_all_spectra_for_bin,
    load_cached_inferred_state,
    load_outlier_data,
)
from rvs_plot_utils import load_linelists

from robusta_hmf import Robusta

plt.style.use("mpl_drip.custom")

# Real paper figures directory (repo_root/paper/figs).
PAPER_FIGS = Path(__file__).resolve().parents[2] / "paper" / "figs"

# Match analyse_bins.py conventions.
BEST_MODEL_METRIC = "std_z"
PLOTS_DIR = Path("plots_analysis") / BEST_MODEL_METRIC
RESULTS_DIR = Path("gaia_rvs_results")

# Exact filenames included by paper/main.tex (hardcoded so the \includegraphics
# always resolves regardless of float-formatting drift in recomputed weights).
SPEC_FIGS = {
    "gaia_spec_1": dict(  # fig:gaia_spec_1 — 88 Her (Be star), bin 0 outlier
        i_bin=0,
        source_id=1363284299777747584,
        kind="outlier",
        filename="bin_00_K10_Q5.00_idx_00567_srcid_1363284299777747584_weight_0.009.pdf",
        suptitle_kwargs=dict(
            t=r"$\textsf{\textbf{Gaia Example: Be-Hosting Binary}}$",
            fontsize="24",
            c="dimgrey",
            y=0.955,
        ),
    ),
    "gaia_spec_normal": dict(  # fig:gaia_spec_normal — typical spectrum, bin 1
        i_bin=1,
        source_id=5309096898078973568,
        kind="normal",
        filename="bin_01_K10_Q5.00_idx_14536_srcid_5309096898078973568_weight_0.985.pdf",
        suptitle_kwargs=dict(
            t=r"$\textsf{\textbf{Gaia Example: Typical Star (Bin 1)}}$",
            fontsize="24",
            c="dimgrey",
            y=0.955,
        ),
    ),
    "gaia_spec_2a": dict(  # fig:gaia_spec_2 (top) — M-dwarf, bin 13 outlier
        i_bin=13,
        source_id=3136952686035250688,
        kind="outlier",
        filename="bin_13_K10_Q5.00_idx_00344_srcid_3136952686035250688_weight_0.173.pdf",
    ),
    "gaia_spec_2b": dict(  # fig:gaia_spec_2 (bottom) — M-dwarf, bin 13 outlier
        i_bin=13,
        source_id=3195919254111314816,
        kind="outlier",
        filename="bin_13_K10_Q5.00_idx_00350_srcid_3195919254111314816_weight_0.453.pdf",
    ),
}


def _bin_summary(i_bin):
    """Load best_K, best_Q for a bin from its saved summary.json."""
    with open(PLOTS_DIR / f"bin_{i_bin:02d}" / "summary.json") as f:
        summary = json.load(f)
    return summary["best_K"], summary["best_Q"]


def _save_spectrum_fig(
    λ_grid,
    flux,
    reconstruction,
    robust_weights,
    source_id,
    i_bin,
    idx,
    per_object_weight,
    best_K,
    best_Q,
    filename,
    suptitle_kwargs=None,
):
    """Build the 3-panel residual figure (strong-lines variant) and save to paper/figs."""
    residual = flux - reconstruction
    try:
        lines = load_linelists()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load line lists: {e}")
        lines = None
    # Strong-lines variant (the one the paper uses); copy the kwargs because
    # _make_residual_figure pops label_fontsize from the dict.
    _, strong_kwargs = LINE_SET_VARIANTS[0]
    fig = _make_residual_figure(
        λ_grid,
        flux,
        reconstruction,
        residual,
        robust_weights,
        source_id,
        i_bin,
        idx,
        per_object_weight,
        best_K,
        best_Q,
        lines,
        dict(strong_kwargs),
        suptitle_kwargs=suptitle_kwargs,
    )
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_FIGS / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _outlier_spectrum(spec):
    """Spectrum figure for an outlier, from the saved outlier_data.npz (no HDF5)."""
    i_bin, source_id = spec["i_bin"], spec["source_id"]
    odata = load_outlier_data(PLOTS_DIR, i_bin)
    if odata is None:
        raise SystemExit(f"No outlier_data.npz for bin {i_bin} in {PLOTS_DIR}")
    sids = np.asarray(odata["source_ids"]).astype(np.int64)
    matches = np.where(sids == source_id)[0]
    if len(matches) == 0:
        raise SystemExit(f"Source {source_id} not found in bin {i_bin} outlier data")
    j = int(matches[0])
    best_K, best_Q = _bin_summary(i_bin)
    _save_spectrum_fig(
        λ_grid=odata["lambda_grid"],
        flux=odata["flux"][j],
        reconstruction=odata["reconstructions"][j],
        robust_weights=odata["robust_weights"][j],
        source_id=int(sids[j]),
        i_bin=i_bin,
        idx=int(odata["indices"][j]),
        per_object_weight=float(odata["scores"][j]),
        best_K=best_K,
        best_Q=best_Q,
        filename=spec["filename"],
        suptitle_kwargs=spec.get("suptitle_kwargs"),
    )


def _normal_spectrum(spec):
    """Spectrum figure for a high-weight (non-outlier) star; needs HDF5 + cached state."""
    i_bin, source_id = spec["i_bin"], spec["source_id"]
    best_K, best_Q = _bin_summary(i_bin)
    state = load_cached_inferred_state(i_bin, best_K, best_Q, RESULTS_DIR)
    if state is None:
        raise SystemExit(f"No cached inferred state for bin {i_bin} (K={best_K}, Q={best_Q})")
    data, bins, _, _ = build_bins_from_config()
    bin_data = bins[i_bin]
    all_Y, all_W, _, _, source_ids = load_all_spectra_for_bin(data, bin_data, cfg.TRAIN_FRAC)
    λ_grid = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]
    data.close()

    sids = np.asarray(source_ids).astype(np.int64)
    matches = np.where(sids == source_id)[0]
    if len(matches) == 0:
        raise SystemExit(f"Source {source_id} not found in bin {i_bin}")
    idx = int(matches[0])

    rhmf = Robusta(rank=best_K, robust_scale=best_Q)
    weights = np.asarray(rhmf.robust_weights(all_Y, all_W, state=state))
    recon = np.asarray(rhmf.synthesize(state=state))
    per_object_weight = float(np.median(weights[idx]))
    _save_spectrum_fig(
        λ_grid=λ_grid,
        flux=all_Y[idx],
        reconstruction=recon[idx],
        robust_weights=weights[idx],
        source_id=source_id,
        i_bin=i_bin,
        idx=idx,
        per_object_weight=per_object_weight,
        best_K=best_K,
        best_Q=best_Q,
        filename=spec["filename"],
        suptitle_kwargs=spec.get("suptitle_kwargs"),
    )


def fig_gaia_spec_1():
    _outlier_spectrum(SPEC_FIGS["gaia_spec_1"])


def fig_gaia_spec_normal():
    _normal_spectrum(SPEC_FIGS["gaia_spec_normal"])


def fig_gaia_spec_2a():
    _outlier_spectrum(SPEC_FIGS["gaia_spec_2a"])


def fig_gaia_spec_2b():
    _outlier_spectrum(SPEC_FIGS["gaia_spec_2b"])


def fig_gaia_spec_2():
    """Fig 9: both bin-13 M-dwarf outliers stacked in one matplotlib figure.

    Two 3-panel residual plots share a single figure (and x-axis), so the panel
    proportions match exactly. Reuses _make_residual_figure (drawing into shared
    axes) so it tracks any formatting change to the individual spectrum figures.
    """
    i_bin = 13
    specs = [SPEC_FIGS["gaia_spec_2a"], SPEC_FIGS["gaia_spec_2b"]]
    odata = load_outlier_data(PLOTS_DIR, i_bin)
    if odata is None:
        raise SystemExit(f"No outlier_data.npz for bin {i_bin} in {PLOTS_DIR}")
    sids = np.asarray(odata["source_ids"]).astype(np.int64)
    best_K, best_Q = _bin_summary(i_bin)
    try:
        lines = load_linelists()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load line lists: {e}")
        lines = None

    # 7 rows = two [3,2,1] objects with a thin invisible spacer between them.
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(12, 16),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1, 0.01, 3, 2, 1]},
    )
    axes[3].set_visible(False)
    groups = [[axes[0], axes[1], axes[2]], [axes[4], axes[5], axes[6]]]

    _, strong_kwargs = LINE_SET_VARIANTS[0]
    for g, (spec, panel_axes) in enumerate(zip(specs, groups)):
        matches = np.where(sids == spec["source_id"])[0]
        if len(matches) == 0:
            raise SystemExit(f"Source {spec['source_id']} not in bin {i_bin} outlier data")
        j = int(matches[0])
        flux = odata["flux"][j]
        reconstruction = odata["reconstructions"][j]
        _make_residual_figure(
            odata["lambda_grid"],
            flux,
            reconstruction,
            flux - reconstruction,
            odata["robust_weights"][j],
            int(sids[j]),
            i_bin,
            int(odata["indices"][j]),
            float(odata["scores"][j]),
            best_K,
            best_Q,
            lines,
            dict(strong_kwargs),
            axes=panel_axes,
            show_xlabel=(g == 1),  # x-label only under the bottom object
            show_legend=(g == 0),  # a single Data/Model legend, on the top object
            label_lines=(g == 0),  # line labels only on the very top panel
        )

    fig.align_ylabels()
    fig.suptitle(
        r"$\textsf{\textbf{Gaia Example: M-dwarfs with Ca II Emission}}$",
        fontsize="24",
        c="dimgrey",
        y=0.935,
    )
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_FIGS / "gaia_spec_2.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_hr_bins():
    from plot_bins import plot_bins

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    plot_bins(save_path=PAPER_FIGS / "hr_bins.pdf")


def fig_stacked_hist():
    from plot_hist_stack import plot_stacked_hist

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    plot_stacked_hist(save_dir=PAPER_FIGS)


# --------------------------------------------------------------------------- #
# Full-RVS experiment (Section: all Gaia RVS sources)
# --------------------------------------------------------------------------- #

# The grid and the final fit, written by 20260731_oom_tests.py and
# fit_final_full_rvs.py respectively.
FULL_RVS_GRID = RESULTS_DIR / "full_rvs_grid_scores_sub10.npz"
FULL_RVS_WEIGHTS = RESULTS_DIR / "full_rvs_final_weights.npz"

# The model adopted for the full-sample fit. This is NOT the grid's raw KL
# argmin -- see fig_full_rvs_cv and the paper text: the KL score alone is
# minimised by a rank-2 model whose reduced chi-squared is 12.7.
FULL_RVS_ADOPTED = (16, 7.5)

# HR diagram framing, matching plot_bins.py and plot_final_full_rvs.py.
HR_XLIM = (-0.5, 3.5)
HR_YLIM = (15, -5)


def _heatmap_axes(ax, q_vals, ranks, show_y=True):
    """Shared (Q, K) heatmap framing, as in the toy example's fig_cv."""
    ax.set_xticks(np.arange(len(q_vals)), labels=[f"{q:g}" for q in q_vals])
    ax.set_yticks(np.arange(len(ranks)), labels=[str(r) for r in ranks])
    ax.set_xlabel("Robust Scale Q")
    if show_y:
        ax.set_ylabel("Rank K")
    else:
        ax.set_yticklabels([])


def _mark_cell(ax, i, j, colour, label):
    """Outline one grid cell and label it, for the two selected models."""
    ax.add_patch(
        plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=colour, lw=2.5, zorder=5, label=label
        )
    )


def fig_full_rvs_cv():
    """Figure: full_rvs_cv.pdf

    The cross-validation score over the (Q, K) grid for the full RVS sample,
    beside the reduced chi-squared of the same models.

    Two panels rather than one because the KL score on its own is misleading
    here: its global minimum sits at K=2, Q=2, a model whose reduced
    chi-squared is 12.7. The score only asks whether the standardised
    residuals look like unit normals, and a rank-deficient model with
    aggressive downweighting satisfies that by discarding most of the data.
    The right-hand panel is what breaks the degeneracy.
    """
    d = np.load(FULL_RVS_GRID)
    kl, chi2 = d["kl"], d["chi2_red"]
    ranks, q_vals = d["ranks"], d["q_vals"]

    i_kl, j_kl = np.unravel_index(np.nanargmin(kl), kl.shape)
    i_ad = int(np.flatnonzero(ranks == FULL_RVS_ADOPTED[0])[0])
    j_ad = int(np.flatnonzero(q_vals == FULL_RVS_ADOPTED[1])[0])

    # Generous wspace: each colour bar carries a two-line label, and at the
    # default spacing the left one is overprinted by the right-hand panel.
    fig = plt.figure(figsize=(14, 5), dpi=100)
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 0.045, 1, 0.045], left=0.06, right=0.90, wspace=0.75
    )
    ax_kl = fig.add_subplot(gs[0, 0])
    cax_kl = fig.add_subplot(gs[0, 1])
    ax_chi = fig.add_subplot(gs[0, 2])
    cax_chi = fig.add_subplot(gs[0, 3])

    text_bbox = dict(boxstyle="square", facecolor="white", alpha=0.7, edgecolor="none")
    text_loc = (0.06, 0.88)

    im_kl = ax_kl.imshow(np.log10(kl), origin="lower", cmap="viridis", aspect="auto")
    _heatmap_axes(ax_kl, q_vals, ranks)
    ax_kl.text(
        *text_loc, "Cross-Validation", transform=ax_kl.transAxes,
        ha="left", va="bottom", bbox=text_bbox,
    )
    fig.colorbar(
        im_kl, cax=cax_kl,
        label=r"$\log_{10}$ KL$(p_z \| \mathcal{N}(0,1))$" + "\n(Lower is Better)",
    )

    # log10 as well: chi2_red spans 0.94 to 12.7, and on a linear scale the
    # rank-2 row flattens everything else to one colour.
    im_chi = ax_chi.imshow(np.log10(chi2), origin="lower", cmap="magma_r", aspect="auto")
    _heatmap_axes(ax_chi, q_vals, ranks, show_y=False)
    ax_chi.text(
        *text_loc, "Goodness of Fit", transform=ax_chi.transAxes,
        ha="left", va="bottom", bbox=text_bbox,
    )
    fig.colorbar(
        im_chi, cax=cax_chi, label=r"$\log_{10} \chi^2_{\rm red}$" + "\n(Zero is Ideal)"
    )

    for ax in (ax_kl, ax_chi):
        _mark_cell(ax, i_kl, j_kl, "tab:red", "KL minimum")
        # Cyan, not white: the adopted cell is dark in the left panel and pale
        # in the right one, and a white outline vanishes in the legend too.
        _mark_cell(ax, i_ad, j_ad, "tab:cyan", "Adopted")
    # Below the axes rather than inside them: both marked cells sit at corners,
    # so any in-axes legend lands on one of the two models it is labelling.
    handles, labels = ax_kl.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=11,
        frameon=False, bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        r"$\textsf{\textbf{Gaia RVS: Hyperparameters (Full Sample)}}$",
        fontsize="24", c="dimgrey", y=1.02,
    )
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGS / "full_rvs_cv.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _load_full_rvs_weights():
    """Per-spectrum scores and HR positions, dropping sources with no astrometry."""
    d = np.load(FULL_RVS_WEIGHTS)
    score, bp_rp, abs_mag_G = d["score"], d["bp_rp"], d["abs_mag_G"]
    finite = np.isfinite(bp_rp) & np.isfinite(abs_mag_G)
    return (
        score[finite], bp_rp[finite], abs_mag_G[finite],
        int(d["best_K"]), float(d["best_Q"]), float(d["threshold"]), len(score),
    )


def fig_full_rvs_outliers():
    """Figure: full_rvs_outliers.pdf

    The outlier population of the full sample: the distribution of the
    object-level weight, and where the flagged spectra sit on the HR diagram.
    """
    score, bp_rp, abs_mag_G, K, Q, threshold, n_total = _load_full_rvs_weights()
    mask = score < threshold

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=100)

    axes[0].hist(score, bins=100, color="C0", alpha=0.85)
    axes[0].axvline(threshold, color="grey", ls="--", label=f"Threshold ({threshold:g})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Object-Level Weight $w_i^{\rm object}$")
    axes[0].set_ylabel("Number of Spectra")
    axes[0].legend(loc="upper left")
    axes[0].text(
        0.04, 0.72, f"{int(mask.sum())} outliers\nof {n_total}",
        transform=axes[0].transAxes, ha="left", va="top",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    axes[1].scatter(
        bp_rp, abs_mag_G, s=0.5, alpha=0.1, c="grey", zorder=0, marker=".", rasterized=True
    )
    sc = axes[1].scatter(
        bp_rp[mask], abs_mag_G[mask], c=score[mask], cmap="viridis_r",
        s=7, alpha=0.85, marker="o", linewidths=0, zorder=5, rasterized=True,
    )
    fig.colorbar(sc, ax=axes[1], label=r"$w_i^{\rm object}$ (Lower is More Anomalous)")
    axes[1].set_xlim(*HR_XLIM)
    axes[1].set_ylim(*HR_YLIM)
    axes[1].set_xlabel("Color (BP $-$ RP)")
    axes[1].set_ylabel("G-Band Absolute Magnitude")

    fig.suptitle(
        r"$\textsf{\textbf{Gaia RVS: Outliers in the Full Sample}}$",
        fontsize="24", c="dimgrey", y=1.02,
    )
    plt.tight_layout()
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGS / "full_rvs_outliers.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_full_rvs_hr_weights():
    """Figure: full_rvs_hr_weights.pdf

    Where on the HR diagram the model fits badly, binned rather than scattered.
    The per-cell median says how well the bulk of a region is reconstructed;
    the per-cell outlier fraction says how much of it gets flagged. At ~1e6
    spectra a scatter plot answers neither question, since it shows only
    whichever points happen to be drawn last.
    """
    score, bp_rp, abs_mag_G, K, Q, threshold, _ = _load_full_rvs_weights()
    extent = (HR_XLIM[0], HR_XLIM[1], min(HR_YLIM), max(HR_YLIM))
    common = dict(gridsize=200, mincnt=5, extent=extent)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=100)

    hb = axes[0].hexbin(
        bp_rp, abs_mag_G, C=score, reduce_C_function=np.median, cmap="viridis", **common
    )
    # The cell medians occupy a band a few hundredths wide near 0.89; on the
    # nominal [0, 1] range of a weight this panel is one flat colour.
    cells = hb.get_array()
    cells = cells.compressed() if np.ma.isMaskedArray(cells) else np.asarray(cells)
    cells = cells[np.isfinite(cells)]
    if cells.size:
        hb.set_clim(np.percentile(cells, 1), np.percentile(cells, 99))
    fig.colorbar(hb, ax=axes[0], extend="both", label=r"Median $w_i^{\rm object}$ per Cell")

    hb2 = axes[1].hexbin(
        bp_rp, abs_mag_G, C=(score < threshold).astype(float),
        reduce_C_function=np.mean, cmap="inferno", **common,
    )
    cells2 = hb2.get_array()
    cells2 = cells2.compressed() if np.ma.isMaskedArray(cells2) else np.asarray(cells2)
    cells2 = cells2[np.isfinite(cells2)]
    if cells2.size:
        hb2.set_clim(0, np.percentile(cells2, 99.5))
    fig.colorbar(
        hb2, ax=axes[1], extend="max",
        label=rf"Fraction with $w_i^{{\rm object}} < {threshold:g}$",
    )

    for ax in axes:
        ax.set_xlim(*HR_XLIM)
        ax.set_ylim(*HR_YLIM)
        ax.set_xlabel("Color (BP $-$ RP)")
    axes[0].set_ylabel("G-Band Absolute Magnitude")
    axes[1].set_yticklabels([])

    fig.suptitle(
        r"$\textsf{\textbf{Gaia RVS: Fit Quality Across the HR Diagram}}$",
        fontsize="24", c="dimgrey", y=1.02,
    )
    plt.tight_layout()
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGS / "full_rvs_hr_weights.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# The exemplar drawn for the outlier-taxonomy paragraph: the lowest-scoring
# member of the chromospheric-emission cluster of red giants. Chosen because
# the binned experiment of Section 5.2 is restricted to the main sequence and
# so cannot reach this population at all.
FULL_RVS_EXAMPLE = 5871016304219325184


def fig_full_rvs_example():
    """Figure: full_rvs_spec_giant.pdf

    One member of the Ca II triplet emission group found by clustering the
    residuals of the full-sample outliers. Built through the same
    ``_save_spectrum_fig`` helper as the binned spectrum figures, so the panel
    layout, line markers and styling match the rest of the paper exactly.
    """
    from plot_final_outlier_spectra import load_outlier_inputs

    d = np.load(FULL_RVS_WEIGHTS)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    state = RESULTS_DIR / f"converged_state_R{K}_Q{Q:.2f}_bin_full_rvs_allrows.npz"
    λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
        FULL_RVS_WEIGHTS, state, float(d["threshold"]), None, "all"
    )
    ids = np.array([m["source_id"] for m in meta])
    hits = np.flatnonzero(ids == FULL_RVS_EXAMPLE)
    if not len(hits):
        raise SystemExit(f"{FULL_RVS_EXAMPLE} is not among the flagged spectra")
    i = int(hits[0])

    _save_spectrum_fig(
        λ_grid=λ_grid,
        flux=Y[i],
        reconstruction=recon[i],
        robust_weights=robust[i],
        source_id=FULL_RVS_EXAMPLE,
        i_bin=0,  # unused by the figure itself; the full-sample fit has no bins
        idx=i,
        per_object_weight=float(meta[i]["score"]),
        best_K=K,
        best_Q=Q,
        filename="full_rvs_spec_giant.pdf",
        suptitle_kwargs=dict(
            t=r"$\textsf{\textbf{Gaia RVS: Chromospherically Active Giant}}$",
            fontsize="24",
            c="dimgrey",
            y=0.955,
        ),
    )


FIGURES = {
    "hr_bins": fig_hr_bins,
    "stacked_hist": fig_stacked_hist,
    "gaia_spec_1": fig_gaia_spec_1,
    "gaia_spec_normal": fig_gaia_spec_normal,
    "gaia_spec_2a": fig_gaia_spec_2a,
    "gaia_spec_2b": fig_gaia_spec_2b,
    "gaia_spec_2": fig_gaia_spec_2,
    "full_rvs_cv": fig_full_rvs_cv,
    "full_rvs_outliers": fig_full_rvs_outliers,
    "full_rvs_hr_weights": fig_full_rvs_hr_weights,
    "full_rvs_example": fig_full_rvs_example,
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate Gaia paper figures into paper/figs.")
    parser.add_argument("figure", choices=list(FIGURES) + ["all"], help="Figure to regenerate.")
    args = parser.parse_args()

    targets = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in targets:
        print(f"== {name} ==")
        FIGURES[name]()


if __name__ == "__main__":
    main()
