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


FIGURES = {
    "hr_bins": fig_hr_bins,
    "stacked_hist": fig_stacked_hist,
    "gaia_spec_1": fig_gaia_spec_1,
    "gaia_spec_normal": fig_gaia_spec_normal,
    "gaia_spec_2a": fig_gaia_spec_2a,
    "gaia_spec_2b": fig_gaia_spec_2b,
    "gaia_spec_2": fig_gaia_spec_2,
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
