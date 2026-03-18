"""
Plot non-outlier spectra (highest per-object weights) for comparison with outliers.

Loads HDF5 data + cached inference state per bin, computes robust weights,
picks the top N spectra by median weight, and generates the same 3-panel
residual plots as for outliers.
"""

import json

import numpy as np
import matplotlib.pyplot as plt

import gaia_config as cfg
from analysis_funcs import (
    build_bins_from_config,
    load_all_spectra_for_bin,
    load_cached_inferred_state,
    plot_spectrum_residual,
)
from robusta_hmf import Robusta

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Which bins to plot (None = all bins with saved results)
BINS_TO_PLOT = None

# Max non-outlier spectra to plot per bin
N_PER_BIN = 10

# Must match analyse_bins.py
BEST_MODEL_METRIC = "std_z"
PLOTS_DIR = "plots_analysis"
RESULTS_DIR = "gaia_rvs_results"
TRAIN_FRAC = cfg.TRAIN_FRAC

# ============================================================================ #


def plot_non_outliers(
    bins_to_plot=BINS_TO_PLOT,
    n_per_bin=N_PER_BIN,
    best_model_metric=BEST_MODEL_METRIC,
    plots_dir=PLOTS_DIR,
    results_dir=RESULTS_DIR,
    train_frac=TRAIN_FRAC,
):
    from pathlib import Path

    plots_dir = Path(plots_dir) / best_model_metric
    results_dir = Path(results_dir)

    if not plots_dir.exists():
        print(f"Error: {plots_dir} does not exist. Run analyse_bins.py first.")
        return

    # Find bins with saved results
    if bins_to_plot is None:
        bins_to_plot = []
        for d in sorted(plots_dir.glob("bin_*")):
            if (d / "summary.json").exists():
                bins_to_plot.append(int(d.name.split("_")[1]))

    if not bins_to_plot:
        print("No bins found with saved results.")
        return

    # Load HDF5 data and bin definitions once
    print("Loading data and bin definitions...")
    data, bins, bp_rp, abs_mag_G = build_bins_from_config()

    print(f"Plotting non-outliers for {len(bins_to_plot)} bins: {bins_to_plot}")

    total_plotted = 0
    for i_bin in bins_to_plot:
        # Load summary for best_K, best_Q
        summary_path = plots_dir / f"bin_{i_bin:02d}" / "summary.json"
        if not summary_path.exists():
            print(f"  Bin {i_bin}: no summary.json, skipping")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        best_K = summary["best_K"]
        best_Q = summary["best_Q"]

        # Load cached inferred state
        cached_state = load_cached_inferred_state(i_bin, best_K, best_Q, results_dir)
        if cached_state is None:
            print(f"  Bin {i_bin}: no cached state, skipping")
            continue

        # Load spectra for this bin
        bin_data = bins[i_bin]
        all_Y, all_W, _, _, source_ids = load_all_spectra_for_bin(
            data, bin_data, train_frac
        )

        # Create Robusta object and compute robust weights + reconstructions
        rhmf = Robusta(rank=best_K, robust_scale=best_Q)
        rhmf._state = cached_state

        all_pixel_weights = rhmf.robust_weights(all_Y, all_W, state=cached_state)
        per_object_scores = np.median(all_pixel_weights, axis=1)
        reconstructions = rhmf.synthesize(state=cached_state)

        # Pick top N by per-object weight (highest = least outlier-y)
        top_indices = np.argsort(per_object_scores)[-n_per_bin:][::-1]

        # Wavelength grid
        λ_grid = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]

        print(
            f"  Bin {i_bin}: plotting {len(top_indices)} non-outliers "
            f"(K={best_K}, Q={best_Q:.2f})"
        )

        save_dir = plots_dir / f"bin_{i_bin:02d}" / "non_outliers"
        for idx in top_indices:
            plot_spectrum_residual(
                λ_grid=λ_grid,
                flux=all_Y[idx],
                reconstruction=reconstructions[idx],
                robust_weights=all_pixel_weights[idx],
                source_id=int(source_ids[idx]),
                i_bin=i_bin,
                idx=int(idx),
                per_object_weight=float(per_object_scores[idx]),
                best_K=best_K,
                best_Q=best_Q,
                save_dir=save_dir,
            )
        total_plotted += len(top_indices)

        # Clean up
        del all_Y, all_W, all_pixel_weights, reconstructions

    print(f"\nDone. Plotted {total_plotted} non-outlier spectra.")


if __name__ == "__main__":
    plot_non_outliers()
