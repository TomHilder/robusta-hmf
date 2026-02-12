"""
Re-plot outlier spectra from saved data, with zero HDF5 access.

This script loads the outlier_data.npz files saved by analyse_bins.py
and regenerates the per-outlier spectrum+residual plots. Use this when
you want to tweak plot styling without re-running the expensive inference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from analysis_funcs import load_outlier_data, plot_spectrum_residual

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Which bins to re-plot (None = all bins with saved outlier data)
BINS_TO_REPLOT = None

# Best model selection metric (must match analyse_bins.py)
BEST_MODEL_METRIC = "std_z"

# Directories (must match analyse_bins.py)
PLOTS_DIR = Path("./plots_analysis")

# Best model params per bin — loaded from summary.json automatically
# (no need to set manually)

# ============================================================================ #


def find_bins_with_outlier_data(plots_dir):
    """Scan for bins that have saved outlier_data.npz files."""
    available = []
    for d in sorted(plots_dir.glob("bin_*")):
        if (d / "outlier_data.npz").exists():
            i_bin = int(d.name.split("_")[1])
            available.append(i_bin)
    return available


def replot(
    bins_to_replot=BINS_TO_REPLOT,
    best_model_metric=BEST_MODEL_METRIC,
    plots_dir=PLOTS_DIR,
):
    """Re-plot outlier spectra from saved outlier_data.npz files."""
    import json

    import numpy as np

    plots_dir = Path(plots_dir) / best_model_metric
    if not plots_dir.exists():
        print(f"Error: {plots_dir} does not exist. Run analyse_bins.py first.")
        return

    # Find available bins
    if bins_to_replot is None:
        bins_to_replot = find_bins_with_outlier_data(plots_dir)

    if len(bins_to_replot) == 0:
        print("No bins found with saved outlier data.")
        return

    print(f"Re-plotting outliers for {len(bins_to_replot)} bins: {bins_to_replot}")

    total_plotted = 0
    for i_bin in bins_to_replot:
        # Load outlier data
        odata = load_outlier_data(plots_dir, i_bin)
        if odata is None:
            print(f"  Bin {i_bin}: no outlier_data.npz found, skipping")
            continue

        # Load summary for best_K, best_Q
        summary_path = plots_dir / f"bin_{i_bin:02d}" / "summary.json"
        if not summary_path.exists():
            print(f"  Bin {i_bin}: no summary.json found, skipping")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        best_K = summary["best_K"]
        best_Q = summary["best_Q"]

        n_outliers = len(odata["indices"])
        print(f"  Bin {i_bin}: re-plotting {n_outliers} outliers (K={best_K}, Q={best_Q:.2f})")

        bin_plots_dir = plots_dir / f"bin_{i_bin:02d}"
        for j in range(n_outliers):
            per_object_weight = float(odata["scores"][j])
            plot_spectrum_residual(
                λ_grid=odata["lambda_grid"],
                flux=odata["flux"][j],
                reconstruction=odata["reconstructions"][j],
                robust_weights=odata["robust_weights"][j],
                source_id=int(odata["source_ids"][j]),
                i_bin=i_bin,
                idx=int(odata["indices"][j]),
                per_object_weight=per_object_weight,
                best_K=best_K,
                best_Q=best_Q,
                save_dir=bin_plots_dir,
            )
        total_plotted += n_outliers

    print(f"\nDone. Re-plotted {total_plotted} outlier spectra.")


if __name__ == "__main__":
    replot()
