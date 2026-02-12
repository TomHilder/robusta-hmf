"""
Cross-bin summary of Robusta results for Gaia RVS spectra.

This script loads saved per-bin results (from analyse_bins.py) and generates
combined plots:
- Combined outlier_summary.csv (sorted by score)
- HR diagrams colored by score and by bin
- Best model vs bin plot
- Best models CSV

Run this AFTER analyse_bins.py has been run on all desired bins.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_funcs import (
    build_bins_from_config,
    plot_best_model_vs_bin,
    plot_outliers_on_hr,
)

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Which bins to summarise (None = all bins with saved results)
BINS_TO_SUMMARISE = None

# Best model selection metric (must match what was used in analyse_bins.py)
BEST_MODEL_METRIC = "std_z"

# Directories (must match analyse_bins.py)
PLOTS_DIR = Path("./plots_analysis")

# ============================================================================ #


def find_available_bins(plots_dir):
    """Scan for bins that have saved summary.json files."""
    available = []
    for d in sorted(plots_dir.glob("bin_*")):
        if (d / "summary.json").exists():
            i_bin = int(d.name.split("_")[1])
            available.append(i_bin)
    return available


def load_bin_summary(plots_dir, i_bin):
    """Load summary.json for a bin."""
    path = plots_dir / f"bin_{i_bin:02d}" / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_bin_outliers(plots_dir, i_bin):
    """Load outliers.csv for a bin, or empty DataFrame if none."""
    path = plots_dir / f"bin_{i_bin:02d}" / "outliers.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                return df
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame()


def summarise(
    bins_to_summarise=BINS_TO_SUMMARISE,
    best_model_metric=BEST_MODEL_METRIC,
    plots_dir=PLOTS_DIR,
):
    """Load per-bin results and generate cross-bin summary plots."""
    plots_dir = Path(plots_dir) / best_model_metric
    if not plots_dir.exists():
        print(f"Error: {plots_dir} does not exist. Run analyse_bins.py first.")
        return

    # Find available bins
    if bins_to_summarise is None:
        bins_to_summarise = find_available_bins(plots_dir)

    if len(bins_to_summarise) == 0:
        print("No bins found with saved results.")
        return

    print(f"Summarising {len(bins_to_summarise)} bins: {bins_to_summarise}")

    # Load per-bin summaries and outliers
    all_outliers = []
    best_models = {}

    for i_bin in bins_to_summarise:
        summary = load_bin_summary(plots_dir, i_bin)
        best_models[i_bin] = (summary["best_K"], summary["best_Q"])

        outliers_df = load_bin_outliers(plots_dir, i_bin)
        if len(outliers_df) > 0:
            all_outliers.append(outliers_df)

        print(
            f"  Bin {i_bin:2d}: K={summary['best_K']}, Q={summary['best_Q']:.2f}, "
            f"N={summary['n_spectra']}, outliers={summary['n_outliers']}"
        )

    # Combined outlier CSV
    if all_outliers:
        all_outliers_df = pd.concat(all_outliers, ignore_index=True)
        all_outliers_df = all_outliers_df.sort_values("score").reset_index(drop=True)

        csv_path = plots_dir / "outlier_summary.csv"
        all_outliers_df.to_csv(csv_path, index=False)
        print(f"\nSaved combined outlier summary to {csv_path}")
        print(f"Total outliers: {len(all_outliers_df)}")

        # Check for duplicates
        duplicate_sources = all_outliers_df[
            all_outliers_df.duplicated("source_id", keep=False)
        ]
        if len(duplicate_sources) > 0:
            n_unique_duplicates = duplicate_sources["source_id"].nunique()
            print(
                f"Found {n_unique_duplicates} sources appearing as "
                f"outliers in multiple bins"
            )
    else:
        all_outliers_df = pd.DataFrame()
        print("\nNo outliers found across any bins")

    # Best models CSV
    if best_models:
        best_models_df = pd.DataFrame(
            [
                {"bin": i_bin, "best_K": K, "best_Q": Q}
                for i_bin, (K, Q) in sorted(best_models.items())
            ]
        )
        best_models_csv = plots_dir / "best_models.csv"
        best_models_df.to_csv(best_models_csv, index=False)
        print(f"Saved best models to {best_models_csv}")

    # Best model vs bin plot
    if len(best_models) > 0:
        print("\nPlotting optimal K and Q vs bin...")
        plot_best_model_vs_bin(
            best_models=best_models,
            save_path=plots_dir / "best_model_vs_bin.pdf",
        )

    # HR diagram plots (need global data for background)
    if len(all_outliers_df) > 0:
        print("Loading global data for HR diagram background...")
        data, _, bp_rp, abs_mag_G = build_bins_from_config()
        source_ids_all = data["source_id"]

        print("Plotting outliers on HR diagram (by score)...")
        plot_outliers_on_hr(
            outliers_df=all_outliers_df,
            bp_rp_all=bp_rp,
            abs_mag_G_all=abs_mag_G,
            source_ids_all=source_ids_all,
            save_path=plots_dir / "outliers_hr_diagram_by_score.pdf",
            color_by="score",
        )
        print("Plotting outliers on HR diagram (by bin)...")
        plot_outliers_on_hr(
            outliers_df=all_outliers_df,
            bp_rp_all=bp_rp,
            abs_mag_G_all=abs_mag_G,
            source_ids_all=source_ids_all,
            save_path=plots_dir / "outliers_hr_diagram_by_bin.pdf",
            color_by="bin",
        )
        print(f"Saved HR diagram plots to {plots_dir}")

        data.close()

    print("\nDone.")


if __name__ == "__main__":
    summarise()
