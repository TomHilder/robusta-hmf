"""
Analyse Robusta results across multiple bins of Gaia RVS spectra.

This script:
1. Loads trained models for each bin
2. Computes CV scores and plots heatmaps
3. Selects best (K, Q) based on std(z) closest to 1.0
4. Identifies outlier spectra (low robust weight)
5. Plots residuals for each outlier
6. Saves summary CSV with all outliers for cross-referencing
"""

from pathlib import Path

import gaia_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_funcs import (
    BinAnalysis,
    build_bins_from_config,
    compute_bin_analysis,
    plot_best_model_vs_bin,
    plot_bin_analysis,
    plot_outliers_on_hr,
)
from tqdm import tqdm

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Whether to save residuals?
SAVE_RESIDUALS = True

# Which bins to analyse (None = all bins with results, or list like [7, 8, 9])
BINS_TO_ANALYSE = list(range(14))

# Model grid (must match what was trained)
# For CV grid searches, use e.g. RANKS = [3, 4, 5, 6, 7, 8, 9, 10]
RANKS = [5]
Q_VALS = [3.0]

# Train/test split (from shared config to ensure consistency)
TRAIN_FRAC = cfg.TRAIN_FRAC

# Outlier detection
WEIGHT_THRESHOLD = 0.9

# Best model selection metric
# Options: "std_z" (default), "chi2_red", "rmse", "mad_z"
#   - std_z: std of z-scores, target 1.0
#   - chi2_red: reduced chi-squared, target 1.0
#   - rmse: weighted RMSE, lower is better
#   - mad_z: median absolute z-score, target 0.6745
# BEST_MODEL_METRIC = "std_z"
BEST_MODEL_METRIC = "std_z"

# Custom outlier scoring function (optional)
# Takes per-pixel weights (1D array) and returns a scalar score.
# Lower score = more outlier-y. Set to None for default (median).
#
# Example custom functions:
#   lambda w: np.median(w)                          # default: median weight
#   lambda w: np.min(w)                             # minimum weight
#   lambda w: np.percentile(w, 10)                  # 10th percentile
#   lambda w: np.mean(w < 0.5)                      # fraction of pixels below 0.5 (higher = outlier, so negate or use 1-)
#   lambda w: np.sum(w < 0.3)                       # count of very low weight pixels (higher = outlier)
#   lambda w: -np.sum(w < 0.3)                      # negated so lower = more outlier-y
#   lambda w: 1 - np.mean(w < 0.5)                  # 1 - fraction below threshold
#
OUTLIER_SCORE_FUNC = None  # None = default (median)

# Directories
RESULTS_DIR = Path("./gaia_rvs_results")
PLOTS_DIR = Path("./plots_analysis")
RESIDUALS_DIR = Path("./residuals")

# ============================================================================ #


def analyse_bin(
    i_bin,
    data,
    bins,
    ranks=RANKS,
    q_vals=Q_VALS,
    train_frac=TRAIN_FRAC,
    weight_threshold=WEIGHT_THRESHOLD,
    outlier_score_func=OUTLIER_SCORE_FUNC,
    best_model_metric=BEST_MODEL_METRIC,
    results_dir=RESULTS_DIR,
    plots_dir=PLOTS_DIR,
    verbose=True,
):
    """
    Analyse a single bin: compute CV scores, find best model, identify outliers,
    and generate all plots.

    Returns
    -------
    BinAnalysis or None
        None if the bin has 0 spectra or no trained models.
    """
    analysis = compute_bin_analysis(
        i_bin=i_bin,
        data=data,
        bins=bins,
        ranks=ranks,
        q_vals=q_vals,
        train_frac=train_frac,
        weight_threshold=weight_threshold,
        results_dir=results_dir,
        best_model_metric=best_model_metric,
        outlier_score_func=outlier_score_func,
        verbose=verbose,
    )

    if analysis is None:
        return None

    plot_bin_analysis(
        analysis=analysis,
        plots_dir=plots_dir,
        save_residuals=SAVE_RESIDUALS,
        residuals_dir=RESIDUALS_DIR,
        verbose=verbose,
    )

    return analysis


def analyse_all_bins(
    bins_to_analyse=BINS_TO_ANALYSE,
    ranks=RANKS,
    q_vals=Q_VALS,
    train_frac=TRAIN_FRAC,
    weight_threshold=WEIGHT_THRESHOLD,
    outlier_score_func=OUTLIER_SCORE_FUNC,
    best_model_metric=BEST_MODEL_METRIC,
    results_dir=RESULTS_DIR,
    plots_dir=PLOTS_DIR,
):
    """
    Analyse multiple bins and save combined outlier summary.

    Parameters
    ----------
    bins_to_analyse : list of int or None
        Bin indices to analyse. None = all bins (0 to N_BINS-1)
    ranks, q_vals, train_frac, weight_threshold, outlier_score_func, results_dir, plots_dir
        See analyse_bin

    Returns
    -------
    all_outliers_df : pd.DataFrame
        Combined outlier summary across all bins
    """
    plots_dir = Path(plots_dir) / best_model_metric
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data and bins
    print("Loading data and building bins...")
    data, bins, bp_rp, abs_mag_G = build_bins_from_config()
    source_ids_all = data["source_id"]

    # Determine which bins to analyse
    if bins_to_analyse is None:
        bins_to_analyse = list(range(cfg.N_BINS))

    print(f"Bins to analyse: {bins_to_analyse}")
    print(f"Ranks: {ranks}")
    print(f"Q values: {q_vals}")
    print(f"Best model metric: {best_model_metric}")
    print(f"Weight threshold: {weight_threshold}")

    # Analyse each bin
    all_outliers = []
    best_models = {}

    for i_bin in bins_to_analyse:
        if i_bin < 0 or i_bin >= len(bins):
            print(f"Warning: bin {i_bin} out of range, skipping")
            continue

        analysis = analyse_bin(
            i_bin=i_bin,
            data=data,
            bins=bins,
            ranks=ranks,
            q_vals=q_vals,
            train_frac=train_frac,
            weight_threshold=weight_threshold,
            outlier_score_func=outlier_score_func,
            best_model_metric=best_model_metric,
            results_dir=results_dir,
            plots_dir=plots_dir,
        )

        if analysis is None:
            continue

        if len(analysis.outliers_df) > 0:
            all_outliers.append(analysis.outliers_df)

        best_models[i_bin] = (analysis.best_K, analysis.best_Q)

    # Combine and save
    if all_outliers:
        all_outliers_df = pd.concat(all_outliers, ignore_index=True)

        # Sort by score (most anomalous = lowest score first)
        all_outliers_df = all_outliers_df.sort_values("score").reset_index(drop=True)

        # Save CSV
        csv_path = plots_dir / "outlier_summary.csv"
        all_outliers_df.to_csv(csv_path, index=False)
        print(f"\nSaved outlier summary to {csv_path}")
        print(f"Total outliers: {len(all_outliers_df)}")

        # Check for duplicates across bins (same source_id)
        duplicate_sources = all_outliers_df[all_outliers_df.duplicated("source_id", keep=False)]
        if len(duplicate_sources) > 0:
            n_unique_duplicates = duplicate_sources["source_id"].nunique()
            print(f"Found {n_unique_duplicates} sources appearing as outliers in multiple bins")
    else:
        all_outliers_df = pd.DataFrame()
        print("\nNo outliers found across any bins")

    # Print best models summary
    print("\nBest models per bin:")
    for i_bin, (K, Q) in sorted(best_models.items()):
        print(f"  Bin {i_bin:2d}: K={K}, Q={Q:.2f}")

    # Save best models to CSV
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

    # Plot best model parameters vs bin
    if len(best_models) > 0:
        print("\nPlotting optimal K and Q vs bin...")
        plot_best_model_vs_bin(
            best_models=best_models,
            save_path=plots_dir / "best_model_vs_bin.pdf",
        )

    # Plot outliers on HR diagram
    if len(all_outliers_df) > 0:
        print("\nPlotting outliers on HR diagram...")
        plot_outliers_on_hr(
            outliers_df=all_outliers_df,
            bp_rp_all=bp_rp,
            abs_mag_G_all=abs_mag_G,
            source_ids_all=source_ids_all,
            save_path=plots_dir / "outliers_hr_diagram_by_score.pdf",
            color_by="score",
        )
        plot_outliers_on_hr(
            outliers_df=all_outliers_df,
            bp_rp_all=bp_rp,
            abs_mag_G_all=abs_mag_G,
            source_ids_all=source_ids_all,
            save_path=plots_dir / "outliers_hr_diagram_by_bin.pdf",
            color_by="bin",
        )
        print(f"Saved HR diagram plots to {plots_dir}")

    # Close data file
    data.close()

    return all_outliers_df


def main():
    """Run analysis using configuration at top of file."""
    analyse_all_bins(
        bins_to_analyse=BINS_TO_ANALYSE,
        ranks=RANKS,
        q_vals=Q_VALS,
        train_frac=TRAIN_FRAC,
        weight_threshold=WEIGHT_THRESHOLD,
        outlier_score_func=OUTLIER_SCORE_FUNC,
        best_model_metric=BEST_MODEL_METRIC,
        results_dir=RESULTS_DIR,
        plots_dir=PLOTS_DIR,
    )


if __name__ == "__main__":
    main()
