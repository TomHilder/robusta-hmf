"""
Per-bin analysis of Robusta results for Gaia RVS spectra.

This script:
1. Loads trained models for each bin
2. Computes CV scores and selects best (K, Q)
3. Identifies outlier spectra (low robust weight)
4. Generates per-bin plots
5. Saves per-bin results to disk (outliers.csv, summary.json, outlier_data.npz,
   inferred state)

Run summarise_bins.py afterwards for cross-bin summary plots (HR diagram,
combined outlier CSV, best model vs bin).
"""

import gc
from pathlib import Path

import gaia_config as cfg
import jax
import matplotlib.pyplot as plt
import numpy as np
from analysis_funcs import (
    build_bins_from_config,
    compute_bin_analysis,
    plot_bin_analysis,
    save_bin_results,
)

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Whether to save residuals?
SAVE_RESIDUALS = True

# Which bins to analyse (None = all bins with results, or list like [7, 8, 9])
# BINS_TO_ANALYSE = [0]
# BINS_TO_ANALYSE = [1, 2, 3]
BINS_TO_ANALYSE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# BINS_TO_ANALYSE = [5, 6, 7, 8, 9, 10, 11, 12, 13]
# BINS_TO_ANALYSE = [4]

# Model grid (must match what was trained)
# For CV grid searches, use e.g. RANKS = [3, 4, 5, 6, 7, 8, 9, 10]
RANKS = [10]
Q_VALS = [5.0]

# Train/test split (from shared config to ensure consistency)
TRAIN_FRAC = cfg.TRAIN_FRAC

# Outlier detection
WEIGHT_THRESHOLD = 0.5

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
OUTLIER_SCORE_FUNC = lambda w: np.percentile(w, 1)  # None = default (median)

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

    # Save per-bin results for summarise_bins.py and replot_outliers.py
    save_bin_results(analysis, plots_dir, results_dir)
    if verbose:
        print(f"Saved per-bin results for bin {i_bin}")

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
    Analyse multiple bins: compute, plot, and save per-bin results.

    Per-bin results are saved to disk by save_bin_results().
    Run summarise_bins.py afterwards for cross-bin plots.

    Parameters
    ----------
    bins_to_analyse : list of int or None
        Bin indices to analyse. None = all bins (0 to N_BINS-1)
    ranks, q_vals, train_frac, weight_threshold, outlier_score_func, results_dir, plots_dir
        See analyse_bin
    """
    plots_dir = Path(plots_dir) / best_model_metric
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data and bins
    print("Loading data and building bins...")
    data, bins, _, _ = build_bins_from_config()

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

        # Free memory between bins
        del analysis
        plt.close("all")
        gc.collect()
        jax.clear_caches()

    # Print summary
    n_outliers_total = sum(len(df) for df in all_outliers)
    print(f"\nDone. Analysed {len(best_models)} bins, found {n_outliers_total} total outliers.")
    print("Per-bin results saved. Run summarise_bins.py for cross-bin plots.")

    # Print best models summary
    print("\nBest models per bin:")
    for i_bin, (K, Q) in sorted(best_models.items()):
        print(f"  Bin {i_bin:2d}: K={K}, Q={Q:.2f}")

    # Close data file
    data.close()


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
