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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import gaia_config as cfg
from analysis_funcs import (
    build_bins_from_config,
    load_all_spectra_for_bin,
    load_bin_results,
    compute_all_cv_scores,
    compute_outlier_scores,
    get_outlier_indices,
    find_best_model,
    plot_cv_heatmaps,
    plot_spectrum_residual,
    plot_outliers_on_hr,
    plot_weights_histograms,
    plot_all_spectra_hr_by_weight,
    plot_best_model_vs_bin,
    default_outlier_score,
)

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Which bins to analyse (None = all bins with results, or list like [7, 8, 9])
BINS_TO_ANALYSE = [7, 8]

# Model grid (must match what was trained)
RANKS = [3, 4, 5, 6, 7, 8, 9, 10]
Q_VALS = [3.0]

# Train/test split (from shared config to ensure consistency)
TRAIN_FRAC = cfg.TRAIN_FRAC

# Outlier detection
WEIGHT_THRESHOLD = 0.5

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
    results_dir=RESULTS_DIR,
    plots_dir=PLOTS_DIR,
    verbose=True,
):
    """
    Analyse a single bin: compute CV scores, find best model, identify outliers.

    Parameters
    ----------
    i_bin : int
        Bin index
    data : MatchedData
        Data object
    bins : list
        List of Bin objects
    ranks : list of int
        Rank values to analyse
    q_vals : list of float
        Q values to analyse
    train_frac : float
        Train fraction (must match training)
    weight_threshold : float
        Threshold for outlier detection (score < threshold = outlier)
    outlier_score_func : callable or None
        Function taking per-pixel weights (1D array) returning a scalar score.
        Lower score = more outlier-y. Default (None) uses median.
    results_dir : Path
        Directory with saved training results
    plots_dir : Path
        Directory for output plots
    verbose : bool
        Print progress

    Returns
    -------
    outliers_df : pd.DataFrame
        DataFrame with outlier info (idx, source_id, score, etc.)
    best_K : int
    best_Q : float
    """
    bin_data = bins[i_bin]
    bin_plots_dir = plots_dir / f"bin_{i_bin:02d}"
    bin_plots_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Analysing bin {i_bin} | N spectra: {bin_data.n_spectra}")
        print(f"{'=' * 60}")

    if bin_data.n_spectra == 0:
        print(f"Skipping bin {i_bin}: no spectra")
        return pd.DataFrame(), None, None

    # Load all spectra
    if verbose:
        print("Loading spectra...")
    all_Y, all_W, train_idx, test_idx, source_ids = load_all_spectra_for_bin(
        data, bin_data, train_frac
    )

    # Split for CV scoring
    Y_train, W_train = all_Y[train_idx], all_W[train_idx]
    Y_test, W_test = all_Y[test_idx], all_W[test_idx]

    # Load model results
    if verbose:
        print("Loading trained models...")
    bin_results = load_bin_results(i_bin, ranks, q_vals, results_dir)

    if len(bin_results.rhmf_objs) == 0:
        print(f"No models found for bin {i_bin}, skipping")
        return pd.DataFrame(), None, None

    # Compute CV scores (also returns inferred states on all data)
    if verbose:
        print("Computing CV scores and inferring on all data...")
    cv_scores, inferred_states = compute_all_cv_scores(
        bin_results, Y_test, W_test, all_Y, all_W, verbose=verbose
    )

    # Plot CV heatmaps
    if verbose:
        print("Plotting CV heatmaps...")
    plot_cv_heatmaps(cv_scores, i_bin, bin_plots_dir)

    # Find best model
    best_K, best_Q, best_idx = find_best_model(cv_scores)
    if verbose:
        print(f"Best model: K={best_K}, Q={best_Q:.2f} (std_z={cv_scores.std_z.flatten()[best_idx]:.4f})")

    # Get best model and its inferred state
    best_rhmf = bin_results.rhmf_objs[best_idx]
    best_state = inferred_states[best_idx]

    # Compute outlier scores using custom or default function
    outlier_scores, all_robust_weights = compute_outlier_scores(
        best_rhmf, all_Y, all_W, best_state, score_func=outlier_score_func
    )

    # Find outliers
    outlier_indices = get_outlier_indices(outlier_scores, weight_threshold)
    if verbose:
        print(f"Found {len(outlier_indices)} outliers with score < {weight_threshold}")

    # Get wavelength grid
    λ_grid = data.λ_grid[cfg.N_CLIP_PIX:-cfg.N_CLIP_PIX]

    # Get reconstructions
    if verbose and len(outlier_indices) > 0:
        print("Plotting outlier spectra...")

    all_reconstructions = best_rhmf.synthesize(state=best_state)

    # Plot weight histograms
    if verbose:
        print("Plotting weight histograms...")
    plot_weights_histograms(
        per_object_weights=outlier_scores,
        all_pixel_weights=all_robust_weights,
        weight_threshold=weight_threshold,
        i_bin=i_bin,
        best_K=best_K,
        best_Q=best_Q,
        save_dir=bin_plots_dir,
    )

    # Plot HR diagram colored by weight for this bin
    if verbose:
        print("Plotting HR diagram by weight...")
    plot_all_spectra_hr_by_weight(
        per_object_weights=outlier_scores,
        bp_rp=bin_data.bp_rp,
        abs_mag_G=bin_data.abs_mag_G,
        bin_indices=bin_data.idx,
        weight_threshold=weight_threshold,
        i_bin=i_bin,
        best_K=best_K,
        best_Q=best_Q,
        save_dir=bin_plots_dir,
    )

    outliers_data = []
    for idx in tqdm(outlier_indices, desc=f"Plotting outliers for bin {i_bin}", disable=not verbose):
        source_id = source_ids[idx]
        score = outlier_scores[idx]

        # Plot
        plot_spectrum_residual(
            λ_grid=λ_grid,
            flux=all_Y[idx],
            reconstruction=all_reconstructions[idx],
            robust_weights=all_robust_weights[idx],
            source_id=source_id,
            i_bin=i_bin,
            idx=idx,
            per_object_weight=score,
            best_K=best_K,
            best_Q=best_Q,
            save_dir=bin_plots_dir,
        )

        outliers_data.append({
            "bin": i_bin,
            "idx": idx,
            "source_id": source_id,
            "score": score,
            "best_K": best_K,
            "best_Q": best_Q,
            "in_train": idx in train_idx,
        })

    outliers_df = pd.DataFrame(outliers_data)
    return outliers_df, best_K, best_Q


def analyse_all_bins(
    bins_to_analyse=BINS_TO_ANALYSE,
    ranks=RANKS,
    q_vals=Q_VALS,
    train_frac=TRAIN_FRAC,
    weight_threshold=WEIGHT_THRESHOLD,
    outlier_score_func=OUTLIER_SCORE_FUNC,
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
    plots_dir = Path(plots_dir)
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
    print(f"Weight threshold: {weight_threshold}")

    # Analyse each bin
    all_outliers = []
    best_models = {}

    for i_bin in bins_to_analyse:
        if i_bin < 0 or i_bin >= len(bins):
            print(f"Warning: bin {i_bin} out of range, skipping")
            continue

        outliers_df, best_K, best_Q = analyse_bin(
            i_bin=i_bin,
            data=data,
            bins=bins,
            ranks=ranks,
            q_vals=q_vals,
            train_frac=train_frac,
            weight_threshold=weight_threshold,
            outlier_score_func=outlier_score_func,
            results_dir=results_dir,
            plots_dir=plots_dir,
        )

        if len(outliers_df) > 0:
            all_outliers.append(outliers_df)

        if best_K is not None:
            best_models[i_bin] = (best_K, best_Q)

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
        results_dir=RESULTS_DIR,
        plots_dir=PLOTS_DIR,
    )


if __name__ == "__main__":
    main()
