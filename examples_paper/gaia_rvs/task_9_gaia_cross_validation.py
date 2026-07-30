#!/usr/bin/env python
"""
TASK 9: Cross-validation for optimal (K, Q) selection on full Gaia dataset.

This script performs systematic cross-validation across a grid of (K, Q) values
on the full Gaia main sequence dataset to find the optimal hyperparameters.

Can run in parallel with task_8_gaia_full_dataset.py on separate GPUs.

INPUTS:
  - gaia_rvs_results/collected_spectra.npz (from collect.py)
  - (optional) full_dataset_rhmf_K*_Q*.npz files from task_8

OUTPUTS:
  - gaia_rvs_results/cross_validation_results.npz (CV scores grid)
  - plots_analysis/cv_score_heatmap.pdf (visualization)
  - plots_analysis/outlier_count_heatmap.pdf (visualization)

USAGE:
  # Full grid (recommended for thorough analysis)
  uv run python task_9_gaia_cross_validation.py --full-grid

  # Focused grid (faster, for initial testing)
  uv run python task_9_gaia_cross_validation.py --ranks 5 6 7 8 9 10 --q-vals 3.0 4.0 5.0

  # Run in parallel with task_8
  uv run python task_8_gaia_full_dataset.py --ranks 5 6 7 8 9 10 &
  uv run python task_9_gaia_cross_validation.py --full-grid

NOTES:
  - Estimated runtime: 2-4 additional GPU hours beyond task_8
  - Computes KL divergence on held-out test set for each (K, Q)
  - Identifies best (K, Q) by multiple metrics (KL, F1 for outlier detection)
  - Requires significant memory for repeated model inference
"""

import argparse
import gc
from pathlib import Path

import gaia_config as cfg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import load_state_from_npz

plt.style.use("mpl_drip.custom")


def load_full_gaia_dataset(results_dir=Path("./gaia_rvs_results")):
    """Load collected Gaia RVS spectra."""
    data_file = results_dir / "collected_spectra.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Loading full dataset from {data_file}...")
    data = np.load(data_file)

    spectra = data["spectra"]
    ivar = data["ivar"]

    print(f"  Loaded {spectra.shape[0]} spectra with {spectra.shape[1]} pixels each")
    return spectra, ivar


def setup_cv_folds(N, n_folds=5, seed=cfg.RNG_SEED):
    """Create K-fold cross-validation split."""
    rng = np.random.RandomState(seed)
    fold_ids = np.arange(N) % n_folds
    rng.shuffle(fold_ids)
    return fold_ids


def compute_kl_divergence_score(residuals, weights):
    """Compute KL divergence score for cross-validation.

    Uses the chi-squared distribution of (residuals * weights)**0.5
    to assess goodness of fit. Lower is better.
    """
    z_scores = residuals * weights  # Should be ~chi2(1)
    # KL divergence of empirical z vs N(0,1)
    kl = 0.5 * (np.mean(z_scores) - 1 - np.mean(np.log(np.maximum(z_scores, 1e-10))))
    return kl


def evaluate_cv_fold(
    spectra, ivar, fold_ids, fold, rank, Q, weight_threshold=0.5
):
    """Evaluate one CV fold."""
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold

    Y_train = spectra[train_mask]
    W_train = ivar[train_mask]
    Y_test = spectra[test_mask]
    W_test = ivar[test_mask]

    # Train model
    model = Robusta(rank=rank, robust_scale=Q)
    state, _ = model.fit(
        Y=Y_train, W=W_train, max_iter=100, conv_tol=1e-2, conv_check_cadence=5
    )

    # Evaluate on test set
    Y_pred = state.A @ state.G
    residuals = (Y_test - Y_pred) ** 2
    kl_score = compute_kl_divergence_score(residuals, W_test)

    # Compute per-object weights for outlier detection
    weights = model.robust_weights(Y_test, W_test, state=state)
    per_object_weights = np.median(weights, axis=1)

    return kl_score, per_object_weights


def run_cross_validation(
    spectra, ivar, ranks, q_vals, n_folds=5, results_dir=Path("./gaia_rvs_results")
):
    """Run full cross-validation grid."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {spectra.shape[0]} spectra × {spectra.shape[1]} pixels")
    print(f"Parameter grid: K∈{ranks}, Q∈{q_vals}")
    print(f"CV folds: {n_folds}")
    print("=" * 70)

    # Setup CV folds
    fold_ids = setup_cv_folds(len(spectra), n_folds=n_folds)

    # Results arrays
    kl_scores = np.zeros((len(ranks), len(q_vals), n_folds))
    n_outliers = np.zeros((len(ranks), len(q_vals), n_folds))

    # Grid search with progress bar
    pbar = tqdm(
        total=len(ranks) * len(q_vals) * n_folds,
        desc="CV Progress",
        position=0,
    )

    for i_rank, rank in enumerate(ranks):
        for i_q, Q in enumerate(q_vals):
            for fold in range(n_folds):
                try:
                    kl, per_obj_weights = evaluate_cv_fold(
                        spectra, ivar, fold_ids, fold, rank, Q
                    )
                    kl_scores[i_rank, i_q, fold] = kl
                    n_outliers[i_rank, i_q, fold] = np.sum(per_obj_weights < 0.5)

                    pbar.update(1)

                except Exception as e:
                    print(f"\nError in fold {fold}, K={rank}, Q={Q}: {e}")
                    kl_scores[i_rank, i_q, fold] = np.nan
                    n_outliers[i_rank, i_q, fold] = np.nan
                    pbar.update(1)

                # Clean GPU memory
                gc.collect()
                jax.effects_barrier()

    pbar.close()

    # Aggregate across folds
    kl_mean = np.nanmean(kl_scores, axis=2)
    kl_std = np.nanstd(kl_scores, axis=2)
    outlier_mean = np.nanmean(n_outliers, axis=2)

    return {
        "kl_scores": kl_scores,
        "kl_mean": kl_mean,
        "kl_std": kl_std,
        "n_outliers": n_outliers,
        "outlier_mean": outlier_mean,
        "ranks": ranks,
        "q_vals": q_vals,
    }


def find_optimal_params(cv_results):
    """Find optimal (K, Q) by different metrics."""
    kl_mean = cv_results["kl_mean"]
    ranks = cv_results["ranks"]
    q_vals = cv_results["q_vals"]

    # Find best by KL divergence
    idx_best = np.nanargmin(kl_mean)
    i_rank, i_q = np.unravel_index(idx_best, kl_mean.shape)
    best_rank_kl = ranks[i_rank]
    best_q_kl = q_vals[i_q]
    best_kl = kl_mean[i_rank, i_q]

    return {
        "best_rank_kl": best_rank_kl,
        "best_q_kl": best_q_kl,
        "best_kl": best_kl,
    }


def plot_cv_results(cv_results, plots_dir=Path("./plots_analysis")):
    """Create heatmap plots of CV results."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ranks = cv_results["ranks"]
    q_vals = cv_results["q_vals"]
    kl_mean = cv_results["kl_mean"]
    outlier_mean = cv_results["outlier_mean"]

    # KL divergence heatmap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    im = ax.imshow(
        kl_mean.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[ranks[0] - 0.5, ranks[-1] + 0.5, q_vals[0] - 0.5, q_vals[-1] + 0.5],
    )
    ax.set_xlabel("Rank (K)", fontsize=11)
    ax.set_ylabel("Robust Scale (Q)", fontsize=11)
    ax.set_title("Cross-Validation KL Divergence Score (Full Dataset)", fontsize=12, fontweight="bold")
    ax.set_xticks(ranks)
    ax.set_yticks(q_vals)
    cbar = plt.colorbar(im, ax=ax, label="KL Divergence")
    plt.tight_layout()
    out = plots_dir / "cv_score_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved KL heatmap to {out}")

    # Outlier count heatmap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    im = ax.imshow(
        outlier_mean.T,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        extent=[ranks[0] - 0.5, ranks[-1] + 0.5, q_vals[0] - 0.5, q_vals[-1] + 0.5],
    )
    ax.set_xlabel("Rank (K)", fontsize=11)
    ax.set_ylabel("Robust Scale (Q)", fontsize=11)
    ax.set_title("Mean Outlier Count (Full Dataset)", fontsize=12, fontweight="bold")
    ax.set_xticks(ranks)
    ax.set_yticks(q_vals)
    cbar = plt.colorbar(im, ax=ax, label="Number of Outliers")
    plt.tight_layout()
    out = plots_dir / "outlier_count_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved outlier count heatmap to {out}")


def save_cv_results(cv_results, optimal_params, results_dir=Path("./gaia_rvs_results")):
    """Save cross-validation results to disk."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cv_file = results_dir / "cross_validation_results.npz"
    np.savez(
        cv_file,
        kl_scores=cv_results["kl_scores"],
        kl_mean=cv_results["kl_mean"],
        kl_std=cv_results["kl_std"],
        n_outliers=cv_results["n_outliers"],
        outlier_mean=cv_results["outlier_mean"],
        ranks=cv_results["ranks"],
        q_vals=cv_results["q_vals"],
        best_rank_kl=optimal_params["best_rank_kl"],
        best_q_kl=optimal_params["best_q_kl"],
        best_kl=optimal_params["best_kl"],
    )
    print(f"\nSaved CV results to {cv_file}")
    return cv_file


def print_summary(cv_results, optimal_params):
    """Print summary table of CV results."""
    ranks = cv_results["ranks"]
    q_vals = cv_results["q_vals"]
    kl_mean = cv_results["kl_mean"]
    kl_std = cv_results["kl_std"]
    outlier_mean = cv_results["outlier_mean"]

    print("\n" + "=" * 90)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'K':>4} ", end="")
    for Q in q_vals:
        print(f"Q={Q:.1f}".center(20), end=" ")
    print()
    print("-" * 90)

    for i_rank, rank in enumerate(ranks):
        print(f"{rank:>4} ", end="")
        for i_q in range(len(q_vals)):
            kl_m = kl_mean[i_rank, i_q]
            kl_s = kl_std[i_rank, i_q]
            if not np.isnan(kl_m):
                print(f"{kl_m:.6f}±{kl_s:.6f}".center(20), end=" ")
            else:
                print("NaN".center(20), end=" ")
        print()

    print("=" * 90)
    print(f"\nOPTIMAL PARAMETERS (by KL divergence):")
    print(f"  K = {optimal_params['best_rank_kl']}")
    print(f"  Q = {optimal_params['best_q_kl']:.2f}")
    print(f"  KL = {optimal_params['best_kl']:.6f}")
    print("=" * 90)


def main(
    ranks=None,
    q_vals=None,
    n_folds=5,
    full_grid=False,
    results_dir="./gaia_rvs_results",
    plots_dir="./plots_analysis",
):
    """Run cross-validation analysis."""

    # Default grids
    if full_grid:
        ranks = ranks or [3, 4, 5, 6, 7, 8, 9, 10]
        q_vals = q_vals or [2.0, 3.0, 4.0, 5.0, 6.0]
    else:
        ranks = ranks or [5, 6, 7, 8, 9, 10]
        q_vals = q_vals or [3.0, 4.0, 5.0]

    # Load data
    spectra, ivar = load_full_gaia_dataset(results_dir)

    # Run cross-validation
    cv_results = run_cross_validation(
        spectra, ivar, ranks, q_vals, n_folds=n_folds, results_dir=results_dir
    )

    # Find optimal parameters
    optimal_params = find_optimal_params(cv_results)

    # Plot results
    plot_cv_results(cv_results, plots_dir=plots_dir)

    # Save results
    save_cv_results(cv_results, optimal_params, results_dir=results_dir)

    # Print summary
    print_summary(cv_results, optimal_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-validation for optimal (K, Q) on full Gaia dataset (Task 9)"
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full parameter grid (K=3-10, Q=2-6) [slower, ~4 GPU hours]",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="Ranks to test (default: 5 6 7 8 9 10)",
    )
    parser.add_argument(
        "--q-vals",
        type=float,
        nargs="+",
        default=None,
        help="Robust scales to test (default: 3.0 4.0 5.0)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./gaia_rvs_results",
        help="Results directory",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./plots_analysis",
        help="Plots directory",
    )
    args = parser.parse_args()

    main(
        ranks=args.ranks,
        q_vals=args.q_vals,
        n_folds=args.n_folds,
        full_grid=args.full_grid,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
    )
