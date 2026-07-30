#!/usr/bin/env python
"""
TASK 8: Apply RHMF to full Gaia main sequence dataset (not binned).

This script applies RHMF to all main-sequence stellar spectra at once, demonstrating
the method's ability to handle a large heterogeneous dataset and automatically identify
outliers across all stellar types simultaneously.

INPUTS:
  - gaia_rvs_results/collected_spectra.npz (from collect.py)

OUTPUTS:
  - gaia_rvs_results/full_dataset_rhmf_K{rank}_Q{Q:.2f}.npz (converged state)
  - gaia_rvs_results/full_dataset_analysis_{rank}_{Q:.2f}.npz (outliers, weights, etc.)
  - plots_analysis/full_dataset_* (diagnostic plots)

USAGE:
  uv run python task_8_gaia_full_dataset.py [--ranks 5 6 7 8 9 10] [--q-vals 3.0 4.0 5.0]

NOTES:
  - Estimated runtime: 4-8 GPU hours for full grid
  - Can be run in parallel with task_9 on separate GPUs
  - Requires significant memory for full dataset (~300M spectra × 2361 pixels)
"""

import argparse
import gc
from pathlib import Path

import gaia_config as cfg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState

plt.style.use("mpl_drip.custom")


def load_full_gaia_dataset(results_dir=Path("./gaia_rvs_results")):
    """Load collected Gaia RVS spectra (full dataset, not binned)."""
    data_file = results_dir / "collected_spectra.npz"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Run collect.py first to gather spectra from Gaia database."
        )

    print(f"Loading full dataset from {data_file}...")
    data = np.load(data_file)

    # Expected keys: 'spectra', 'ivar', 'wavelength', 'source_id', etc.
    spectra = data["spectra"]  # (N, M)
    ivar = data["ivar"]  # (N, M)
    source_id = data["source_id"] if "source_id" in data else np.arange(spectra.shape[0])

    print(f"  Loaded {spectra.shape[0]} spectra with {spectra.shape[1]} pixels each")
    print(f"  Spectra shape: {spectra.shape}, ivar shape: {ivar.shape}")

    return spectra, ivar, source_id


def setup_train_test_split(N, train_frac=cfg.TRAIN_FRAC, seed=cfg.RNG_SEED):
    """Create train/test split for cross-validation."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    split = int(N * train_frac)
    train_idx = indices[:split]
    test_idx = indices[split:]
    return train_idx, test_idx


def train_rhmf_model(
    spectra,
    ivar,
    rank,
    Q,
    train_idx,
    max_iter=200,
    conv_tol=1e-2,
):
    """Train RHMF model on training set."""
    print(f"\n{'='*70}")
    print(f"Training RHMF: K={rank}, Q={Q:.2f}")
    print(f"{'='*70}")

    # Extract training data
    Y_train = spectra[train_idx]
    W_train = ivar[train_idx]

    print(f"Training data shape: {Y_train.shape}")

    # Initialize model
    model = Robusta(rank=rank, robust_scale=Q)

    # Fit on training data
    print("Fitting model...")
    state, loss_history = model.fit(
        Y=Y_train,
        W=W_train,
        max_iter=max_iter,
        conv_tol=conv_tol,
        conv_check_cadence=5,
    )

    print(f"Converged after {len(loss_history)} iterations")
    print(f"Final loss: {loss_history[-1]:.6e}")

    return model, state


def evaluate_on_test_set(model, state, spectra, ivar, test_idx, results_dir):
    """Evaluate model on test set and compute CV score."""
    print(f"\nEvaluating on test set ({len(test_idx)} spectra)...")

    Y_test = spectra[test_idx]
    W_test = ivar[test_idx]

    # Infer on test set
    test_state, _ = model.infer(
        Y_infer=Y_test,
        W_infer=W_test,
        max_iter=100,
        conv_tol=1e-2,
    )

    # Compute reconstruction error
    Y_pred = test_state.A @ test_state.G
    residuals = (Y_test - Y_pred) ** 2 * W_test
    cv_score = np.sqrt(np.mean(residuals))

    print(f"Test CV score (RMSE): {cv_score:.6e}")

    return test_state, cv_score


def identify_outliers(
    model, state, spectra, ivar, source_id, weight_threshold=0.5, results_dir=Path("./")
):
    """Identify outliers based on robust weights."""
    print("\nIdentifying outliers...")

    # Compute robust weights
    weights = model.robust_weights(spectra, ivar, state=state)  # (N, M)
    per_object_weights = np.median(weights, axis=1)  # (N,)

    # Identify outliers
    outlier_mask = per_object_weights < weight_threshold
    n_outliers = np.sum(outlier_mask)

    print(f"  Found {n_outliers} outliers ({100*n_outliers/len(per_object_weights):.1f}%)")

    # Save outlier list
    outlier_indices = np.where(outlier_mask)[0]
    outlier_source_ids = source_id[outlier_indices]

    return {
        "outlier_indices": outlier_indices,
        "outlier_source_ids": outlier_source_ids,
        "per_object_weights": per_object_weights,
        "per_pixel_weights": weights,
    }


def save_results(
    model, state, cv_score, outlier_info, spectra, ivar, source_id, rank, Q, results_dir
):
    """Save model state, CV scores, and outlier list to disk."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    state_file = results_dir / f"full_dataset_rhmf_K{rank}_Q{Q:.2f}.npz"
    state.save(state_file)
    print(f"\nSaved model state to {state_file}")

    # Save analysis results
    analysis_file = results_dir / f"full_dataset_analysis_{rank}_{Q:.2f}.npz"
    np.savez(
        analysis_file,
        cv_score=cv_score,
        outlier_indices=outlier_info["outlier_indices"],
        outlier_source_ids=outlier_info["outlier_source_ids"],
        per_object_weights=outlier_info["per_object_weights"],
        source_id=source_id,
        rank=rank,
        Q=Q,
    )
    print(f"Saved analysis to {analysis_file}")

    return state_file, analysis_file


def plot_results(
    spectra, ivar, source_id, per_object_weights, rank, Q, plots_dir=Path("./plots_analysis")
):
    """Create diagnostic plots."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Weight distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(per_object_weights, bins=100, color="C0", alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Outlier threshold (0.5)")
    ax.set_xlabel("Median Robust Weight per Spectrum", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Full Dataset Weight Distribution (K={rank}, Q={Q:.2f})", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = plots_dir / f"full_dataset_weights_K{rank}_Q{Q:.2f}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved weight distribution to {out}")


def main(ranks, q_vals, results_dir="./gaia_rvs_results", plots_dir="./plots_analysis"):
    """Run full dataset analysis for a grid of (rank, Q) values."""

    # Load data
    spectra, ivar, source_id = load_full_gaia_dataset(results_dir)

    # Train/test split
    train_idx, test_idx = setup_train_test_split(len(spectra))

    # Results accumulator
    results_summary = []

    # Grid search over (rank, Q)
    for rank in ranks:
        for Q in q_vals:
            try:
                # Train model
                model, state = train_rhmf_model(
                    spectra, ivar, rank, Q, train_idx, max_iter=200, conv_tol=1e-2
                )

                # Evaluate on test set
                test_state, cv_score = evaluate_on_test_set(
                    model, state, spectra, ivar, test_idx, results_dir
                )

                # Identify outliers
                outlier_info = identify_outliers(
                    model, state, spectra, ivar, source_id, weight_threshold=0.5
                )

                # Save results
                state_file, analysis_file = save_results(
                    model, state, cv_score, outlier_info, spectra, ivar, source_id,
                    rank, Q, results_dir
                )

                # Plot results
                plot_results(spectra, ivar, source_id, outlier_info["per_object_weights"],
                           rank, Q, plots_dir)

                # Record summary
                results_summary.append({
                    "rank": rank,
                    "Q": Q,
                    "cv_score": float(cv_score),
                    "n_outliers": len(outlier_info["outlier_indices"]),
                    "outlier_fraction": len(outlier_info["outlier_indices"]) / len(spectra),
                })

                # Clean up GPU memory
                gc.collect()
                jax.effects_barrier()

            except Exception as e:
                print(f"\nERROR for K={rank}, Q={Q:.2f}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save summary
    summary_file = Path(results_dir) / "full_dataset_summary.npy"
    np.save(summary_file, results_summary)
    print(f"\n\nSaved summary to {summary_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Full Dataset RHMF Analysis")
    print("="*70)
    print(f"{'K':>4} {'Q':>6} {'CV Score':>12} {'N Outliers':>12} {'Outlier %':>10}")
    print("-"*70)
    for r in results_summary:
        print(
            f"{r['rank']:>4} {r['Q']:>6.2f} {r['cv_score']:>12.6e} {r['n_outliers']:>12} {r['outlier_fraction']:>9.1%}"
        )
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply RHMF to full Gaia main sequence dataset (Task 8)"
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8, 9, 10],
        help="Ranks to test (default: 5 6 7 8 9 10)",
    )
    parser.add_argument(
        "--q-vals",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 5.0],
        help="Robust scales to test (default: 3.0 4.0 5.0)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./gaia_rvs_results",
        help="Results directory (default: ./gaia_rvs_results)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./plots_analysis",
        help="Plots directory (default: ./plots_analysis)",
    )
    args = parser.parse_args()

    main(
        ranks=args.ranks,
        q_vals=args.q_vals,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
    )
