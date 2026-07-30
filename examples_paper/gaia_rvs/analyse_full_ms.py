"""
Analyse the full main-sequence RHMF models trained by train_full_ms.py:
cross-validation over the (K, Q) grid, best-model selection, and outlier
identification -- the full-dataset analysis requested by the referee.

Reuses the per-bin machinery from analysis_funcs.py end to end (the full-MS
sample is treated as one big bin with tag "full_ms"), so scoring metrics,
inference batching, and outlier scoring are identical to the per-bin analysis
in the paper. The outlier score is the 1st-percentile robust weight per
spectrum, matching analyse_bins.py.

OUTPUTS (in ./gaia_rvs_results and ./plots_full_ms):
    full_ms_cv_scores.npz             -- all four CV metrics over the grid
    inferred_all_data_R*_bin_full_ms.npz -- cached best-model inference
    full_ms_outliers.csv              -- source ids + scores of outliers
    plots_full_ms/cv_heatmaps.pdf     -- CV metric grids
    plots_full_ms/weights_hist.pdf    -- per-spectrum weight distribution
    plots_full_ms/basis_vectors.pdf   -- best-model eigenspectra

USAGE:
    uv run python analyse_full_ms.py
    (grid must match what train_full_ms.py trained; override with --ranks/--q-vals)
"""

import argparse
from pathlib import Path

import gaia_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_funcs import (
    batched_infer,
    clip_edge_pix,
    compute_all_cv_scores,
    compute_outlier_scores,
    find_best_model,
    get_outlier_indices,
    get_test_train_split_idx,
    load_bin_results,
    load_cached_inferred_state,
    prep_data,
)
from train_full_ms import BIN_TAG, Q_VALS, RANKS, RESULTS_DIR, build_full_ms_sample

from robusta_hmf import save_state_to_npz

plt.style.use("mpl_drip.custom")

PLOTS_DIR = Path("./plots_full_ms")
WEIGHT_THRESHOLD = 0.5
OUTLIER_SCORE_FUNC = lambda w: np.percentile(w, 1)  # matches analyse_bins.py
BEST_MODEL_METRIC = "std_z"


def load_all_full_ms_data(data, idx, train_frac=cfg.TRAIN_FRAC):
    """All (train + test) Y, W for the full-MS sample, plus the split."""
    train_idx, test_idx = get_test_train_split_idx(len(idx), train_frac=train_frac)
    all_flux, all_u_flux = clip_edge_pix(*data.get_flux_batch(idx))
    all_Y, all_W = prep_data(all_flux, all_u_flux)
    return all_Y, all_W, train_idx, test_idx


def plot_cv_heatmaps(cv_scores, out):
    metrics = [("std_z", "std(z) [target 1]"), ("chi2_red", r"$\chi^2_{\rm red}$ [target 1]"),
               ("rmse", "weighted RMSE"), ("mad_z", "MAD(z) [target 0.6745]")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=100)
    for ax, (name, label) in zip(axes, metrics):
        vals = getattr(cv_scores, name)
        im = ax.imshow(vals, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(cv_scores.q_vals)))
        ax.set_xticklabels([f"{q:g}" for q in cv_scores.q_vals])
        ax.set_yticks(range(len(cv_scores.ranks)))
        ax.set_yticklabels(cv_scores.ranks)
        ax.set_xlabel("Q")
        ax.set_ylabel("K")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
    fig.suptitle(r"$\textsf{\textbf{Full Main Sequence: CV Scores}}$",
                fontsize="24", c="dimgrey", y=1.05)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_weight_hist(outlier_scores, threshold, out):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(outlier_scores, bins=100, color="C0", alpha=0.8)
    ax.axvline(threshold, color="grey", ls="--", label=f"Threshold ({threshold})")
    ax.set_yscale("log")
    ax.set_xlabel("1st-Percentile Robust Weight per Spectrum")
    ax.set_ylabel("Count")
    ax.legend()
    fig.suptitle(r"$\textsf{\textbf{Full Main Sequence: Outlier Scores}}$",
                fontsize="24", c="dimgrey", y=0.96)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_basis(rhmf, state, λ_grid, out, max_show=10):
    basis = rhmf.basis_vectors(state=state)  # (K, M)
    n_show = min(max_show, basis.shape[0])
    fig, ax = plt.subplots(figsize=(12, 1.2 * n_show + 2), dpi=100)
    for k in range(n_show):
        ax.plot(λ_grid, basis[k] / np.linalg.norm(basis[k]) + 0.12 * k,
                color=f"C{k % 10}", lw=1.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized basis + offset")
    fig.suptitle(r"$\textsf{\textbf{Full Main Sequence: Best-Model Eigenspectra}}$",
                fontsize="24", c="dimgrey", y=0.99)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main(ranks, q_vals, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Building full main-sequence sample...")
    data, idx, ids = build_full_ms_sample()
    print(f"Full-MS sample: {len(idx)} unique spectra")

    print("Loading spectra...")
    all_Y, all_W, train_idx, test_idx = load_all_full_ms_data(data, idx)
    Y_test, W_test = all_Y[test_idx], all_W[test_idx]

    print("Loading trained models...")
    results = load_bin_results(BIN_TAG, ranks, q_vals, results_dir)
    n_expected = len(ranks) * len(q_vals)
    if len(results.rhmf_objs) < n_expected:
        raise SystemExit(
            f"Only {len(results.rhmf_objs)}/{n_expected} models found -- "
            "finish train_full_ms.py first (all shards)."
        )

    print("Computing CV scores on the held-out test set...")
    cv_scores = compute_all_cv_scores(results, Y_test, W_test)

    np.savez(
        results_dir / "full_ms_cv_scores.npz",
        std_z=cv_scores.std_z, chi2_red=cv_scores.chi2_red,
        rmse=cv_scores.rmse, mad_z=cv_scores.mad_z,
        ranks=ranks, q_vals=q_vals,
    )
    plot_cv_heatmaps(cv_scores, plots_dir / "cv_heatmaps.pdf")

    best_K, best_Q, best_idx = find_best_model(cv_scores, metric=BEST_MODEL_METRIC)
    print(f"\nBest model by {BEST_MODEL_METRIC}: K={best_K}, Q={best_Q:.2f}")

    best_rhmf = results.rhmf_objs[best_idx]
    cached = load_cached_inferred_state(BIN_TAG, best_K, best_Q, results_dir)
    if cached is not None:
        print("Loaded cached all-data inference")
        best_state = cached
    else:
        print("Inferring best model on all data (batched)...")
        best_state = batched_infer(
            best_rhmf, all_Y, all_W,
            batch_size=50_000, max_iter=1000, conv_tol=1e-4, conv_check_cadence=5,
        )
        save_state_to_npz(
            best_state,
            results_dir / f"inferred_all_data_R{best_K}_Q{best_Q:.2f}_bin_{BIN_TAG}.npz",
        )

    print("Computing outlier scores...")
    outlier_scores, _ = compute_outlier_scores(
        best_rhmf, all_Y, all_W, best_state, score_func=OUTLIER_SCORE_FUNC
    )
    outlier_indices = get_outlier_indices(outlier_scores, WEIGHT_THRESHOLD)
    print(f"Found {len(outlier_indices)} outliers "
          f"({100 * len(outlier_indices) / len(idx):.2f}% of {len(idx)})")

    pd.DataFrame({
        "idx": outlier_indices,
        "source_id": ids[outlier_indices],
        "score": outlier_scores[outlier_indices],
        "best_K": best_K,
        "best_Q": best_Q,
        "in_train": np.isin(outlier_indices, train_idx),
    }).sort_values("score").to_csv(results_dir / "full_ms_outliers.csv", index=False)
    print(f"Wrote {results_dir / 'full_ms_outliers.csv'}")

    λ_grid = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]
    plot_weight_hist(outlier_scores, WEIGHT_THRESHOLD, plots_dir / "weights_hist.pdf")
    plot_basis(best_rhmf, best_state, λ_grid, plots_dir / "basis_vectors.pdf")

    print("\nDone. Summary:")
    print(f"  Sample:      {len(idx)} spectra")
    print(f"  Best model:  K={best_K}, Q={best_Q:.2f} (by {BEST_MODEL_METRIC})")
    print(f"  Outliers:    {len(outlier_indices)} (score < {WEIGHT_THRESHOLD})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--q-vals", type=float, nargs="+", default=Q_VALS)
    args = parser.parse_args()
    main(args.ranks, args.q_vals)
