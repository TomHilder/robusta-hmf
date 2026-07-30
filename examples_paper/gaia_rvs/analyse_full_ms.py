"""
Analyse the full-sample RHMF models trained by train_full_ms.py:
cross-validation over the (K, Q) grid, best-model selection, and outlier
identification -- the full-dataset analysis requested by the referee.

Use the same --sample as training: "ms" (main-sequence bin union, default) or
"all" (whole matched RVS catalogue).

Reuses the per-bin machinery from analysis_funcs.py end to end (the sample is
treated as one big bin with tag "full_ms"/"full_rvs"), so scoring metrics,
inference batching, and outlier scoring are identical to the per-bin analysis
in the paper. The outlier score is the 1st-percentile robust weight per
spectrum, matching analyse_bins.py.

CV EFFICIENCY: model ranking is scored on a fixed random subsample of the
held-out test set, capped at --cv-max-test spectra (default 50,000; seeded,
so reproducible). The four CV statistics are means/medians over per-pixel
z-scores -- with >=50k spectra x ~2300 pixels (>1e8 residuals) they are
converged to far better precision than the differences between grid points,
so scoring the full multi-hundred-thousand-spectrum test set would spend GPU
hours changing nothing. Pass --cv-max-test 0 to disable the cap. The final
best-model inference and outlier identification always use ALL spectra.

OUTPUTS (in ./gaia_rvs_results and ./plots_<tag>):
    <tag>_cv_scores.npz               -- all four CV metrics over the grid
    inferred_all_data_R*_bin_<tag>.npz -- cached best-model inference
    <tag>_scores.npz                  -- source id + outlier score for EVERY
                                         spectrum, so the threshold can be
                                         changed without re-running inference
    <tag>_outliers.csv                -- source ids + scores of outliers
    plots_<tag>/cv_heatmaps.pdf       -- CV metric grids
    plots_<tag>/weights_hist.pdf      -- per-spectrum weight distribution
    plots_<tag>/basis_vectors.pdf     -- best-model eigenspectra

USAGE:
    uv run python analyse_full_ms.py [--sample all]
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
from train_full_ms import Q_VALS, RANKS, RESULTS_DIR, build_sample

from robusta_hmf import save_state_to_npz

plt.style.use("mpl_drip.custom")

WEIGHT_THRESHOLD = 0.5
OUTLIER_SCORE_FUNC = lambda w: np.percentile(w, 1, axis=1)  # matches analyse_bins.py
BEST_MODEL_METRIC = "std_z"
CV_MAX_TEST = 50_000  # cap on test spectra used for CV scoring (0 = no cap)


def load_all_full_ms_data(data, idx, train_frac=cfg.TRAIN_FRAC):
    """All (train + test) Y, W for the full-MS sample, plus the split."""
    train_idx, test_idx = get_test_train_split_idx(len(idx), train_frac=train_frac)
    all_flux, all_u_flux = clip_edge_pix(*data.get_flux_batch(idx))
    all_Y, all_W = prep_data(all_flux, all_u_flux)
    return all_Y, all_W, train_idx, test_idx


def plot_cv_heatmaps(cv_scores, out, label):
    metrics = [("std_z", "std(z) [target 1]"), ("chi2_red", r"$\chi^2_{\rm red}$ [target 1]"),
               ("rmse", "weighted RMSE"), ("mad_z", "MAD(z) [target 0.6745]")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=100)
    for ax, (name, metric_label) in zip(axes, metrics):
        vals = getattr(cv_scores, name)
        im = ax.imshow(vals, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(cv_scores.q_vals)))
        ax.set_xticklabels([f"{q:g}" for q in cv_scores.q_vals])
        ax.set_yticks(range(len(cv_scores.ranks)))
        ax.set_yticklabels(cv_scores.ranks)
        ax.set_xlabel("Q")
        ax.set_ylabel("K")
        ax.set_title(metric_label)
        plt.colorbar(im, ax=ax)
    fig.suptitle(rf"$\textsf{{\textbf{{{label}: CV Scores}}}}$",
                fontsize="24", c="dimgrey", y=1.05)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_weight_hist(outlier_scores, threshold, out, label):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(outlier_scores, bins=100, color="C0", alpha=0.8)
    ax.axvline(threshold, color="grey", ls="--", label=f"Threshold ({threshold})")
    ax.set_yscale("log")
    ax.set_xlabel("1st-Percentile Robust Weight per Spectrum")
    ax.set_ylabel("Count")
    ax.legend()
    fig.suptitle(rf"$\textsf{{\textbf{{{label}: Outlier Scores}}}}$",
                fontsize="24", c="dimgrey", y=0.96)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_basis(rhmf, state, λ_grid, out, label, max_show=10):
    basis = rhmf.basis_vectors(state=state)  # (K, M)
    n_show = min(max_show, basis.shape[0])
    fig, ax = plt.subplots(figsize=(12, 1.2 * n_show + 2), dpi=100)
    for k in range(n_show):
        ax.plot(λ_grid, basis[k] / np.linalg.norm(basis[k]) + 0.12 * k,
                color=f"C{k % 10}", lw=1.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized basis + offset")
    fig.suptitle(rf"$\textsf{{\textbf{{{label}: Best-Model Eigenspectra}}}}$",
                fontsize="24", c="dimgrey", y=0.99)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main(ranks, q_vals, sample="ms", cv_max_test=CV_MAX_TEST,
         results_dir=RESULTS_DIR):
    print(f"Building sample '{sample}'...")
    data, idx, ids, tag = build_sample(sample)
    print(f"Sample '{tag}': {len(idx)} unique spectra")

    sample_label = "Full Main Sequence" if sample == "ms" else "Full RVS Sample"
    plots_dir = Path(f"./plots_{tag}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spectra...")
    all_Y, all_W, train_idx, test_idx = load_all_full_ms_data(data, idx)

    # CV efficiency: score models on a fixed, seeded random subsample of the
    # test set. The metrics are converged long before 50k spectra; the final
    # inference/outlier stage below still uses every spectrum.
    if cv_max_test and len(test_idx) > cv_max_test:
        rng = np.random.default_rng(cfg.RNG_SEED)
        cv_test_idx = rng.choice(test_idx, size=cv_max_test, replace=False)
        print(f"CV scoring on {cv_max_test} of {len(test_idx)} test spectra "
              "(seeded subsample; pass --cv-max-test 0 to use all)")
    else:
        cv_test_idx = test_idx
    Y_test, W_test = all_Y[cv_test_idx], all_W[cv_test_idx]

    print("Loading trained models...")
    results = load_bin_results(tag, ranks, q_vals, results_dir)
    n_expected = len(ranks) * len(q_vals)
    if len(results.rhmf_objs) < n_expected:
        raise SystemExit(
            f"Only {len(results.rhmf_objs)}/{n_expected} models found -- "
            "finish train_full_ms.py first (all shards, same --sample)."
        )

    print("Computing CV scores on the held-out test set...")
    cv_scores = compute_all_cv_scores(results, Y_test, W_test)

    np.savez(
        results_dir / f"{tag}_cv_scores.npz",
        std_z=cv_scores.std_z, chi2_red=cv_scores.chi2_red,
        rmse=cv_scores.rmse, mad_z=cv_scores.mad_z,
        ranks=ranks, q_vals=q_vals,
    )
    plot_cv_heatmaps(cv_scores, plots_dir / "cv_heatmaps.pdf", sample_label)

    best_K, best_Q, best_idx = find_best_model(cv_scores, metric=BEST_MODEL_METRIC)
    print(f"\nBest model by {BEST_MODEL_METRIC}: K={best_K}, Q={best_Q:.2f}")

    best_rhmf = results.rhmf_objs[best_idx]
    cached = load_cached_inferred_state(tag, best_K, best_Q, results_dir)
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
            results_dir / f"inferred_all_data_R{best_K}_Q{best_Q:.2f}_bin_{tag}.npz",
        )

    print("Computing outlier scores...")
    # return_weights=False: the per-pixel weight matrix is Y-sized (~9 GB here)
    # and nothing below needs it -- only the per-spectrum score.
    outlier_scores, _ = compute_outlier_scores(
        best_rhmf, all_Y, all_W, best_state, score_func=OUTLIER_SCORE_FUNC,
        return_weights=False, verbose=True,
    )
    outlier_indices = get_outlier_indices(outlier_scores, WEIGHT_THRESHOLD)
    print(f"Found {len(outlier_indices)} outliers "
          f"({100 * len(outlier_indices) / len(idx):.2f}% of {len(idx)})")

    # Scores for EVERY spectrum, not just those past the threshold, so the cut
    # can be revisited without re-running inference.
    in_train = np.zeros(len(idx), dtype=bool)
    in_train[train_idx] = True
    scores_file = results_dir / f"{tag}_scores.npz"
    np.savez(
        scores_file,
        source_id=ids,
        score=outlier_scores,
        in_train=in_train,
        best_K=best_K,
        best_Q=best_Q,
        threshold=WEIGHT_THRESHOLD,
    )
    print(f"Wrote {scores_file} ({len(idx)} spectra)")

    pd.DataFrame({
        "idx": outlier_indices,
        "source_id": ids[outlier_indices],
        "score": outlier_scores[outlier_indices],
        "best_K": best_K,
        "best_Q": best_Q,
        "in_train": np.isin(outlier_indices, train_idx),
    }).sort_values("score").to_csv(results_dir / f"{tag}_outliers.csv", index=False)
    print(f"Wrote {results_dir / f'{tag}_outliers.csv'}")

    λ_grid = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]
    plot_weight_hist(outlier_scores, WEIGHT_THRESHOLD, plots_dir / "weights_hist.pdf", sample_label)
    plot_basis(best_rhmf, best_state, λ_grid, plots_dir / "basis_vectors.pdf", sample_label)

    print("\nDone. Summary:")
    print(f"  Sample:      {tag} ({len(idx)} spectra)")
    print(f"  Best model:  K={best_K}, Q={best_Q:.2f} (by {BEST_MODEL_METRIC})")
    print(f"  Outliers:    {len(outlier_indices)} (score < {WEIGHT_THRESHOLD})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--q-vals", type=float, nargs="+", default=Q_VALS)
    parser.add_argument("--sample", choices=["ms", "all"], default="ms",
                        help="'ms' = main-sequence bin union; 'all' = whole RVS sample")
    parser.add_argument("--cv-max-test", type=int, default=CV_MAX_TEST,
                        help="Max test spectra for CV scoring (0 = use all)")
    args = parser.parse_args()
    main(args.ranks, args.q_vals, sample=args.sample, cv_max_test=args.cv_max_test)
