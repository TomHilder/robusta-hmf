"""
Compare analysis results across different CV metrics.

Reads from plots_analysis/{metric}/ subdirectories.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

PLOTS_DIR = Path("./plots_analysis")
METRICS = ["std_z", "mad_z", "chi2_red", "rmse"]  # Will skip if not found
OUTPUT_DIR = PLOTS_DIR / "comparison"

# ============================================================================ #


def load_metric_results(plots_dir, metric):
    """Load results for a single metric."""
    metric_dir = plots_dir / metric
    if not metric_dir.exists():
        return None, None

    # Load outlier summary
    outlier_csv = metric_dir / "outlier_summary.csv"
    outliers_df = None
    if outlier_csv.exists():
        outliers_df = pd.read_csv(outlier_csv)

    # Load best models from dedicated CSV (includes bins with 0 outliers)
    best_models = {}
    best_models_csv = metric_dir / "best_models.csv"
    if best_models_csv.exists():
        best_models_df = pd.read_csv(best_models_csv)
        for _, row in best_models_df.iterrows():
            best_models[int(row["bin"])] = (int(row["best_K"]), float(row["best_Q"]))
    elif outliers_df is not None and len(outliers_df) > 0:
        # Fallback: extract from outlier summary (won't have bins with 0 outliers)
        for i_bin in outliers_df["bin"].unique():
            bin_data = outliers_df[outliers_df["bin"] == i_bin].iloc[0]
            best_models[i_bin] = (int(bin_data["best_K"]), float(bin_data["best_Q"]))

    return outliers_df, best_models


def load_all_metrics(plots_dir, metrics):
    """Load results for all available metrics."""
    results = {}
    for metric in metrics:
        outliers_df, best_models = load_metric_results(plots_dir, metric)
        if outliers_df is not None or best_models:
            results[metric] = {
                "outliers_df": outliers_df,
                "best_models": best_models,
            }
    return results


def print_summary_table(results):
    """Print summary table comparing metrics."""
    print("\n" + "=" * 70)
    print("COMPARISON ACROSS CV METRICS")
    print("=" * 70)

    # Header
    metrics = list(results.keys())
    print(f"\n{'Metric':<12} | {'N Outliers':>10} | {'Unique Sources':>14} | Best K values")
    print("-" * 70)

    for metric in metrics:
        data = results[metric]
        outliers_df = data["outliers_df"]
        best_models = data["best_models"]

        n_outliers = len(outliers_df) if outliers_df is not None else 0
        n_unique = outliers_df["source_id"].nunique() if outliers_df is not None and len(outliers_df) > 0 else 0

        # Best K values per bin
        if best_models:
            bins_sorted = sorted(best_models.keys())
            k_values = [str(best_models[b][0]) for b in bins_sorted]
            k_str = ",".join(k_values)
        else:
            k_str = "N/A"

        print(f"{metric:<12} | {n_outliers:>10} | {n_unique:>14} | {k_str}")

    print()


def plot_best_k_comparison(results, output_dir):
    """Plot best K vs bin for each metric."""
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    colors = plt.cm.tab10.colors

    for i, (metric, data) in enumerate(results.items()):
        best_models = data["best_models"]
        if not best_models:
            continue

        bins = sorted(best_models.keys())
        k_values = [best_models[b][0] for b in bins]

        ax.plot(bins, k_values, marker=markers[i % len(markers)],
                color=colors[i], label=metric, linewidth=2, markersize=8)

    ax.set_xlabel("Bin Index")
    ax.set_ylabel("Best K (Rank)")
    ax.set_title("Optimal Rank by CV Metric")
    ax.legend()
    ax.set_xticks(bins)
    ax.set_yticks(range(1, max(max(r["best_models"].values())[0] for r in results.values() if r["best_models"]) + 2))

    plt.tight_layout()
    plt.savefig(output_dir / "best_k_comparison.pdf", bbox_inches="tight")
    plt.close()

    return output_dir / "best_k_comparison.pdf"


def plot_outlier_counts(results, output_dir):
    """Plot outlier counts per bin for each metric."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get all bins across all metrics
    all_bins = set()
    for data in results.values():
        if data["outliers_df"] is not None:
            all_bins.update(data["outliers_df"]["bin"].unique())
    all_bins = sorted(all_bins)

    if not all_bins:
        plt.close()
        return None

    width = 0.8 / len(results)
    x = np.arange(len(all_bins))

    for i, (metric, data) in enumerate(results.items()):
        outliers_df = data["outliers_df"]
        if outliers_df is None:
            continue

        counts = [len(outliers_df[outliers_df["bin"] == b]) for b in all_bins]
        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=metric)

    ax.set_xlabel("Bin Index")
    ax.set_ylabel("Number of Outliers")
    ax.set_title("Outlier Counts by CV Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(all_bins)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "outlier_counts_comparison.pdf", bbox_inches="tight")
    plt.close()

    return output_dir / "outlier_counts_comparison.pdf"


def plot_outlier_agreement(results, output_dir):
    """
    Plot agreement between metrics on which sources are outliers.

    Shows what percentage of outliers from metric A are also flagged by metric B.
    Matrix[i,j] = percentage of metric i's outliers that are also in metric j.
    Diagonal is always 100%. Asymmetric because metrics have different totals.
    """
    # Get sets of outlier source_ids for each metric
    outlier_sets = {}
    for metric, data in results.items():
        if data["outliers_df"] is not None and len(data["outliers_df"]) > 0:
            outlier_sets[metric] = set(data["outliers_df"]["source_id"].unique())

    if len(outlier_sets) < 2:
        return None

    # Create agreement matrix (percentage)
    metrics = list(outlier_sets.keys())
    n = len(metrics)
    agreement_matrix = np.zeros((n, n))

    for i, m1 in enumerate(metrics):
        n_m1 = len(outlier_sets[m1])
        for j, m2 in enumerate(metrics):
            shared = len(outlier_sets[m1] & outlier_sets[m2])
            agreement_matrix[i, j] = 100 * shared / n_m1 if n_m1 > 0 else 0

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(agreement_matrix, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(metrics)

    # Add text annotations with contrasting colors
    for i in range(n):
        for j in range(n):
            val = agreement_matrix[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=color)

    ax.set_title("Outlier Agreement Between Metrics")
    ax.set_xlabel("Also flagged by")
    ax.set_ylabel("Outliers from")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percent")

    plt.tight_layout()
    plt.savefig(output_dir / "outlier_agreement.pdf", bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\n  Outlier agreement:")
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i != j:
                pct = agreement_matrix[i, j]
                print(f"    {pct:.0f}% of {m1} outliers are also {m2} outliers")

    return output_dir / "outlier_agreement.pdf"


def main():
    """Run comparison analysis."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results from each metric...")
    results = load_all_metrics(PLOTS_DIR, METRICS)

    if not results:
        print("No metric results found!")
        return

    print(f"Found results for: {list(results.keys())}")

    # Print summary table
    print_summary_table(results)

    # Generate plots
    print("Generating comparison plots...")

    plot1 = plot_best_k_comparison(results, output_dir)
    print(f"  Saved: {plot1}")

    plot2 = plot_outlier_counts(results, output_dir)
    if plot2:
        print(f"  Saved: {plot2}")

    plot3 = plot_outlier_agreement(results, output_dir)
    if plot3:
        print(f"  Saved: {plot3}")

    print(f"\nAll comparison plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
