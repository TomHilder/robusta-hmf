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

        # Cross-bin outlier consistency analysis
        # 1. Build bin membership: which bins does each source belong to?
        source_to_bins = {}  # source_id -> set of bin indices
        for i_bin in bins_to_summarise:
            sids_path = plots_dir / f"bin_{i_bin:02d}" / "all_source_ids.npy"
            if sids_path.exists():
                sids = np.load(sids_path)
                for sid in sids:
                    source_to_bins.setdefault(int(sid), set()).add(i_bin)

        # 2. For outlier sources, compare bins-as-outlier vs bins-as-member
        outlier_source_ids = all_outliers_df["source_id"].unique()
        outlier_bins = all_outliers_df.groupby("source_id")["bin"].apply(set)

        n_multi_bin_member = 0  # outliers that belong to >1 bin
        n_outlier_in_all = 0  # of those, outlier in ALL their bins
        n_outlier_in_some = 0  # of those, outlier in only SOME bins

        for sid in outlier_source_ids:
            member_bins = source_to_bins.get(int(sid), set())
            if len(member_bins) <= 1:
                continue
            n_multi_bin_member += 1
            o_bins = outlier_bins.get(sid, set())
            if o_bins >= member_bins:  # outlier in all bins it belongs to
                n_outlier_in_all += 1
            else:
                n_outlier_in_some += 1

        print(f"Unique outlier sources: {len(outlier_source_ids)}")
        if n_multi_bin_member > 0:
            print(f"Outlier sources belonging to >1 bin: {n_multi_bin_member}")
            print(
                f"  Outlier in ALL their bins: {n_outlier_in_all} "
                f"({100 * n_outlier_in_all / n_multi_bin_member:.1f}%)"
            )
            print(
                f"  Outlier in only SOME bins: {n_outlier_in_some} "
                f"({100 * n_outlier_in_some / n_multi_bin_member:.1f}%)"
            )
        else:
            if len(source_to_bins) == 0:
                print(
                    "No all_source_ids.npy files found — "
                    "re-run analyse_bins.py to generate them"
                )
            else:
                print("No outlier sources belong to multiple bins")
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


def dual_bin_score_analysis(
    bins_to_summarise=BINS_TO_SUMMARISE,
    best_model_metric=BEST_MODEL_METRIC,
    plots_dir=PLOTS_DIR,
    threshold=0.5,
):
    """Cross-bin score comparison for sources that fall in exactly two bins.

    For every source that lives in two overlapping bins, load the per-object
    outlier score in each bin and report:
      - How many are outliers in both / only one / neither bin under `threshold`.
      - Distribution of |score_A - score_B| across all dual-bin sources.
      - For sources flagged as outlier in only one of their two bins, the
        distribution of the OTHER (non-flagged) bin's score — i.e. how close
        the missed bin was to crossing the threshold.

    Saves a per-source CSV of pairs and a (score_A, score_B) scatter plot.
    Reads `all_source_ids.npy` + `all_outlier_scores.npy` already on disk;
    nothing is recomputed.
    """
    plots_dir = Path(plots_dir) / best_model_metric
    if not plots_dir.exists():
        print(f"Error: {plots_dir} does not exist.")
        return

    if bins_to_summarise is None:
        bins_to_summarise = find_available_bins(plots_dir)

    if len(bins_to_summarise) == 0:
        print("No bins found.")
        return

    # Build {source_id: {bin_idx: score}} over every (source, bin) pair on disk.
    source_scores = {}
    for i_bin in bins_to_summarise:
        sids_path = plots_dir / f"bin_{i_bin:02d}" / "all_source_ids.npy"
        scores_path = plots_dir / f"bin_{i_bin:02d}" / "all_outlier_scores.npy"
        if not (sids_path.exists() and scores_path.exists()):
            print(f"Skipping bin {i_bin}: missing all_source_ids/all_outlier_scores")
            continue
        sids = np.load(sids_path)
        scores = np.load(scores_path)
        for sid, sc in zip(sids, scores):
            source_scores.setdefault(int(sid), {})[i_bin] = float(sc)

    # Restrict to sources living in exactly two bins (matches paper's framing).
    dual = {sid: b for sid, b in source_scores.items() if len(b) == 2}
    multi = {sid: b for sid, b in source_scores.items() if len(b) > 2}
    print(f"Sources in exactly 2 bins: {len(dual)}")
    if multi:
        print(f"Sources in >2 bins (excluded): {len(multi)}")

    if len(dual) == 0:
        print("No dual-bin sources found.")
        return

    rows = []
    for sid, bins in dual.items():
        (bA, sA), (bB, sB) = sorted(bins.items())
        rows.append(
            {
                "source_id": sid,
                "bin_A": bA,
                "score_A": sA,
                "bin_B": bB,
                "score_B": sB,
                "delta": abs(sA - sB),
                "outlier_A": sA < threshold,
                "outlier_B": sB < threshold,
            }
        )
    df = pd.DataFrame(rows)

    both = df[df["outlier_A"] & df["outlier_B"]]
    one = df[df["outlier_A"] ^ df["outlier_B"]]
    neither = df[~df["outlier_A"] & ~df["outlier_B"]]
    any_ = df[df["outlier_A"] | df["outlier_B"]]

    print(f"\nThreshold: w < {threshold}")
    print(f"  Outlier in both bins:        {len(both)}")
    print(f"  Outlier in only one bin:     {len(one)}")
    print(f"  Outlier in neither bin:      {len(neither)}")
    print(f"  Outlier in at least one bin: {len(any_)}")

    print(f"\n|score_A - score_B| across all {len(df)} dual-bin sources:")
    print(f"  median   = {df['delta'].median():.3f}")
    print(f"  mean     = {df['delta'].mean():.3f}")
    print(f"  90 %ile  = {df['delta'].quantile(0.9):.3f}")
    print(f"  max      = {df['delta'].max():.3f}")

    if len(both) > 0:
        print(f"\n|score_A - score_B| across the {len(both)} 'outlier in both bins' sources:")
        print(f"  median   = {both['delta'].median():.3f}")
        print(f"  mean     = {both['delta'].mean():.3f}")
        print(f"  90 %ile  = {both['delta'].quantile(0.9):.3f}")
        print(f"  max      = {both['delta'].max():.3f}")
        print(f"  mean of min(score_A, score_B) = {np.minimum(both['score_A'], both['score_B']).mean():.3f}")
        print(f"  mean of max(score_A, score_B) = {np.maximum(both['score_A'], both['score_B']).mean():.3f}")

    if len(one) > 0:
        print(f"\n|score_A - score_B| across the {len(one)} 'outlier in only one bin' sources:")
        print(f"  median   = {one['delta'].median():.3f}")
        print(f"  mean     = {one['delta'].mean():.3f}")
        print(f"  90 %ile  = {one['delta'].quantile(0.9):.3f}")
        print(f"  max      = {one['delta'].max():.3f}")
        print(f"  mean of min(score_A, score_B) = {np.minimum(one['score_A'], one['score_B']).mean():.3f}")
        print(f"  mean of max(score_A, score_B) = {np.maximum(one['score_A'], one['score_B']).mean():.3f}")

        other_scores = np.where(one["outlier_A"], one["score_B"], one["score_A"])
        print(
            f"\nFor the {len(one)} 'outlier in only one bin' sources, "
            f"the OTHER bin's score:"
        )
        print(f"  median               = {np.median(other_scores):.3f}")
        print(f"  mean                 = {np.mean(other_scores):.3f}")
        print(
            f"  fraction within 0.1 of threshold = "
            f"{np.mean(np.abs(other_scores - threshold) < 0.1):.2f}"
        )
        print(
            f"  fraction within 0.2 of threshold = "
            f"{np.mean(np.abs(other_scores - threshold) < 0.2):.2f}"
        )

    csv_path = plots_dir / "dual_bin_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-source pairs to {csv_path}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["score_A"], df["score_B"], s=8, alpha=0.5)
    ax.plot([0, 1], [0, 1], "k-", lw=0.8, alpha=0.4)
    ax.axvline(threshold, color="r", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(threshold, color="r", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$w_i^{\rm object}$ in bin A")
    ax.set_ylabel(r"$w_i^{\rm object}$ in bin B")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plot_path = plots_dir / "dual_bin_scores_scatter.pdf"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved scatter to {plot_path}")

    if len(both) > 0 and len(one) > 0:
        delta_max = max(both["delta"].max(), one["delta"].max())
        bins = np.linspace(0, delta_max * 1.02, 25)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            both["delta"],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            label=f"outlier in both bins ($N={len(both)}$)",
        )
        ax.hist(
            one["delta"],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            label=f"outlier in only one bin ($N={len(one)}$)",
        )
        ax.set_xlabel(r"$|w_i^{\rm object, A} - w_i^{\rm object, B}|$")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        hist_path = plots_dir / "dual_bin_delta_hist.pdf"
        fig.savefig(hist_path)
        plt.close(fig)
        print(f"Saved |Δscore| histogram to {hist_path}")


if __name__ == "__main__":
    summarise()
