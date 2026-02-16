from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot_bins import PAPER_PLOTS_DIR

plt.style.use("mpl_drip.custom")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Which bins to re-plot (None = all bins with saved outlier data)
BINS_TO_REPLOT = None

# Best model selection metric (must match analyse_bins.py)
BEST_MODEL_METRIC = "std_z"

# Directories (must match analyse_bins.py)
PLOTS_DIR = Path("./plots_analysis")


# Best model params per bin — loaded from summary.json automatically
# (no need to set manually)

# ============================================================================ #


def plot_stacked_hist(
    bins_to_replot=BINS_TO_REPLOT,
    best_model_metric=BEST_MODEL_METRIC,
    plots_dir=PLOTS_DIR,
    save_dir=None,
    n_bins_hist=80,
):
    plots_dir = Path(plots_dir) / best_model_metric
    if not plots_dir.exists():
        print(f"Error: {plots_dir} does not exist. Run analyse_bins.py first.")
        return

    # Find available bins (those with all_outlier_scores.npy)
    if bins_to_replot is None:
        bins_to_replot = []
        for d in sorted(plots_dir.glob("bin_*")):
            if (d / "all_outlier_scores.npy").exists():
                i_bin = int(d.name.split("_")[1])
                bins_to_replot.append(i_bin)

    if len(bins_to_replot) == 0:
        print("No bins found with saved scores. Run analyse_bins.py first.")
        return

    print(f"Plotting weights histograms for {len(bins_to_replot)} bins: {bins_to_replot}")

    cols = sns.hls_palette(n_colors=len(bins_to_replot), l=0.4, s=0.65)

    gs = grid_spec.GridSpec(len(bins_to_replot), 1)
    fig = plt.figure(dpi=100, figsize=[10, 16], layout="compressed")

    ax_objs = []

    bins = np.linspace(0.0, 1.0, n_bins_hist + 1)

    for i, i_bin in enumerate(bins_to_replot):
        scores = np.load(plots_dir / f"bin_{i_bin:02d}" / "all_outlier_scores.npy")

        n_outliers = np.sum(scores < 0.5)
        # print(f"Bin {i_bin}: {n_outliers} outliers (score < 0.5) out of {len(scores)} total")

        ax = fig.add_subplot(gs[i : i + 1, 0:])
        n, bins, patches = ax.hist(
            scores,
            bins=bins,
            color=cols[i],
            label=f"Bin {i_bin}",
            density=True,
            histtype="stepfilled",
            linewidth=1.0,
            rasterized=True,
        )
        for patch in patches:
            patch.set_facecolor((*cols[i], 1.0))  # fill alpha = 0.4
            patch.set_edgecolor(("white", 1.0))  # edge alpha = 1.0
        ax.set_yscale("log")
        # ax.legend(loc=(0.82, 0.55))
        # Bin number label
        ax.text(
            0.85,
            0.25,
            rf"\textbf{{Bin {i_bin}}}",
            transform=ax.transAxes,
            fontsize=18,
            # fontweight="bold",
            color=cols[i],
        )
        # Label for number of outliers and total
        ax.text(
            0.00,
            0.43,
            rf"{n_outliers} outliers / {len(scores)} spectra",
            # rf"$N_{{\mathrm{{outliers}}}}={n_outliers}$"
            # + "\n"
            # + rf"$N_{{\mathrm{{total}}}}={len(scores)}$",
            transform=ax.transAxes,
            fontsize=16,
            # color=cols[i],
            color="k",
        )

        ax_objs.append(ax)

        # Vertical dashed line at threshold = 0.5
        ax.axvline(0.5, color="grey", linestyle=":", alpha=0.8, ymax=0.6, linewidth=5.0)

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(-0.05, 1.05)
        ax_objs[-1].set_ylim(n[n > 0].min() / 3, n[n > 0].max() * 2)

        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])

        if i == len(bins_to_replot) - 1:
            ax_objs[-1].set_xlabel("Per-Spectrum Weight")
        else:
            ax_objs[-1].set_xticklabels([])

        ax_objs[-1].set_xticks(np.arange(0.0, 1.1, 0.1))

        spines = ["top", "right", "left"]  # , "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].tick_params(which="both", left=False, bottom=True, right=False, top=False)

        # Put ticks on top of histograms
        ax_objs[-1].set_axisbelow(False)

    gs.update(hspace=-0.4)
    fig.supylabel("Log Density", x=0.07)
    fig.suptitle(
        r"$\textsf{\textbf{Gaia Example: Spectrum Weight Distributions}}$",
        fontsize="24",
        c="dimgrey",
        y=0.91,
    )
    if save_dir is not None:
        plt.savefig(save_dir / "stacked_hist.pdf", bbox_inches="tight")
    plt.show()


def plot_outlier_density(
    bins_to_replot=BINS_TO_REPLOT,
    best_model_metric=BEST_MODEL_METRIC,
    plots_dir=PLOTS_DIR,
    save_dir=None,
):
    """x-axis: bin number, y-axis: outlier density (fraction of spectra with score < 0.5)"""
    plots_dir = Path(plots_dir) / best_model_metric
    if not plots_dir.exists():
        print(f"Error: {plots_dir} does not exist. Run analyse_bins.py first.")
        return

    # Find available bins (those with all_outlier_scores.npy)
    if bins_to_replot is None:
        bins_to_replot = []
        for d in sorted(plots_dir.glob("bin_*")):
            if (d / "all_outlier_scores.npy").exists():
                i_bin = int(d.name.split("_")[1])
                bins_to_replot.append(i_bin)

    if len(bins_to_replot) == 0:
        print("No bins found with saved scores. Run analyse_bins.py first.")
        return

    print(f"Plotting outlier density for {len(bins_to_replot)} bins: {bins_to_replot}")

    outlier_densities = []
    for i_bin in bins_to_replot:
        scores = np.load(plots_dir / f"bin_{i_bin:02d}" / "all_outlier_scores.npy")
        n_outliers = np.sum(scores < 0.5)
        outlier_density = n_outliers / len(scores)
        outlier_densities.append(outlier_density)

    plt.figure(figsize=[10, 6], dpi=100)
    plt.plot(bins_to_replot, np.array(outlier_densities) * 100, marker="o", linestyle="-")
    plt.xlabel("Bin Number")
    plt.ylabel(r"Outlier Percentage")
    plt.yscale("log")
    # plt.grid()
    if save_dir is not None:
        plt.savefig(save_dir / "outlier_density.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_stacked_hist()
    plot_outlier_density()
    plot_stacked_hist(save_dir=PAPER_PLOTS_DIR)
    # plot_outlier_density(plots_dir=PAPER_PLOTS_DIR)
