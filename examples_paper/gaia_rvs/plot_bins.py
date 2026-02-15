"""
Plot the Gaia RVS bins on an HR diagram with bin indices and spectra counts.

Bin geometry is imported from gaia_config.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from analysis_funcs import build_bins_from_config

plt.style.use("mpl_drip.custom")

PAPER_PLOTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "papers/robust-hmf/paper/documents/figs"
)
assert PAPER_PLOTS_DIR.exists(), "PAPER_PLOTS_DIR does not exist, please update the path."


def plot_bins(save_path=None):
    """
    Plot the HR diagram with bins, showing index and spectra count for each.

    Parameters
    ----------
    save_path : str or Path, optional
        If provided, save the figure to this path instead of showing.
    """
    print("Loading data and building bins...")
    data, bins, bp_rp, abs_mag_G = build_bins_from_config()

    fig, ax = plt.subplots(dpi=100, figsize=[10, 8], layout="compressed")

    # Change the color of the figure background (inside the axes)
    # ax.set_facecolor("lightblue")

    # Plot all sources as background (edgecolor white for visibility)
    ax.scatter(
        bp_rp,
        abs_mag_G,
        s=0.5,
        alpha=0.5,
        c="k",
        zorder=0,
        marker=".",
        edgecolors="white",
        linewidths=0.02,
        rasterized=True,
    )

    # cols = plt.cm.viridis(np.linspace(0, 1, len(bins)))
    cols = plt.cm.tab20(np.linspace(0, 1, len(bins)))
    cols = sns.hls_palette(n_colors=len(bins), l=0.4, s=0.65)

    # Plot bins with labels
    for i, b in enumerate(bins):
        if b.n_spectra > 0:
            ax.scatter(
                b.bp_rp,
                b.abs_mag_G,
                color=cols[i],
                s=5,
                alpha=0.5,
                zorder=1,
                marker=".",
                edgecolors="white",
                linewidths=0.05,
                rasterized=True,
            )
            # Add text label at bin centre
            ax.annotate(
                f"bin {i}: {b.n_spectra:,}",
                xy=(b.bp_rp_prop.centre, b.abs_mag_G_prop.centre),
                xytext=(23, 23),
                textcoords="offset points",
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=cols[i],
                    edgecolor="white",
                    alpha=0.9,
                ),
                zorder=10,
                rotation=25,
            )

    ax.set_ylim(15, -5)
    ax.set_xlim(-0.5, 3.5)
    # ax.set_xlabel("Color (BP - RP)")
    ax.set_xlabel(r"${\rm BP} - {\rm RP}$ [mag]")
    ax.set_ylabel("G-Band Absolute Magnitude [mag]")
    # ax.set_title("Gaia RVS Spectral Bins")

    fig.suptitle(
        r"$\textsf{\textbf{Gaia Example: Main Sequence Bins}}$",
        fontsize="24",
        c="dimgrey",
        y=1.04,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    plot_bins(save_path="plots_analysis/hr_bins.png")
    plot_bins(save_path=PAPER_PLOTS_DIR / "hr_bins.pdf")
