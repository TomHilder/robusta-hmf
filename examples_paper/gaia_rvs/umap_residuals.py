from glob import glob
from pathlib import Path

import gaia_config as cfg
import matplotlib.pyplot as plt
import mpl_drip
import numpy as np
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler

plt.style.use("mpl_drip.custom")

RESIDUALS_DIR = Path("./residuals")


def get_i_bin(fname):
    # f"{source_id}_residual_bin_{i_bin:02d}.npy"
    basename = Path(fname).stem
    parts = basename.split("_")
    i_bin = int(parts[-1])
    return i_bin


if __name__ == "__main__":
    # Load all residuals from .npy files in the specified directory
    residuals_files = sorted(glob(str(RESIDUALS_DIR / "*.npy")))
    all_residuals = []
    i_bins = []
    for file in residuals_files:
        residuals = np.load(file)
        all_residuals.append(residuals)
        i_bins.append(get_i_bin(file))
    print(f"Number of files loaded: {len(all_residuals)}")
    all_residuals = np.vstack(all_residuals)
    bins = np.array(i_bins)
    print(all_residuals.shape)
    print(bins.shape)

    # Standardize the residuals
    # scaler = StandardScaler()
    # all_residuals_scaled = scaler.fit_transform(all_residuals)

    # Apply UMAP for dimensionality reduction
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reducer = umap.UMAP(random_state=0)
    embedding = reducer.fit_transform(all_residuals)
    # embedding = reducer.fit_transform(all_residuals_scaled)
    print("UMAP embedding shape:", embedding.shape)

    cols = plt.cm.tab20(np.linspace(0, 1, cfg.N_BINS))
    cols = sns.hls_palette(n_colors=cfg.N_BINS, l=0.4, s=0.65)
    colors = [cols[b] for b in bins]

    # Plot the UMAP results
    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=80,
        alpha=1.0,
        c=colors,
        edgecolors="white",
        linewidths=0.5,
    )
    # Legend with one entry per bin (0-13)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Bin {i_bin}",
            markerfacecolor=cols[i_bin],
            markersize=8,
        )
        for i_bin in range(cfg.N_BINS)
    ]
    plt.grid(True, linestyle="-", alpha=0.5)
    # Legend with two columns, add some padding from the axes, also white background box but no frame
    plt.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(1.0, 0.0),
        frameon=True,
        fancybox=True,
        facecolor="white",
        borderaxespad=2.0,
        edgecolor="k",
    )
    plt.title("UMAP Projection of Residuals")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    # plt.grid(True)
    plt.savefig("umap_residuals.png", dpi=300)
    plt.show()
