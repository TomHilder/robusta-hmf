from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import mpl_drip
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

plt.style.use("mpl_drip.custom")

RESIDUALS_DIR = Path("./residuals")

if __name__ == "__main__":
    # Load all residuals from .npy files in the specified directory
    residuals_files = sorted(glob(str(RESIDUALS_DIR / "*.npy")))
    all_residuals = []
    for file in residuals_files:
        residuals = np.load(file)
        all_residuals.append(residuals)
    print(f"Number of files loaded: {len(all_residuals)}")
    all_residuals = np.vstack(all_residuals)
    print(all_residuals.shape)

    # Standardize the residuals
    # scaler = StandardScaler()
    # all_residuals_scaled = scaler.fit_transform(all_residuals)

    # Apply UMAP for dimensionality reduction
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(all_residuals)
    # embedding = reducer.fit_transform(all_residuals_scaled)
    print("UMAP embedding shape:", embedding.shape)

    # Plot the UMAP results
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=25, alpha=1.0)
    plt.title("UMAP Projection of Residuals")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    # plt.grid(True)
    plt.savefig("umap_residuals.png", dpi=300)
    plt.show()
