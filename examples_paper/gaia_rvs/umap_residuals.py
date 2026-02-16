from glob import glob
from pathlib import Path

import gaia_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from collect import MatchedData, compute_abs_mag

plt.style.use("mpl_drip.custom")

RESIDUALS_DIR = Path("./residuals")
PLOTS_DIR = Path("./plots_analysis")
BEST_MODEL_METRIC = "std_z"


def get_i_bin(fname):
    # f"{source_id}_residual_bin_{i_bin:02d}.npy"
    basename = Path(fname).stem
    parts = basename.split("_")
    i_bin = int(parts[-1])
    return i_bin


def get_source_id(fname):
    # f"{source_id}_residual_bin_{i_bin:02d}.npy"
    basename = Path(fname).stem
    source_id = int(basename.split("_")[0])
    return source_id


def load_residuals_and_metadata(residuals_dir=RESIDUALS_DIR):
    """
    Load all residuals and build per-outlier metadata arrays.

    Returns
    -------
    all_residuals : (N, n_pixels) array
    metadata : dict of arrays, each length N, with keys:
        "bin", "source_id", "bp_rp", "abs_mag_G", "parallax",
        "phot_g_mean_mag", "score"
    """
    residuals_files = sorted(glob(str(residuals_dir / "*.npy")))
    if len(residuals_files) == 0:
        raise FileNotFoundError(f"No .npy files in {residuals_dir}")

    all_residuals = []
    i_bins = []
    source_ids = []
    for file in residuals_files:
        all_residuals.append(np.load(file))
        i_bins.append(get_i_bin(file))
        source_ids.append(get_source_id(file))

    all_residuals = np.vstack(all_residuals)
    i_bins = np.array(i_bins)
    source_ids = np.array(source_ids)

    print(f"Loaded {len(source_ids)} residuals from {residuals_dir}")

    # Load metadata from CSV (no HDF5/spectra access needed)
    data = MatchedData()
    all_source_ids = data["source_id"]
    all_bp_rp = data["bp_rp"]
    all_phot_g = data["phot_g_mean_mag"]
    all_parallax = data["parallax"]
    all_abs_mag_G = compute_abs_mag(all_phot_g, all_parallax)
    data.close()

    # Build lookup: source_id -> index in full dataset
    sid_to_idx = {int(sid): i for i, sid in enumerate(all_source_ids)}

    # Match outlier source_ids to metadata
    bp_rp = np.full(len(source_ids), np.nan)
    abs_mag_G = np.full(len(source_ids), np.nan)
    parallax = np.full(len(source_ids), np.nan)
    phot_g_mean_mag = np.full(len(source_ids), np.nan)

    for j, sid in enumerate(source_ids):
        idx = sid_to_idx.get(int(sid))
        if idx is not None:
            bp_rp[j] = all_bp_rp[idx]
            abs_mag_G[j] = all_abs_mag_G[idx]
            parallax[j] = all_parallax[idx]
            phot_g_mean_mag[j] = all_phot_g[idx]

    n_matched = np.sum(~np.isnan(bp_rp))
    print(f"Matched {n_matched}/{len(source_ids)} to metadata")

    # Load per-outlier scores from saved outlier_data.npz
    scores = np.full(len(source_ids), np.nan)
    plots_dir = PLOTS_DIR / BEST_MODEL_METRIC
    for i_bin_val in np.unique(i_bins):
        odata_path = plots_dir / f"bin_{i_bin_val:02d}" / "outlier_data.npz"
        if odata_path.exists():
            odata = np.load(odata_path)
            odata_sids = odata["source_ids"]
            odata_scores = odata["scores"]
            sid_to_score = {int(s): sc for s, sc in zip(odata_sids, odata_scores)}
            for j, sid in enumerate(source_ids):
                if i_bins[j] == i_bin_val and int(sid) in sid_to_score:
                    scores[j] = sid_to_score[int(sid)]

    metadata = {
        "bin": i_bins,
        "source_id": source_ids,
        "bp_rp": bp_rp,
        "abs_mag_G": abs_mag_G,
        "parallax": parallax,
        "phot_g_mean_mag": phot_g_mean_mag,
        "score": scores,
    }

    return all_residuals, metadata


def plot_umap(
    embedding,
    metadata,
    color_by="score",
    cmap=None,
    title=None,
    save_path=None,
    show=True,
):
    """
    Plot UMAP embedding colored by a chosen property.

    Parameters
    ----------
    embedding : (N, 2) array
    metadata : dict of arrays
    color_by : str
        Key in metadata to color by. Options: "bin", "bp_rp", "abs_mag_G",
        "parallax", "phot_g_mean_mag", "score".
    cmap : str or None
        Matplotlib colormap. If None, auto-selected based on color_by.
    title : str or None
    save_path : str or Path or None
    show : bool
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    c_values = metadata[color_by]

    if color_by == "bin":
        # Categorical coloring
        unique_bins = np.unique(c_values)
        cols = sns.hls_palette(n_colors=cfg.N_BINS, l=0.4, s=0.65)
        colors = [cols[int(b)] for b in c_values]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=80,
            alpha=1.0,
            c=colors,
            edgecolors="white",
            linewidths=0.5,
        )
        # Legend
        present_bins = sorted(set(int(b) for b in unique_bins))
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Bin {b}",
                markerfacecolor=cols[b],
                markersize=8,
            )
            for b in present_bins
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=2,
            fontsize=12,
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="k",
            borderaxespad=2.0,
        )
    else:
        # Continuous coloring
        if cmap is None:
            cmap = {
                "bp_rp": "RdYlBu_r",
                "abs_mag_G": "viridis",
                "parallax": "magma",
                "phot_g_mean_mag": "viridis",
                "score": "viridis_r",
            }.get(color_by, "viridis")

        label = {
            "bp_rp": "BP - RP",
            "abs_mag_G": "Absolute G Magnitude",
            "parallax": "Parallax [mas]",
            "phot_g_mean_mag": "G-band Apparent Magnitude",
            "score": "Outlier Score",
        }.get(color_by, color_by)

        # Sort so most interesting points (lowest score, extreme values) on top
        order = np.argsort(c_values)[::-1] if color_by == "score" else np.arange(len(c_values))

        scatter = ax.scatter(
            embedding[order, 0],
            embedding[order, 1],
            c=c_values[order],
            cmap=cmap,
            s=80,
            alpha=1.0,
            edgecolors="white",
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label=label)

    ax.grid(True, linestyle="-", alpha=0.5)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"UMAP Projection of Residuals (colored by {color_by})")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def find_residual_plot(source_id, i_bin, plots_dir, best_model_metric):
    """Find the residual PDF for a given source_id and bin."""
    bin_dir = Path(plots_dir) / best_model_metric / f"bin_{int(i_bin):02d}"
    matches = list(bin_dir.glob(f"*srcid_{int(source_id)}*.pdf"))
    if matches:
        return matches[0]
    return None


def plot_umap_interactive(
    embedding,
    metadata,
    color_by="score",
    cmap=None,
    plots_dir=PLOTS_DIR,
    best_model_metric=BEST_MODEL_METRIC,
):
    """
    Interactive UMAP plot. Click a point to open its residual PDF in Preview.

    Parameters
    ----------
    embedding : (N, 2) array
    metadata : dict of arrays
    color_by : str
        Key in metadata to color by.
    cmap : str or None
    plots_dir : Path
    best_model_metric : str
    """
    import subprocess

    fig, ax = plt.subplots(figsize=(10, 8))

    c_values = metadata[color_by]

    if color_by == "bin":
        unique_bins = np.unique(c_values)
        cols = sns.hls_palette(n_colors=cfg.N_BINS, l=0.4, s=0.65)
        colors = [cols[int(b)] for b in c_values]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=15,
            alpha=0.8,
            c=colors,
            edgecolors="none",
            picker=True,
        )
        present_bins = sorted(set(int(b) for b in unique_bins))
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o", color="w", label=f"Bin {b}",
                markerfacecolor=cols[b], markersize=6,
            )
            for b in present_bins
        ]
        ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=9)
    else:
        if cmap is None:
            cmap = {
                "bp_rp": "RdYlBu_r", "abs_mag_G": "viridis",
                "parallax": "magma", "phot_g_mean_mag": "viridis",
                "score": "viridis_r",
            }.get(color_by, "viridis")

        label = {
            "bp_rp": "BP - RP", "abs_mag_G": "Absolute G Magnitude",
            "parallax": "Parallax [mas]", "phot_g_mean_mag": "G-band Apparent Magnitude",
            "score": "Outlier Score",
        }.get(color_by, color_by)

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=c_values,
            cmap=cmap,
            s=15,
            alpha=0.8,
            edgecolors="none",
            picker=True,
        )
        plt.colorbar(scatter, ax=ax, label=label)

    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP (colored by {color_by}) — click a point to open residual plot")

    def on_pick(event):
        ind = event.ind[0]  # index of clicked point
        sid = int(metadata["source_id"][ind])
        i_bin = int(metadata["bin"][ind])
        score = metadata["score"][ind]
        print(f"Clicked: source_id={sid}, bin={i_bin}, score={score:.4f}")

        pdf_path = find_residual_plot(sid, i_bin, plots_dir, best_model_metric)
        if pdf_path is not None:
            print(f"  Opening {pdf_path}")
            subprocess.Popen(["open", str(pdf_path)])
        else:
            print(f"  No residual plot found for source_id={sid} in bin {i_bin}")

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()


if __name__ == "__main__":
    all_residuals, metadata = load_residuals_and_metadata()

    # UMAP
    reducer = umap.UMAP(random_state=1)
    embedding = reducer.fit_transform(all_residuals)
    print("UMAP embedding shape:", embedding.shape)

    # Interactive plot (click to open residual PDFs)
    plot_umap_interactive(embedding, metadata, color_by="score")

    # Static plots for saving
    for color_by in ["bin", "bp_rp", "abs_mag_G", "score"]:
        print(f"Plotting colored by {color_by}...")
        plot_umap(
            embedding,
            metadata,
            color_by=color_by,
            save_path=f"umap_residuals_{color_by}.png",
            show=False,
        )

    print("Done.")
