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
EMBEDDING_CACHE = Path("./umap_embedding.npz")

# Filter out sources that are outliers in only some of the bins they belong to
FILTER_INCONSISTENT = False


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


def filter_inconsistent_outliers(
    all_residuals,
    metadata,
    plots_dir=PLOTS_DIR,
    best_model_metric=BEST_MODEL_METRIC,
):
    """
    Remove sources that belong to >1 bin but are outliers in only some.

    Keeps: single-bin outliers + multi-bin outliers that are flagged in ALL bins.

    Returns filtered copies of all_residuals and metadata.
    """
    import pandas as pd

    plots_dir = Path(plots_dir) / best_model_metric

    # Build bin membership from all_source_ids.npy
    source_to_bins = {}
    for d in sorted(plots_dir.glob("bin_*")):
        sids_path = d / "all_source_ids.npy"
        if sids_path.exists():
            i_bin = int(d.name.split("_")[1])
            for sid in np.load(sids_path):
                source_to_bins.setdefault(int(sid), set()).add(i_bin)

    # Build outlier membership from outliers.csv
    source_to_outlier_bins = {}
    for d in sorted(plots_dir.glob("bin_*")):
        csv_path = d / "outliers.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                df = pd.read_csv(csv_path)
                i_bin = int(d.name.split("_")[1])
                for sid in df["source_id"]:
                    source_to_outlier_bins.setdefault(int(sid), set()).add(i_bin)
            except pd.errors.EmptyDataError:
                pass

    # Find sources to exclude: belong to >1 bin, outlier in < all bins
    exclude_sids = set()
    for sid, member_bins in source_to_bins.items():
        if len(member_bins) <= 1:
            continue
        outlier_bins = source_to_outlier_bins.get(sid, set())
        if len(outlier_bins) > 0 and not outlier_bins >= member_bins:
            exclude_sids.add(sid)

    # Filter
    keep = np.array([int(sid) not in exclude_sids for sid in metadata["source_id"]])
    n_removed = np.sum(~keep)
    print(
        f"Filtered {n_removed} residuals from {len(set(metadata['source_id'][~keep]))} "
        f"inconsistent sources (outlier in some but not all bins)"
    )

    filtered_residuals = all_residuals[keep]
    filtered_metadata = {k: v[keep] for k, v in metadata.items()}

    return filtered_residuals, filtered_metadata


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
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Bin {b}",
                markerfacecolor=cols[b],
                markersize=6,
            )
            for b in present_bins
        ]
        ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=9)
    else:
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


def plot_umap_dual_bin(
    embedding,
    metadata,
    dual_csv_path=PLOTS_DIR / BEST_MODEL_METRIC / "dual_bin_scores.csv",
    save_path="umap_dual_bin.pdf",
    show=False,
):
    """Plot only UMAP points whose source belongs to two overlapping bins,
    coloured by whether the source was flagged outlier in both bins or only one.

    Requires `dual_bin_scores.csv` from summarise_bins.dual_bin_score_analysis().
    Each source flagged in both bins contributes two points (one per bin); each
    source flagged in only one bin contributes a single point.
    """
    import pandas as pd

    df = pd.read_csv(dual_csv_path)
    flagged = df[df["outlier_A"] | df["outlier_B"]]
    in_both = set(
        flagged.loc[flagged["outlier_A"] & flagged["outlier_B"], "source_id"].astype(int)
    )
    in_one = set(
        flagged.loc[flagged["outlier_A"] ^ flagged["outlier_B"], "source_id"].astype(int)
    )

    keep_idx, labels = [], []
    for i, sid in enumerate(metadata["source_id"]):
        sid_int = int(sid)
        if sid_int in in_both:
            keep_idx.append(i)
            labels.append("both")
        elif sid_int in in_one:
            keep_idx.append(i)
            labels.append("one")

    keep_idx = np.array(keep_idx)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Draw connecting lines between the two UMAP entries of each "both" source.
    pairs_by_source = {}
    for i in keep_idx:
        pairs_by_source.setdefault(int(metadata["source_id"][i]), []).append(i)
    for sid, idxs in pairs_by_source.items():
        if sid in in_both and len(idxs) == 2:
            i1, i2 = idxs
            ax.plot(
                [embedding[i1, 0], embedding[i2, 0]],
                [embedding[i1, 1], embedding[i2, 1]],
                color="C0",
                alpha=0.4,
                lw=0.7,
                zorder=1,
            )

    for lbl, color, name in [
        ("both", "C0", "outlier in both bins"),
        ("one", "C1", "outlier in only one bin"),
    ]:
        m = labels == lbl
        ax.scatter(
            embedding[keep_idx[m], 0],
            embedding[keep_idx[m], 1],
            s=40,
            alpha=0.85,
            c=color,
            label=f"{name} ($N={int(m.sum())}$)",
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.set_title("Dual-bin outliers: agreement across overlapping bins")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    n_both = int((labels == "both").sum())
    n_one = int((labels == "one").sum())
    print(
        f"Plotted {len(keep_idx)} dual-bin outlier points "
        f"({n_both} both, {n_one} one-only) → {save_path}"
    )


def save_embedding(embedding, metadata, path=EMBEDDING_CACHE):
    """Save UMAP embedding and per-row metadata to a single .npz for reuse."""
    np.savez(path, embedding=embedding, **metadata)


def load_embedding(path=EMBEDDING_CACHE):
    """Load the cached (embedding, metadata) pair saved by save_embedding."""
    data = np.load(path, allow_pickle=False)
    embedding = data["embedding"]
    metadata = {k: data[k] for k in data.files if k != "embedding"}
    return embedding, metadata


def save_umap_table(
    embedding,
    metadata,
    plots_dir=PLOTS_DIR,
    best_model_metric=BEST_MODEL_METRIC,
    save_path="umap_residuals_table.csv",
    sort_by="umap_1",
    dual_csv_path=None,
):
    """Save a per-point lookup table for the UMAP, sorted by UMAP1 by default.

    Columns: umap_1, umap_2, source_id, bin, score, pdf_path, dual_bin,
    outlier_in_both, paired_pdf_path. The last three columns require
    `dual_bin_scores.csv` from `summarise_bins.dual_bin_score_analysis()`;
    if it isn't present they default to False / empty.
    """
    import csv

    if dual_csv_path is None:
        dual_csv_path = Path(plots_dir) / best_model_metric / "dual_bin_scores.csv"

    in_two_bins = set()
    outlier_in_both_set = set()
    other_bin = {}  # source_id -> {this_bin: that_bin} for outlier-in-both sources
    if Path(dual_csv_path).exists():
        with open(dual_csv_path) as f:
            for row in csv.DictReader(f):
                sid = int(row["source_id"])
                bA, bB = int(row["bin_A"]), int(row["bin_B"])
                in_two_bins.add(sid)
                if row["outlier_A"] == "True" and row["outlier_B"] == "True":
                    outlier_in_both_set.add(sid)
                    other_bin[sid] = {bA: bB, bB: bA}
    else:
        print(f"  Note: {dual_csv_path} not found — dual_bin / outlier_in_both / paired_pdf_path columns will be empty")

    rows = []
    for i in range(len(embedding)):
        sid = int(metadata["source_id"][i])
        bin_i = int(metadata["bin"][i])
        score = float(metadata["score"][i])
        pdf = find_residual_plot(sid, bin_i, plots_dir, best_model_metric)
        is_dual = sid in in_two_bins
        is_outlier_both = sid in outlier_in_both_set
        paired_pdf = ""
        if is_outlier_both and bin_i in other_bin.get(sid, {}):
            paired = find_residual_plot(
                sid, other_bin[sid][bin_i], plots_dir, best_model_metric
            )
            if paired:
                paired_pdf = str(paired)
        rows.append(
            {
                "umap_1": float(embedding[i, 0]),
                "umap_2": float(embedding[i, 1]),
                "source_id": sid,
                "bin": bin_i,
                "score": score,
                "pdf_path": str(pdf) if pdf else "",
                "dual_bin": is_dual,
                "outlier_in_both": is_outlier_both,
                "paired_pdf_path": paired_pdf,
            }
        )

    rows.sort(key=lambda r: r[sort_by])

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_missing = sum(1 for r in rows if not r["pdf_path"])
    print(f"Saved {len(rows)}-row UMAP lookup table to {save_path}")
    if n_missing:
        print(f"  WARNING: {n_missing} points have no matching PDF on disk")


if __name__ == "__main__":
    # Compute the embedding once and cache it; all downstream plots and the
    # lookup table share the same coordinates. Delete EMBEDDING_CACHE to refit.
    if EMBEDDING_CACHE.exists():
        print(f"Loading cached embedding from {EMBEDDING_CACHE}")
        embedding, metadata = load_embedding()
    else:
        all_residuals, metadata = load_residuals_and_metadata()
        if FILTER_INCONSISTENT:
            all_residuals, metadata = filter_inconsistent_outliers(all_residuals, metadata)
        # n_jobs=1 makes UMAP fully deterministic given random_state.
        reducer = umap.UMAP(random_state=1, n_jobs=1)
        embedding = reducer.fit_transform(all_residuals)
        save_embedding(embedding, metadata)
        print(f"Cached embedding to {EMBEDDING_CACHE}")
    print("UMAP embedding shape:", embedding.shape)

    # Per-point lookup table — pairs UMAP coords with source_id, bin, score, PDF.
    save_umap_table(embedding, metadata)

    # Dual-bin agreement plot (skipped if dual_bin_scores.csv hasn't been built).
    try:
        plot_umap_dual_bin(embedding, metadata)
    except FileNotFoundError as e:
        print(f"Skipping dual-bin plot: {e}")

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
