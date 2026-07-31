"""t-SNE of the outlier residual spectra, clustered into kinds of outlier.

The per-outlier figures from ``plot_final_outlier_spectra.py`` show one star at
a time, which is the wrong tool for the question "how many *sorts* of outlier
are there". This module embeds the ~1500 residual spectra in two dimensions,
finds the clumps, and draws what each clump looks like, so the categories can
be read off a handful of figures instead of a thousand.

PIPELINE
    residual = flux - reconstruction, for every spectrum below the threshold
      -> features (see below)
      -> PCA to --n-pca dimensions (denoises, and t-SNE on 50 dimensions is far
         faster and better behaved than on 2321)
      -> t-SNE to 2 dimensions
      -> HDBSCAN on the embedding, which finds clumps of any shape and is
         allowed to leave points unassigned rather than forcing every star into
         a cluster

FEATURES. ``--features normed`` (the default) divides each residual by its own
L2 norm, so the embedding groups by the *shape* of the residual rather than its
size. That is what makes the clusters mean "kind of outlier": raw amplitude
tracks brightness and how badly the star is fit, and left in, it dominates the
embedding and the clusters come out as brightness bins. ``raw`` keeps the
amplitude, and ``chi`` divides by the per-pixel uncertainty instead, which
weights the residual by how surprising it is rather than how large.

READ THE CLUSTERS, NOT THE COORDINATES. t-SNE preserves neighbourhoods, not
distances: the gap between two clumps and the size of a clump carry no
meaning, and neither does the choice of axes. The clustering is done on the
embedding because that is where the clumps are visible, so it inherits the same
caveat. Treat a cluster as a hypothesis to check against its spectra, which is
what the per-cluster figures are for.

OUTPUTS (in ./plots_<tag>_final/outlier_tsne by default)
    tsne_overview.pdf          -- the embedding coloured by cluster, score,
                                  colour and absolute magnitude
    pca_projections.pdf        -- PC1/PC2, PC1/PC3, PC2/PC3, by cluster
    hr_by_cluster.pdf          -- the HR diagram, outliers coloured by cluster
    cluster_median_residuals.pdf -- every cluster's median residual, stacked;
                                  the one figure to categorise from
    cluster_NN.pdf             -- per cluster: representative spectra, their
                                  residuals, and the cluster median
    outlier_clusters.csv       -- source_id, score, position, embedding, cluster
    cluster_summary.csv        -- a row per cluster: size, emission or
                                  absorption, strength, where its strongest
                                  feature sits, how much got downweighted
    outlier_tsne_embedding.npz -- cached embedding, so replotting is instant

USAGE
    uv run python tsne_outlier_residuals.py

    # a coarser or finer partition
    uv run python tsne_outlier_residuals.py --min-cluster-size 30

    # re-cluster and redraw from the cached embedding, without re-running t-SNE
    uv run python tsne_outlier_residuals.py --reuse-embedding --min-cluster-size 10

Needs the spectra and the converged state, like plot_final_outlier_spectra.py,
plus scikit-learn. No GPU.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
# _frame_hr is the existing HR-diagram framing, shared rather than restated so
# every HRD in the folder keeps the same limits and labels.
from plot_final_full_rvs import _frame_hr, check_text_rendering
from plot_final_outlier_spectra import DEFAULT_WEIGHTS, load_outlier_inputs
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

N_PCA = 50
PERPLEXITY = 30.0
MIN_CLUSTER_SIZE = 15
RANDOM_SEED = 42

# Representative members drawn per cluster, and how many go in the stacked
# example panel of the per-cluster figure.
N_EXAMPLES = 6

# HDBSCAN labels unassigned points -1; they are drawn but never get a colour of
# their own, since "did not join a clump" is not a kind of outlier.
NOISE_LABEL = -1
NOISE_COLOUR = "lightgrey"

# ============================================================================ #


def build_features(residual, ivar, kind):
    """The matrix t-SNE actually sees.

    ``residual`` and ``ivar`` are (N, M); returns (N, M). Masked pixels have
    zero inverse variance and a residual of zero by construction, so they
    contribute nothing to any of the three and need no special handling.
    """
    if kind == "raw":
        return residual
    if kind == "chi":
        return residual * np.sqrt(ivar)
    if kind == "normed":
        norm = np.linalg.norm(residual, axis=1, keepdims=True)
        # A residual of exactly zero cannot happen for a spectrum that scored
        # below the threshold, but dividing by it would poison the whole matrix.
        return residual / np.where(norm > 0, norm, 1.0)
    raise ValueError(f"Unknown feature kind {kind!r}")


def embed(features, n_pca=N_PCA, perplexity=PERPLEXITY, seed=RANDOM_SEED):
    """PCA then t-SNE.

    Returns ``(xy (N, 2), scores (N, n_pca), explained_variance_ratio)``.
    """
    n_pca = min(n_pca, *features.shape)
    t0 = time.time()
    pca = PCA(n_components=n_pca, random_state=seed)
    scores = pca.fit_transform(features)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {n_pca} components ({var:.1%} of the variance) in {time.time() - t0:.1f} s")

    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    xy = tsne.fit_transform(scores)
    print(f"  t-SNE in {time.time() - t0:.1f} s (KL {tsne.kl_divergence_:.3f})")
    return xy, scores, pca.explained_variance_ratio_


def cluster(xy, min_cluster_size=MIN_CLUSTER_SIZE):
    """HDBSCAN on the embedding. Returns integer labels, -1 for unassigned."""
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(xy)
    ids = sorted(set(labels) - {NOISE_LABEL})
    n_noise = int(np.sum(labels == NOISE_LABEL))
    print(f"  {len(ids)} clusters, {n_noise} of {len(labels)} points unassigned")
    for c in ids:
        print(f"    cluster {c:2d}: {int(np.sum(labels == c)):4d} spectra")
    return labels


def summarise_clusters(λ_grid, residual, robust, meta_df, labels, out):
    """A row per cluster describing what kind of residual it is.

    The point of the exercise is a taxonomy, and reading one off twenty-odd
    figures is slow. These are the numbers that separate the categories in
    practice: whether the median residual is dominated by emission or by
    absorption, how strong it is, where its strongest feature sits (the grid is
    the Ca II triplet region, so the wavelength usually names the line), and
    how much of the spectrum the model had to downweight.
    """
    rows = []
    for c in sorted(set(labels) - {NOISE_LABEL}):
        m = labels == c
        r = residual[m]
        med = np.median(r, axis=0)
        peak = int(np.argmax(np.abs(med)))
        sub = meta_df[m]
        rows.append(
            {
                "cluster": c,
                "n": int(m.sum()),
                "median_score": sub["score"].median(),
                "median_bp_rp": sub["bp_rp"].median(),
                "median_abs_mag_G": sub["abs_mag_G"].median(),
                "resid_rms": float(np.median(np.sqrt(np.mean(r**2, axis=1)))),
                "median_resid_max": float(med.max()),
                "median_resid_min": float(med.min()),
                # Positive residual means the data sits above the model, i.e.
                # emission the model does not have; negative means absorption
                # deeper than the model can make.
                "character": "emission" if med.max() > -med.min() else "absorption",
                "peak_wavelength_nm": float(λ_grid[peak]),
                "frac_pix_downweighted": float(np.median(np.mean(robust[m] < 0.5, axis=1))),
            }
        )
    df = pd.DataFrame(rows).sort_values("resid_rms", ascending=False)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return df


def cluster_colours(labels):
    """A colour per cluster id, plus grey for the unassigned."""
    ids = sorted(set(labels) - {NOISE_LABEL})
    cmap = plt.get_cmap("tab20" if len(ids) > 10 else "tab10")
    colours = {c: cmap(i % cmap.N) for i, c in enumerate(ids)}
    colours[NOISE_LABEL] = NOISE_COLOUR
    return colours


def _scatter_clusters(ax, xy, labels, colours, label_centroids=True, s=14):
    """The embedding, one colour per cluster, unassigned points underneath."""
    noise = labels == NOISE_LABEL
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=6, c=NOISE_COLOUR, marker=".", zorder=1)
    for c in sorted(set(labels) - {NOISE_LABEL}):
        m = labels == c
        ax.scatter(
            xy[m, 0], xy[m, 1], s=s, color=colours[c], marker="o", linewidths=0, zorder=3,
            label=f"{c} (n={int(m.sum())})",
        )
        if label_centroids:
            ax.annotate(
                str(c),
                np.median(xy[m], axis=0),
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            )


def _tsne_axes(ax):
    # No tick labels: t-SNE coordinates are not a quantity, and numbering them
    # invites reading distances off the axes that the embedding does not encode.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def plot_overview(xy, labels, colours, meta_df, out, label):
    """The embedding four times over: by cluster, score, colour, magnitude."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=140)
    axes = axes.ravel()

    _scatter_clusters(axes[0], xy, labels, colours)
    axes[0].set_title("HDBSCAN clusters")
    axes[0].legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)

    for ax, col, cmap, name in (
        (axes[1], "score", "viridis_r", "Outlier score (lower = more anomalous)"),
        (axes[2], "bp_rp", "coolwarm", "Color (BP - RP)"),
        (axes[3], "abs_mag_G", "magma", "G-Band Absolute Magnitude"),
    ):
        v = meta_df[col].to_numpy()
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap, s=14, marker="o", linewidths=0)
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_title(f"Coloured by {name.split(' (')[0].lower()}")

    for ax in axes:
        _tsne_axes(ax)
    _suptitle(fig, f"{label}: t-SNE of outlier residual spectra")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_pca_projections(scores, labels, colours, var, out, label):
    """The first three principal components against each other, by cluster.

    A linear counterpart to the t-SNE panel: here the axes *are* meaningful and
    distances mean something, so a structure that survives in both is real
    rather than an artefact of the embedding.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=140)
    for ax, (i, j) in zip(axes, pairs):
        noise = labels == NOISE_LABEL
        if noise.any():
            ax.scatter(scores[noise, i], scores[noise, j], s=6, c=NOISE_COLOUR, marker=".")
        for c in sorted(set(labels) - {NOISE_LABEL}):
            m = labels == c
            ax.scatter(
                scores[m, i], scores[m, j], s=12, color=colours[c], marker="o", linewidths=0
            )
        ax.set_xlabel(f"PC{i + 1} ({var[i]:.1%})")
        ax.set_ylabel(f"PC{j + 1} ({var[j]:.1%})")
    _suptitle(fig, f"{label}: residual PCA, coloured by t-SNE cluster")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_hr_by_cluster(meta_df, labels, colours, bp_rp_all, abs_mag_all, out, label):
    """Where each cluster sits on the HR diagram, over the full sample in grey."""
    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    ax.scatter(
        bp_rp_all, abs_mag_all, s=0.5, alpha=0.1, c="grey", zorder=0, marker=".", rasterized=True
    )
    x, y = meta_df["bp_rp"].to_numpy(), meta_df["abs_mag_G"].to_numpy()
    noise = labels == NOISE_LABEL
    if noise.any():
        ax.scatter(x[noise], y[noise], s=12, c=NOISE_COLOUR, marker=".", zorder=3)
    handles = []
    for c in sorted(set(labels) - {NOISE_LABEL}):
        m = labels == c
        ax.scatter(
            x[m], y[m], s=45, color=colours[c], marker="o",
            edgecolors="k", linewidths=0.4, zorder=5,
        )
        handles.append(
            Line2D([], [], ls="", marker="o", color=colours[c], label=f"{c} (n={int(m.sum())})")
        )
    ax.legend(handles=handles, fontsize=9, ncol=2, loc="best", framealpha=0.9)
    _frame_hr(ax)
    ax.set_title(f"{label}: outlier clusters on the HR diagram")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cluster_medians(λ_grid, residual, labels, colours, out, label):
    """Every cluster's median residual, stacked -- the categorisation sheet.

    One row per cluster, on a shared wavelength axis and a shared vertical
    scale, so the shapes are directly comparable: this is the figure to read
    the taxonomy off, with the per-cluster figures as the follow-up.
    """
    ids = sorted(set(labels) - {NOISE_LABEL})
    if not ids:
        print("note: no clusters, so no median-residual sheet")
        return
    fig, axes = plt.subplots(
        len(ids), 1, figsize=(14, 1.9 * len(ids) + 1.5), dpi=140, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, ids):
        m = labels == c
        r = residual[m]
        lo, med, hi = np.percentile(r, [16, 50, 84], axis=0)
        ax.fill_between(λ_grid, lo, hi, color=colours[c], alpha=0.3, lw=0)
        ax.plot(λ_grid, med, color=colours[c], lw=1.3)
        ax.axhline(0, c="k", lw=0.6, alpha=0.5)
        ax.set_ylabel(f"{c}\n(n={int(m.sum())})", rotation=0, ha="right", va="center")
        ax.set_xlim(λ_grid[0], λ_grid[-1])
    # A shared scale set by the bulk, so one violent cluster does not flatten
    # every other row into a straight line.
    span = np.percentile(np.abs(residual), 99.5)
    axes[0].set_ylim(-span, span)
    axes[-1].set_xlabel("Wavelength [nm]")
    _suptitle(fig, f"{label}: median residual per cluster (16-84 percentile band)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cluster_detail(λ_grid, Y, recon, residual, meta_df, labels, c, colour, out, label):
    """One cluster in detail: representative spectra, residuals, and the median.

    Members are ranked by distance to the cluster's median residual, so the
    examples drawn are the ones that look most like the cluster rather than an
    arbitrary handful.
    """
    m = np.flatnonzero(labels == c)
    med = np.median(residual[m], axis=0)
    order = m[np.argsort(np.linalg.norm(residual[m] - med, axis=1))]
    picks = order[:N_EXAMPLES]

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 14), dpi=140, sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.6]}
    )
    # Offsets from the data itself, so the stack neither overlaps nor spreads
    # into a band of whitespace when the residuals are small.
    step_flux = 1.35 * np.nanmax(np.ptp(Y[picks], axis=1)) if len(picks) else 1.0
    step_resid = 1.4 * np.percentile(np.abs(residual[picks]), 99) if len(picks) else 1.0

    for i, idx in enumerate(picks):
        off = -i * step_flux
        axes[0].plot(λ_grid, Y[idx] + off, c="k", lw=1.0, zorder=3)
        axes[0].plot(λ_grid, recon[idx] + off, c="tab:red", lw=1.0, ls=(0, (5, 1)), zorder=4)
        axes[1].plot(λ_grid, residual[idx] - i * step_resid, c=colour, lw=1.0)
        axes[1].axhline(-i * step_resid, c="k", lw=0.5, alpha=0.4)
        row = meta_df.iloc[idx]
        # Above the trace's own maximum, not at a fixed offset from the
        # baseline: the spectra are continuum-normalised but not identically
        # scaled, and a fixed offset lands the label on top of the data.
        axes[0].annotate(
            f"Gaia DR3 {int(row['source_id'])}  score {row['score']:.3f}",
            (λ_grid[0], off + np.nanmax(Y[idx]) + 0.04 * step_flux),
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )

    axes[0].set_ylabel("Flux (offset)")
    axes[0].legend(
        handles=[
            Line2D([], [], c="k", lw=1.2, label="Data"),
            Line2D([], [], c="tab:red", lw=1.2, ls=(0, (5, 1)), label="Model"),
        ],
        loc="lower right", fontsize=9, framealpha=0.9,
    )
    axes[1].set_ylabel("Residual (offset)")

    lo, mid, hi = np.percentile(residual[labels == c], [16, 50, 84], axis=0)
    axes[2].fill_between(λ_grid, lo, hi, color=colour, alpha=0.3, lw=0)
    axes[2].plot(λ_grid, mid, color=colour, lw=1.4)
    axes[2].axhline(0, c="k", lw=0.6, alpha=0.5)
    axes[2].set_ylabel("Cluster median\nresidual")
    axes[2].set_xlabel("Wavelength [nm]")

    for ax in axes:
        ax.set_xlim(λ_grid[0], λ_grid[-1])

    sub = meta_df[labels == c]
    _suptitle(
        fig,
        f"{label}: cluster {c}, {len(m)} spectra, median score {sub['score'].median():.3f}",
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _suptitle(fig, text, y=1.01):
    """Matches plot_final_full_rvs._suptitle: mathtext cannot do \\textsf."""
    if plt.rcParams.get("text.usetex", False):
        fig.suptitle(rf"$\textsf{{\textbf{{{text}}}}}$", fontsize="20", c="dimgrey", y=y)
    else:
        fig.suptitle(text, fontsize="20", c="dimgrey", y=y, fontweight="bold")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None, help="default: the npz's threshold")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--features",
        default="normed",
        choices=("normed", "raw", "chi"),
        help="what t-SNE sees; see the module docstring (default: %(default)s)",
    )
    p.add_argument("--n-pca", type=int, default=N_PCA)
    p.add_argument("--perplexity", type=float, default=PERPLEXITY)
    p.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--limit", type=int, default=None, help="only the N worst outliers")
    p.add_argument(
        "--reuse-embedding",
        action="store_true",
        help="load the cached embedding instead of re-running PCA and t-SNE",
    )
    p.add_argument("--no-latex", action="store_true")
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False
    if not args.weights.exists():
        raise SystemExit(f"{args.weights} does not exist -- run fit_final_full_rvs.py first.")

    d = np.load(args.weights)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    label = "Full Main Sequence" if tag == "full_ms" else "Full RVS Sample"
    state_file = args.state or (
        args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    if not state_file.exists():
        raise SystemExit(f"{state_file} does not exist -- pass --state explicitly.")
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/outlier_tsne")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "outlier_tsne_embedding.npz"

    check_text_rendering()

    # The same loader the per-outlier figures use, so "the residuals" here are
    # exactly the ones drawn there.
    λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
        args.weights, state_file, threshold, args.limit, args.sample
    )
    residual = Y - recon
    meta_df = pd.DataFrame(meta)

    if args.reuse_embedding and cache.exists():
        c = np.load(cache)
        xy, scores, var = c["xy"], c["pca_scores"], c["explained_variance_ratio"]
        if len(xy) != len(residual):
            raise SystemExit(
                f"{cache} holds {len(xy)} points but this run has {len(residual)} -- "
                "the threshold or --limit changed; drop --reuse-embedding."
            )
        print(f"Reusing the embedding in {cache}")
    else:
        print(f"Embedding {len(residual)} residuals ({args.features} features)...", flush=True)
        features = build_features(residual, W, args.features)
        xy, scores, var = embed(features, args.n_pca, args.perplexity, args.seed)
        np.savez(
            cache,
            xy=xy,
            pca_scores=scores,
            explained_variance_ratio=var,
            source_id=meta_df["source_id"].to_numpy(),
            features=args.features,
        )
        print(f"Wrote {cache}")

    print(f"Clustering (min_cluster_size={args.min_cluster_size})...", flush=True)
    labels = cluster(xy, args.min_cluster_size)
    colours = cluster_colours(labels)

    meta_df["tsne_1"], meta_df["tsne_2"] = xy[:, 0], xy[:, 1]
    meta_df["cluster"] = labels
    csv = out_dir / "outlier_clusters.csv"
    meta_df.to_csv(csv, index=False)
    print(f"Wrote {csv}")
    summarise_clusters(
        λ_grid, residual, robust, meta_df, labels, out_dir / "cluster_summary.csv"
    )

    plot_overview(xy, labels, colours, meta_df, out_dir / "tsne_overview.pdf", label)
    plot_pca_projections(scores, labels, colours, var, out_dir / "pca_projections.pdf", label)
    plot_hr_by_cluster(
        meta_df, labels, colours, d["bp_rp"], d["abs_mag_G"],
        out_dir / "hr_by_cluster.pdf", label,
    )
    plot_cluster_medians(
        λ_grid, residual, labels, colours, out_dir / "cluster_median_residuals.pdf", label
    )
    for c in sorted(set(labels) - {NOISE_LABEL}):
        plot_cluster_detail(
            λ_grid, Y, recon, residual, meta_df, labels, c, colours[c],
            out_dir / f"cluster_{c:02d}.pdf", label,
        )

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
