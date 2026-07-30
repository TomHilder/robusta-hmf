"""Regenerate the toy-example paper figures individually.

Each of the paper figures produced inline by ``analyse_toy.py`` is
reproduced here as a standalone function that performs its own minimal setup,
builds the figure, and writes it into the repo's real paper figures directory.

Usage
-----
    uv run python make_paper_figs.py <name>

where ``<name>`` is one of:
    toy_weights           -> weights_per_object_clean_vs_outlier.pdf
    toy_residuals         -> absorption_line_residuals.pdf
    cv                    -> test_set_score_heatmap.pdf
    toy_spectra           -> normal_vs_outlier_spectra_reconstructions.pdf
    toy_diversity         -> toy_dataset_diversity.pdf
    eigenspectra          -> toy_eigenspectra_comparison.pdf
    explained_variance    -> toy_explained_variance.pdf
    coefficients          -> toy_coefficient_distributions.pdf
    all                   -> all of the above

This script is additive: it does not import or modify analyse_toy.py.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from r_pca import RobustPCA
from run_toy_gen_and_fits import M_PIXELS, N_SPECTRA, N_TRAIN, Q_VALS, RANKS
from sklearn.metrics import f1_score
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState, load_state_from_npz

plt.style.use("mpl_drip.custom")

# Output directory: the repo's real paper figs dir, defined relative to this script.
PAPER_FIGS = Path(__file__).resolve().parent.parent.parent / "paper" / "figs"

# Directory layout (mirrors analyse_toy.py)
SCRIPT_DIR = Path(__file__).resolve().parent
results_dir = SCRIPT_DIR / "toy_model_results"

# Hyperparameters of the model used for the per-spectrum / per-pixel figures.
PLOT_Q = 5
PLOT_K = 5

# Mean used for imputing NaNs before fitting PCA / RPCA (matches analyse_toy.py).
_SPECTRA_MEAN = 0.0


@dataclass(frozen=True)
class Results:
    N: int
    M: int
    K: int
    Q: float
    state: RHMFState


def _load_results():
    """Load the toy data and all converged model states.

    Returns the loaded ``data`` npz, the list of ``Results`` (one per (Q, K)
    grid point), and the matching list of ``Robusta`` objects with their states
    overridden by the converged states.
    """
    data_file = results_dir / f"data_N{N_SPECTRA}_M{M_PIXELS}.npz"
    data = np.load(data_file)

    Q_grid, Rank_grid = np.meshgrid(Q_VALS, RANKS)
    Q_vals = Q_grid.flatten()
    K_vals = Rank_grid.flatten()

    results = []
    for Q, rank in tqdm(zip(Q_vals, K_vals), total=len(Q_vals), desc="Loading states"):
        state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_N{N_SPECTRA}_M{M_PIXELS}.npz"
        state = load_state_from_npz(state_file)
        results.append(Results(N=N_SPECTRA, M=M_PIXELS, K=rank, Q=Q, state=state))

    rhmf_objs = [Robusta(rank=r.K, robust_scale=r.Q) for r in results]
    for obj, res in zip(rhmf_objs, results):
        obj._state = res.state

    return data, results, rhmf_objs


def _all_data_arrays(data):
    """Extract the full (train + test) spectra arrays used for inference/plotting."""
    all_noisy_spectra = data["noisy_spectra"]  # with NaN for plotting
    all_spectra_for_fit = np.nan_to_num(data["noisy_spectra"], nan=_SPECTRA_MEAN)
    all_ivar = data["ivar"]
    grid = data["grid"]
    return all_noisy_spectra, all_spectra_for_fit, all_ivar, grid


def _load_or_compute_rpca(all_spectra_for_fit, max_iter=500, tol=1e-4):
    """Right singular vectors of the Robust-PCA low-rank part, cached to disk.

    RPCA is deterministic and slow (~minutes), and only its basis is needed for
    the comparison fit, so we cache Vh of the low-rank component keyed by data
    shape and fit params. Delete the cache file (or change params) to recompute.
    """
    n, m = all_spectra_for_fit.shape
    cache_file = results_dir / f"rpca_cache_N{n}_M{m}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        if int(cached["max_iter"]) == max_iter and float(cached["tol"]) == tol:
            print(f"Loaded cached Robust PCA basis from {cache_file.name}")
            return cached["Vh_rpca"]
    print("Running Robust PCA (this may take a while)...")
    rpca = RobustPCA(all_spectra_for_fit)
    rpca_L, _ = rpca.fit(max_iter=max_iter, iter_print=1, tol=tol)
    _, _, Vh_rpca = np.linalg.svd(rpca_L, full_matrices=False)
    np.savez(cache_file, Vh_rpca=Vh_rpca, max_iter=max_iter, tol=tol)
    print(f"Robust PCA complete; cached to {cache_file.name}")
    return Vh_rpca


def _plot_model_and_state(data, results, rhmf_objs, all_spectra_for_fit, all_ivar):
    """Select the (PLOT_Q, PLOT_K) model and infer its state on all data."""
    result_ind = np.where(
        (np.array([r.Q for r in results]) == PLOT_Q) & (np.array([r.K for r in results]) == PLOT_K)
    )[0][0]
    plot_rhmf: Robusta = rhmf_objs[result_ind]

    print("Inferring on all data for visualizations...")
    all_state, _ = plot_rhmf.infer(
        Y_infer=all_spectra_for_fit,
        W_infer=all_ivar,
        max_iter=1000,
        conv_tol=1e-2,
        conv_check_cadence=1,
    )
    return plot_rhmf, all_state


def split_by_near_uniform(x, *, factor=3.0, step=None, return_breaks=False):
    """Split 1D array x into subarrays where spacing is ~uniform.

    Parameters
    ----------
    x : array_like
        Sorted 1D array.
    factor : float, optional
        Any gap > factor * step starts a new chunk. Default 3.0.
    step : float or None, optional
        Expected step size. If None, uses median(diff(x)).
    return_breaks : bool, optional
        If True, also return the break indices (start positions of new chunks).
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size <= 1:
        return ([x.copy()], np.array([], int)) if return_breaks else [x.copy()]

    d = np.diff(x)
    if step is None:
        step = np.median(d)  # robust against one big gap
    breaks = np.where(d > factor * step)[0] + 1
    chunks = np.split(x, breaks)
    return (chunks, breaks) if return_breaks else chunks


def fig_toy_weights():
    """Figure: weights_per_object_clean_vs_outlier.pdf

    Histogram of median robust weight per spectrum, split into normal vs outlier
    spectra (single-panel version).
    """
    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)
    os_mask = data["os_mask"]

    plot_rhmf, all_state = _plot_model_and_state(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )

    # Per-pixel robust weights on all data, then per-object median.
    weights = plot_rhmf.robust_weights(all_spectra_for_fit, all_ivar, state=all_state)
    per_object_weights = np.median(weights, axis=1)  # Changed from mean to median

    outlier_spectra_mask = os_mask.any(axis=1)
    clean_spectra_mask = ~outlier_spectra_mask
    per_object_weights_clean = per_object_weights[clean_spectra_mask]
    per_object_weights_outlier = per_object_weights[outlier_spectra_mask]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
    bins = np.linspace(0, 1, 51)
    ax.hist(
        per_object_weights_clean,
        bins=bins,
        alpha=0.7,
        color="C0",
        label="Normal Spectra",
        # hatch="XX",
        # edgecolor="blue",
        # lw=0,
    )
    ax.hist(
        per_object_weights_outlier,
        bins=bins,
        alpha=0.7,
        color="C1",
        label="Outlier Spectra",
        hatch="oo",
        edgecolor="#8B4513",
        lw=0,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Median Robust Weight per Spectrum")  # Changed from 'Mean' to 'Median'
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", borderaxespad=1)
    fig.suptitle(
        r"$\textsf{\textbf{Toy Example: Object Weights}}$",
        fontsize="24",
        c="dimgrey",
        y=0.945,
    )
    plt.tight_layout()
    out = PAPER_FIGS / "weights_per_object_clean_vs_outlier.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_toy_residuals():
    """Figure: absorption_line_residuals.pdf

    Three-panel plot for one spectrum that contains injected outlier absorption
    lines: data + RHMF/RPCA/PCA fits (top), residuals (middle), robust weights
    (bottom), with shaded outlier regions.
    """
    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)
    op_mask = data["op_mask"]
    oc_mask = data["oc_mask"]
    al_mask = data["al_mask"]

    plot_rhmf, all_state = _plot_model_and_state(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )

    weights = plot_rhmf.robust_weights(all_spectra_for_fit, all_ivar, state=all_state)

    # PCA basis (for comparison fit)
    U, S, Vh = np.linalg.svd(all_spectra_for_fit, full_matrices=False)
    V = Vh.T
    pca_basis = V[:, :PLOT_K]

    # Robust PCA basis (for comparison fit) — cached to disk (RPCA is slow).
    Vh_rpca = _load_or_compute_rpca(all_spectra_for_fit)
    rpca_basis = Vh_rpca.T[:, :PLOT_K]

    # Spectra with injected absorption lines
    al_spectra_idx = np.where(np.any(al_mask, axis=1))[0]
    al_mask_al_spectra = al_mask[al_spectra_idx, :]

    i_al_spec = 0
    al_line_chunks = split_by_near_uniform(grid[al_mask_al_spectra[i_al_spec]], factor=2.0)

    op_mask_al_spectra = op_mask[al_spectra_idx, :]
    oc_mask_al_spectra = oc_mask[al_spectra_idx, :]

    reconstructions_al = all_state.A[al_spectra_idx, :] @ all_state.G.T
    residuals = all_noisy_spectra[al_spectra_idx, :] - reconstructions_al
    robust_weights = weights[al_spectra_idx, :]

    spec_i = np.nan_to_num(all_noisy_spectra[al_spectra_idx, :][i_al_spec], nan=_SPECTRA_MEAN)
    pca_coeffs = spec_i @ pca_basis
    pca_recon = pca_coeffs @ pca_basis.T
    rpca_coeffs = spec_i @ rpca_basis
    rpca_recon = rpca_coeffs @ rpca_basis.T

    fig, ax = plt.subplots(
        3, 1, figsize=(12, 8), dpi=100, sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    ax[0].plot(
        grid / 10,
        all_noisy_spectra[al_spectra_idx, :][i_al_spec],
        c="k",
        lw=3.0,
        zorder=7,
        label="Toy Data",
    )
    ax[0].plot(
        grid / 10,
        reconstructions_al[i_al_spec],
        c="tab:green",
        lw=2.0,
        zorder=10,
        ls=(0, (5, 1)),
        label="RHMF Fit",
    )
    ax[0].plot(
        grid / 10,
        rpca_recon,
        c="tab:blue",
        lw=2.0,
        zorder=9,
        ls=(0, (1, 1)),
        label="RPCA Fit",
    )
    ax[0].plot(
        grid / 10,
        pca_recon,
        c="tab:red",
        lw=2.0,
        zorder=8,
        ls=(0, (3, 1, 1, 1, 1, 1)),
        label="PCA Fit",
    )
    ax[1].plot(grid / 10, residuals[i_al_spec, :], color="k", alpha=1.0, lw=3.0)
    ax[2].plot(grid / 10, robust_weights[i_al_spec], color="k", alpha=1.0, lw=3.0)
    alpha = 0.5
    for j in range(3):
        ax[j].set_xlim(grid.min() / 10 - 2, grid.max() / 10 + 2)
        if j == 0:
            ymin = np.nanmin(all_noisy_spectra[al_spectra_idx, :][i_al_spec]) - 0.1
            ymax = np.nanmax(all_noisy_spectra[al_spectra_idx, :][i_al_spec]) + 0.1
            ax[j].set_ylabel("Flux")
        elif j == 2:
            ymin = -0.05
            ymax = 1.05
        else:
            ymin = -0.6
            ymax = 0.6
        ax[j].set_ylim(ymin, ymax)
        for i in range(3):
            ax[j].fill_betweenx(
                y=[ymin, ymax],
                x1=[al_line_chunks[i].min() / 10 - 0.3],
                x2=[al_line_chunks[i].max() / 10 + 0.3],
                color="C0",
                alpha=alpha,
                zorder=-2,
                linewidth=0,
                label="Outlier Lines" if i == 0 and j == 1 else None,
            )
        ax[j].vlines(
            grid[op_mask_al_spectra[i_al_spec]] / 10,
            ymin=ymin,
            ymax=ymax,
            color="C1",
            alpha=alpha,
            lw=5,
            zorder=-2,
            label="Outlier Pixels" if j == 1 else None,
        )
        ax[j].vlines(
            grid[oc_mask_al_spectra[i_al_spec]] / 10,
            ymin=ymin,
            ymax=ymax,
            color="C2",
            alpha=alpha,
            lw=2,
            zorder=-2,
            label="Outlier Column" if j == 1 else None,
        )
    ax[-1].set_xlabel("Wavelength [nm]")
    ax[1].set_ylabel("Residual \nFlux")
    ax[2].set_ylabel("Robust \nWeight")
    ax[0].legend(loc=(0.3, 0.6))
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),  # figure coordinates
        borderaxespad=0,
    )
    fig.align_ylabels()
    fig.suptitle(
        r"$\textsf{\textbf{Toy Example: Spectrum Containing Outliers}}$",
        fontsize="24",
        c="dimgrey",
        y=1.015,
    )
    plt.tight_layout()
    out = PAPER_FIGS / "absorption_line_residuals.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_cv():
    """Figure: test_set_score_heatmap.pdf

    Three-panel hyperparameter heatmap: cross-validation KL score (top),
    per-pixel F1 (bottom left), per-object F1 (bottom right), over the (Q, K)
    grid, computed by inferring each model on the held-out test set.
    """
    data, results, rhmf_objs = _load_results()

    test_spectra = data["test_spectra"]
    test_ivar = data["test_ivar"]

    test_states = []
    for rhmf in tqdm(rhmf_objs, desc="Inferring on test set"):
        test_set_state, _ = rhmf.infer(
            Y_infer=test_spectra,
            W_infer=test_ivar,
            max_iter=1000,
            conv_tol=1e-2,
            conv_check_cadence=1,
        )
        test_states.append(test_set_state)

    # Ground-truth test set outlier masks
    test_outlier_mask = data["total_outlier_mask"][N_TRAIN:]  # per-pixel
    test_os_mask = data["os_mask"][N_TRAIN:]
    test_outlier_object = test_os_mask.any(axis=1)

    scores = []
    f1_pixel_scores = []
    f1_object_scores = []
    for rhmf, state in zip(rhmf_objs, test_states):
        residuals = rhmf.residuals(Y=test_spectra, state=state)
        robust_weights = rhmf.robust_weights(test_spectra, test_ivar, state=state)

        # KL divergence from N(0,1) via closed form for fitted Gaussian to z-scores.
        z_scores = residuals * np.sqrt(test_ivar) * np.sqrt(robust_weights)
        mu_z = np.mean(z_scores)
        sigma_z = np.std(z_scores)
        kl_score = -np.log(sigma_z) + (sigma_z**2 + mu_z**2) / 2.0 - 0.5
        scores.append(kl_score)

        # Per-pixel F1 (weight < 0.5 => predicted outlier)
        y_true_pixel = test_outlier_mask.flatten().astype(int)
        y_pred_pixel = (robust_weights.flatten() < 0.5).astype(int)
        f1_pixel = f1_score(y_true_pixel, y_pred_pixel)
        f1_pixel_scores.append(f1_pixel)

        # Per-object F1 (median weight per spectrum, threshold 0.9)
        median_weights = np.median(robust_weights, axis=1)
        y_true_object = test_outlier_object.astype(int)
        y_pred_object = (median_weights < 0.9).astype(int)
        f1_object = f1_score(y_true_object, y_pred_object)
        f1_object_scores.append(f1_object)

    scores = np.array(scores).reshape(len(RANKS), len(Q_VALS))
    f1_pixel_scores = np.array(f1_pixel_scores).reshape(len(RANKS), len(Q_VALS))
    f1_object_scores = np.array(f1_object_scores).reshape(len(RANKS), len(Q_VALS))

    fig = plt.figure(figsize=(10, 8), dpi=100)

    gs_top = fig.add_gridspec(
        1, 2, width_ratios=[1, 0.04], left=0.28, right=0.72, top=0.9, bottom=0.54
    )
    gs_bot = fig.add_gridspec(
        1, 3, width_ratios=[1, 1, 0.04], left=0.08, right=0.92, top=0.44, bottom=0.06, wspace=0.15
    )

    ax_cv = fig.add_subplot(gs_top[0, 0])
    cax_cv = fig.add_subplot(gs_top[0, 1])
    ax_f1_pix = fig.add_subplot(gs_bot[0, 0])
    ax_f1_obj = fig.add_subplot(gs_bot[0, 1])
    cax_f1 = fig.add_subplot(gs_bot[0, 2])

    q_labels = [str(q) for q in Q_VALS]
    rank_labels = [str(r) for r in RANKS]

    text_bbox = dict(
        boxstyle="square",
        facecolor="white",
        alpha=0.7,
        edgecolor="none",
    )
    text_loc = (0.08, 0.85)  # relative to axes

    im0 = ax_cv.imshow(
        np.log10(scores),
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    ax_cv.set_xticks(np.arange(len(Q_VALS)), labels=q_labels)
    ax_cv.set_yticks(np.arange(len(RANKS)), labels=rank_labels)
    ax_cv.set_xlabel("Robust Scale Q")
    ax_cv.set_ylabel("Rank K")
    ax_cv.text(
        *text_loc,
        r"Cross-Validation",
        transform=ax_cv.transAxes,
        ha="left",
        va="bottom",
        bbox=text_bbox,
    )
    fig.colorbar(
        im0, cax=cax_cv, label=r"$\log_{10}$ KL$(p_z \| \mathcal{N}(0,1))$" + "\n(Lower is Better)"
    )

    bottom_cmap = "magma_r"

    im1 = ax_f1_pix.imshow(
        f1_pixel_scores,
        origin="lower",
        cmap=bottom_cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax_f1_pix.set_xticks(np.arange(len(Q_VALS)), labels=q_labels)
    ax_f1_pix.set_yticks(np.arange(len(RANKS)), labels=rank_labels)
    ax_f1_pix.set_xlabel("Robust Scale Q")
    ax_f1_pix.set_ylabel("Rank K")
    ax_f1_pix.text(
        *text_loc,
        r"Per-Pixel Identification",
        transform=ax_f1_pix.transAxes,
        ha="left",
        va="bottom",
        bbox=text_bbox,
    )

    im2 = ax_f1_obj.imshow(
        f1_object_scores,
        origin="lower",
        cmap=bottom_cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax_f1_obj.set_xticks(np.arange(len(Q_VALS)), labels=q_labels)
    ax_f1_obj.set_yticklabels([])
    ax_f1_obj.set_xlabel("Robust Scale Q")
    ax_f1_obj.text(
        *text_loc,
        r"Per-Object Identification",
        transform=ax_f1_obj.transAxes,
        ha="left",
        va="bottom",
        bbox=text_bbox,
    )

    fig.colorbar(im1, cax=cax_f1, label="F1 Score \n(Higher is Better)")

    fig.suptitle(
        r"$\textsf{\textbf{Toy Example: Hyperparameters}}$",
        fontsize="24",
        c="dimgrey",
        y=0.98,
    )
    out = PAPER_FIGS / "test_set_score_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_toy_spectra():
    """Figure: normal_vs_outlier_spectra_reconstructions.pdf

    Stacked plot of a few normal spectra (top) and a few outlier spectra
    (bottom), each overlaid with its RHMF reconstruction.
    """
    rng = default_rng(99202012345)

    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)
    os_mask = data["os_mask"]

    plot_rhmf, all_state = _plot_model_and_state(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )

    # Reproduce analyse_toy.py's global-RNG state exactly: before this figure's
    # selection, the rng is advanced once by a size-5 choice (analyse_toy.py
    # line ~116, used for a different plot). Replay it so the same example
    # spectra are chosen as in the committed paper figure.
    rng.choice(all_noisy_spectra.shape[0], size=5, replace=False)

    N_CLEAN_PLOT = 3
    N_OUTLIER_PLOT = 3

    is_outlier_spectrum = os_mask.any(axis=1)
    noise_level = np.nanstd(all_noisy_spectra, axis=1)
    noise_threshold = np.nanpercentile(noise_level, 90)
    low_noise = noise_level < noise_threshold

    clean_indices = np.where(~is_outlier_spectrum & low_noise)[0]
    outlier_indices = np.where(is_outlier_spectrum & low_noise)[0]

    clean_plot_idx = rng.choice(clean_indices, size=N_CLEAN_PLOT, replace=False)
    outlier_plot_idx = rng.choice(outlier_indices, size=N_OUTLIER_PLOT, replace=False)

    combined_idx = np.concatenate([clean_plot_idx, outlier_plot_idx])
    combined_predictions = plot_rhmf.synthesize(indices=combined_idx, state=all_state)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    offset = 0.0
    offset_step = 1.0

    data_lw = 2.8
    fit_lw = 1.5

    # Outlier spectra on bottom
    for i_off, idx in enumerate(outlier_plot_idx):
        pred_i = N_CLEAN_PLOT + i_off
        ax.plot(
            grid / 10,
            all_noisy_spectra[idx, :] + offset,
            color="C1",
            alpha=1.0,
            lw=data_lw,
            label="Outlier Spectra" if i_off == 0 else None,
        )
        ax.plot(
            grid / 10,
            combined_predictions[pred_i, :] + offset,
            color="k",
            alpha=1,
            lw=fit_lw,
            ls=(0, (5, 1)),
            label="RHMF Fits" if i_off == 0 else None,
        )
        offset += offset_step

    offset += offset_step * 0.0

    # Clean spectra on top
    for i_off, idx in enumerate(clean_plot_idx):
        pred_i = i_off
        ax.plot(
            grid / 10,
            all_noisy_spectra[idx, :] + offset,
            color="C0",
            alpha=1.0,
            lw=data_lw,
            label="Normal Spectra" if i_off == 0 else None,
        )
        ax.plot(
            grid / 10,
            combined_predictions[pred_i, :] + offset,
            color="k",
            alpha=1,
            ls=(0, (5, 1)),
            lw=fit_lw,
        )
        offset += offset_step

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Flux + offset")
    ax.set_xlim(grid.min() / 10 - 4, grid.max() / 10 + 4)
    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index("Normal Spectra"),
        labels.index("Outlier Spectra"),
        labels.index("RHMF Fits"),
    ]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
    )
    fig.suptitle(
        r"$\textsf{\textbf{Toy Example: Fits to Normal and Outlier Spectra}}$",
        fontsize="24",
        c="dimgrey",
        y=0.97,
    )
    out = PAPER_FIGS / "normal_vs_outlier_spectra_reconstructions.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_eigenspectra_comparison():
    """Figure: toy_eigenspectra_comparison.pdf

    4-method side-by-side comparison of basis vectors (eigenspectra):
    PCA, RPCA, RHMF, and ground truth on same wavelength grid.
    """
    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)

    # Get train/test split (same as RHMF training for fair comparison)
    N = all_spectra_for_fit.shape[0]
    rng = np.random.RandomState(0)  # Match analyse_toy.py seed
    indices = rng.permutation(N)
    split = int(N * 0.5)  # 50% train (from run_toy_gen_and_fits.py)
    train_idx = indices[:split]

    # Train PCA on training set only (fair comparison with RHMF)
    U_pca, S_pca, Vh_pca = np.linalg.svd(all_spectra_for_fit[train_idx], full_matrices=False)
    pca_basis = Vh_pca[:PLOT_K, :].T  # (M, K)

    # Train RPCA on training set only (fair comparison with RHMF)
    Vh_rpca = _load_or_compute_rpca(all_spectra_for_fit[train_idx])
    rpca_basis = Vh_rpca[:PLOT_K, :].T  # (M, K)

    # Get RHMF basis from the trained model. The factorization is only identified up
    # to a KxK orthogonal rotation within the learned subspace, so we rotate to the
    # same canonical frame PCA uses: principal axes of the coefficients, ordered by
    # coefficient variance. Without this, each component is an arbitrary mixture of
    # the others and comparison against PCA/RPCA/truth is meaningless.
    result_ind = np.where(
        (np.array([r.Q for r in results]) == PLOT_Q) & (np.array([r.K for r in results]) == PLOT_K)
    )[0][0]
    trained_state = results[result_ind].state

    G = np.array(trained_state.G)  # (M, K), orthonormal columns
    A = np.array(trained_state.A)  # (N, K)
    evals, V = np.linalg.eigh(A.T @ A)
    V = V[:, np.argsort(evals)[::-1]]  # order by coefficient variance, descending
    rhmf_basis = G @ V  # (M, K), still orthonormal

    # Get true basis (if available), ordered by true coefficient variance so it
    # follows the same canonical convention as the fitted methods.
    true_basis = data.get("true_basis", None)
    if true_basis is not None:
        true_basis = true_basis.T  # stored (K, M) -> (M, K)
        if true_basis.shape[1] > PLOT_K:
            true_basis = true_basis[:, :PLOT_K]
        true_coeffs = data.get("true_coeffs", None)
        if true_coeffs is not None:
            # Order by each component's data-power contribution E[a^2]*||g||^2 (raw
            # second moment, not variance, since the fitted methods run on
            # non-mean-subtracted data). This matches the canonical ordering the
            # fitted methods use.
            coeff_ms = np.mean(np.array(true_coeffs)[:, :PLOT_K] ** 2, axis=0)
            power = coeff_ms * np.sum(true_basis**2, axis=0)
            true_basis = true_basis[:, np.argsort(power)[::-1]]

    # Ensure all bases are (M, K)
    if pca_basis.shape[1] > PLOT_K:
        pca_basis = pca_basis[:, :PLOT_K]
    if rpca_basis.shape[1] > PLOT_K:
        rpca_basis = rpca_basis[:, :PLOT_K]
    if rhmf_basis.shape[1] > PLOT_K:
        rhmf_basis = rhmf_basis[:, :PLOT_K]

    # L2-normalize all bases for fair comparison
    pca_basis = pca_basis / np.linalg.norm(pca_basis, axis=0, keepdims=True)
    rpca_basis = rpca_basis / np.linalg.norm(rpca_basis, axis=0, keepdims=True)
    rhmf_basis = rhmf_basis / np.linalg.norm(rhmf_basis, axis=0, keepdims=True)
    if true_basis is not None:
        true_basis = true_basis / np.linalg.norm(true_basis, axis=0, keepdims=True)

        # Eigenvector signs are arbitrary for every method; flip each component to
        # positively align with its best-matching true component for readability.
        for basis in (pca_basis, rpca_basis, rhmf_basis):
            for k in range(basis.shape[1]):
                overlaps = true_basis.T @ basis[:, k]
                j = np.argmax(np.abs(overlaps))
                if overlaps[j] < 0:
                    basis[:, k] *= -1

    # Create figure: 4 columns (methods) × K rows (components)
    n_methods = 4 if true_basis is not None else 3
    fig, axes = plt.subplots(PLOT_K, n_methods, figsize=(14, 12), sharex=True, dpi=100)
    if PLOT_K == 1:
        axes = axes.reshape(1, -1)

    methods = ["PCA", "RPCA", "RHMF", "True"] if true_basis is not None else ["PCA", "RPCA", "RHMF"]
    bases = [pca_basis, rpca_basis, rhmf_basis]
    if true_basis is not None:
        bases.append(true_basis)

    for i in range(PLOT_K):
        for j, (method, basis) in enumerate(zip(methods, bases)):
            ax = axes[i, j]
            # Ensure basis has correct shape (M, K) and extract component i
            if basis.shape[1] > i:  # Check that component i exists
                component = basis[:, i]
                ax.plot(grid / 10, component, color=f"C{i}", lw=2, alpha=0.9)
            else:
                ax.text(0.5, 0.5, f"N/A", ha="center", va="center", transform=ax.transAxes)

            if i == 0:
                ax.set_title(method, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"K{i+1}", fontsize=10)
            else:
                ax.set_yticklabels([])

            ylim = 0.05 if i == 0 else 0.15
            ax.set_ylim(-ylim, ylim)
            ax.tick_params(labelsize=9)

    axes[-1, 0].set_xlabel("Wavelength [nm]", fontsize=10)

    fig.suptitle(r"$\textsf{\textbf{Toy Dataset: Eigenspectra Comparison}}$",
                fontsize="20", c="dimgrey", y=0.98)
    plt.tight_layout()

    out = PAPER_FIGS / "toy_eigenspectra_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _canonical_coefficients(data, results, rhmf_objs, all_spectra_for_fit, all_ivar):
    """Shared conventions for the method-comparison figures.

    Fits PCA and RPCA on the same training split used for RHMF, and expresses
    the RHMF solution in the canonical frame (principal axes of the coefficient
    second moment, ordered by coefficient power) that the SVD gives PCA/RPCA
    for free. Returns:
        rhmf_basis : (M, K) rotated orthonormal RHMF basis
        pca_basis  : (M, K) train-set PCA basis
        rpca_basis : (M, K) train-set RPCA basis
        rhmf_A     : (N, K) all-data RHMF coefficients in the canonical frame
        true_power : (K,) per-component data power of the true components,
                     E[a^2] * ||g||^2, sorted descending
    """
    N = all_spectra_for_fit.shape[0]
    rng = np.random.RandomState(0)
    train_idx = rng.permutation(N)[: int(N * 0.5)]
    Y_train = all_spectra_for_fit[train_idx]

    U_pca, S_pca, Vh_pca = np.linalg.svd(Y_train, full_matrices=False)
    pca_basis = Vh_pca[:PLOT_K, :].T

    Vh_rpca = _load_or_compute_rpca(Y_train)
    rpca_basis = Vh_rpca[:PLOT_K, :].T

    # RHMF: coefficients for every spectrum (G held at its trained value), then
    # rotate to the principal axes of the coefficient second moment.
    plot_rhmf, all_state = _plot_model_and_state(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )
    A = np.array(all_state.A)
    G = np.array(all_state.G)
    evals, V = np.linalg.eigh(A.T @ A)
    V = V[:, np.argsort(evals)[::-1]]
    rhmf_A = A @ V
    rhmf_basis = G @ V

    true_basis_raw = np.array(data["true_basis"])[:PLOT_K, :]  # (K, M), not unit norm
    true_coeffs = np.array(data["true_coeffs"])[:, :PLOT_K]
    true_power = np.sort(
        np.mean(true_coeffs**2, axis=0) * np.sum(true_basis_raw**2, axis=1)
    )[::-1]

    return rhmf_basis, pca_basis, rpca_basis, rhmf_A, true_power


def fig_explained_variance():
    """Figure: toy_explained_variance.pdf

    Per-component captured data power for PCA, RPCA, RHMF, and the truth, as a
    percentage of the total mean-square data power. All methods are fit on the
    training set, evaluated on the full dataset, and expressed in the same
    canonical frame as the eigenspectra figure, so the panels are directly
    comparable.
    """
    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)

    pca_basis, rpca_basis, rhmf_A_rot, true_power = _canonical_coefficients(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )[1:]

    total_power = np.mean(np.sum(all_spectra_for_fit**2, axis=1))

    # Per-component captured power: mean-square projection coefficient (bases are
    # orthonormal, so this is the power along each component direction).
    pca_power = np.mean((all_spectra_for_fit @ pca_basis) ** 2, axis=0)
    rpca_power = np.mean((all_spectra_for_fit @ rpca_basis) ** 2, axis=0)
    rhmf_power = np.mean(rhmf_A_rot**2, axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=100, sharey=True)
    panels = [
        ("PCA", pca_power),
        ("RPCA", rpca_power),
        ("RHMF", rhmf_power),
        ("True", true_power),
    ]
    for ax, (label, power) in zip(axes, panels):
        frac = 100 * power / total_power
        ax.bar(np.arange(1, PLOT_K + 1), frac, color="C0", alpha=0.7, edgecolor="black", lw=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("Component", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, PLOT_K + 1))
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Captured data power (%)", fontsize=10)

    fig.suptitle(r"$\textsf{\textbf{Toy Dataset: Captured Power by Component}}$",
                fontsize="20", c="dimgrey", y=0.98)
    plt.tight_layout()

    out = PAPER_FIGS / "toy_explained_variance.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_coefficient_distributions():
    """Figure: toy_coefficient_distributions.pdf

    Per-component coefficient distributions for PCA, RPCA, and RHMF, split into
    normal vs outlier spectra. Directly addresses whether outliers can be
    identified from coefficient values alone for each method. All methods use
    the shared conventions in _canonical_coefficients.
    """
    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)
    is_outlier = data["os_mask"].any(axis=1)

    _, pca_basis, rpca_basis, rhmf_A, _ = _canonical_coefficients(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )

    # Projection coefficients onto the (orthonormal) train-set bases.
    pca_A = all_spectra_for_fit @ pca_basis
    rpca_A = all_spectra_for_fit @ rpca_basis

    fig, axes = plt.subplots(3, PLOT_K, figsize=(14, 8), dpi=100)

    for i, (label, A) in enumerate([("PCA", pca_A), ("RPCA", rpca_A), ("RHMF", rhmf_A)]):
        for k in range(PLOT_K):
            ax = axes[i, k]
            bins = np.histogram_bin_edges(A[:, k], bins=40)
            ax.hist(A[~is_outlier, k], bins=bins, color="C0", alpha=0.7,
                    label="Normal Spectra")
            ax.hist(A[is_outlier, k], bins=bins, color="C1", alpha=0.7,
                    hatch="oo", edgecolor="#8B4513", lw=0, label="Outlier Spectra")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"K{k+1}", fontsize=11, fontweight="bold")
            if k == 0:
                ax.set_ylabel(label, fontsize=10)
            if i == 2:
                ax.set_xlabel("Coefficient", fontsize=9)
            ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False,
               bbox_to_anchor=(0.99, 1.005))

    fig.suptitle(r"$\textsf{\textbf{Toy Dataset: Coefficient Distributions}}$",
                fontsize="20", c="dimgrey", y=0.98)
    plt.tight_layout()

    out = PAPER_FIGS / "toy_coefficient_distributions.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_toy_diversity():
    """Figure: toy_dataset_diversity.pdf

    9-panel display showing representative clean and outlier spectra from the
    toy dataset, with visual encoding of different outlier types.
    """
    data, _, _ = _load_results()
    all_noisy_spectra = data["noisy_spectra"]
    os_mask = data["os_mask"]
    op_mask = data["op_mask"]
    oc_mask = data["oc_mask"]
    al_mask = data["al_mask"]
    missing_mask = data["missing_mask"]
    weird_spectra_idx = data["weird_spectra_idx"]
    grid = data["grid"]

    # Select representative spectra
    clean_mask = ~(os_mask | op_mask | oc_mask | al_mask | missing_mask).any(axis=1)
    clean_indices = np.where(clean_mask)[0]
    rng = default_rng(seed=42)
    clean_selected = rng.choice(clean_indices, size=3, replace=False)

    # Spectrum outlier
    spectrum_outlier_idx = weird_spectra_idx[0]

    # Pixel outlier only
    pixel_only_mask = op_mask.any(axis=1) & ~(os_mask | oc_mask | al_mask | missing_mask).any(axis=1)
    pixel_outlier_idx = np.where(pixel_only_mask)[0][0] if pixel_only_mask.any() else np.where(op_mask.any(axis=1))[0][0]

    # Column outlier only
    column_only_mask = oc_mask.any(axis=1) & ~(os_mask | op_mask | al_mask | missing_mask).any(axis=1)
    column_outlier_idx = np.where(column_only_mask)[0][0] if column_only_mask.any() else np.where(oc_mask.any(axis=1))[0][0]

    # Absorption line outlier only
    al_only_mask = al_mask.any(axis=1) & ~(os_mask | op_mask | oc_mask | missing_mask).any(axis=1)
    al_outlier_idx = np.where(al_only_mask)[0][0] if al_only_mask.any() else np.where(al_mask.any(axis=1))[0][0]

    # Missing data only
    missing_only_mask = missing_mask.any(axis=1) & ~(os_mask | op_mask | oc_mask | al_mask).any(axis=1)
    missing_idx = np.where(missing_only_mask)[0][0] if missing_only_mask.any() else np.where(missing_mask.any(axis=1))[0][0]

    # Complex case (multiple outlier types)
    multi_outlier_mask = (op_mask | oc_mask | al_mask).any(axis=1) & missing_mask.any(axis=1)
    if multi_outlier_mask.any():
        complex_idx = np.where(multi_outlier_mask)[0][0]
    else:
        complex_idx = np.where((op_mask | oc_mask | al_mask | missing_mask).any(axis=1))[0][-1]

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=100, sharex=True)
    axes = axes.flatten()

    indices = [
        *clean_selected,
        spectrum_outlier_idx,
        pixel_outlier_idx,
        column_outlier_idx,
        al_outlier_idx,
        missing_idx,
        complex_idx,
    ]

    labels = [
        "Clean Spectrum 1",
        "Clean Spectrum 2",
        "Clean Spectrum 3",
        "Spectrum Outlier",
        "Pixel Outliers",
        "Column Outlier",
        "Absorption Line Outlier",
        "Missing Data",
        "Mixed Outliers",
    ]

    flux_min, flux_max = -0.3, 1.3

    for i, (ax, idx, label) in enumerate(zip(axes, indices, labels)):
        spec = np.nan_to_num(all_noisy_spectra[idx, :], nan=_SPECTRA_MEAN)
        ax.plot(grid / 10, spec, color="black", lw=1.0, alpha=1.0, zorder=3)

        # Outlier highlighting (low alpha so data dominates)
        if os_mask[idx, :].any():
            ax.axvspan(grid.min() / 10, grid.max() / 10, alpha=0.08, color="C1", zorder=-1)

        if op_mask[idx, :].any():
            op_pixels = np.where(op_mask[idx, :])[0]
            ax.vlines(grid[op_pixels] / 10, ymin=flux_min, ymax=flux_max,
                     color="gray", alpha=0.3, lw=0.5, zorder=0)

        if oc_mask[idx, :].any():
            oc_pixels = np.where(oc_mask[idx, :])[0]
            if oc_pixels.size > 0:
                oc_wavelengths = grid[oc_pixels] / 10
                ax.axvspan(oc_wavelengths.min() - 2, oc_wavelengths.max() + 2,
                          alpha=0.08, color="gray", zorder=-1)

        if al_mask[idx, :].any():
            al_pixels = np.where(al_mask[idx, :])[0]
            ax.vlines(grid[al_pixels] / 10, ymin=flux_min, ymax=flux_max,
                     color="C0", alpha=0.3, lw=0.5, zorder=0)

        # Set fixed flux range
        ax.set_ylim(flux_min, flux_max)

        # Minimalist styling
        ax.set_title(label, fontsize=9, fontweight="bold", pad=4)

        is_left = i % 3 == 0
        is_bottom = i >= 6

        if is_left:
            ax.set_ylabel("Flux", fontsize=8)
        else:
            ax.set_ylabel("")

        if is_bottom:
            ax.set_xlabel("Wavelength [nm]", fontsize=8)
        else:
            ax.set_xlabel("")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelsize=7)
        if not is_left:
            ax.set_yticklabels([])
        if not is_bottom:
            ax.set_xticklabels([])

    fig.suptitle(r"$\textsf{\textbf{Toy Dataset: Representative Spectra}}$",
                fontsize="18", c="dimgrey", y=0.98)
    plt.tight_layout()

    out = PAPER_FIGS / "toy_dataset_diversity.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


FIGURES = {
    "toy_weights": fig_toy_weights,
    "toy_residuals": fig_toy_residuals,
    "cv": fig_cv,
    "toy_spectra": fig_toy_spectra,
    "toy_diversity": fig_toy_diversity,
    "eigenspectra": fig_eigenspectra_comparison,
    "explained_variance": fig_explained_variance,
    "coefficients": fig_coefficient_distributions,
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate toy-example paper figures.")
    parser.add_argument(
        "name",
        choices=list(FIGURES) + ["all"],
        help="Which figure to generate (or 'all').",
    )
    args = parser.parse_args()

    if not PAPER_FIGS.exists():
        raise FileNotFoundError(f"Paper figs directory does not exist: {PAPER_FIGS}")

    if args.name == "all":
        for name, fn in FIGURES.items():
            print(f"\n=== {name} ===")
            fn()
    else:
        FIGURES[args.name]()


if __name__ == "__main__":
    main()
