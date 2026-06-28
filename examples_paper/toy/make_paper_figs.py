"""Regenerate the four toy-example paper figures individually.

Each of the four paper figures produced inline by ``analyse_toy.py`` is
reproduced here as a standalone function that performs its own minimal setup,
builds the figure, and writes it into the repo's real paper figures directory.

Usage
-----
    uv run python make_paper_figs.py <name>

where ``<name>`` is one of:
    toy_weights    -> weights_per_object_clean_vs_outlier.pdf
    toy_residuals  -> absorption_line_residuals.pdf
    cv             -> test_set_score_heatmap.pdf
    toy_spectra    -> normal_vs_outlier_spectra_reconstructions.pdf
    all            -> all of the above

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

    Histogram of mean robust weight per spectrum, split into normal vs outlier
    spectra (single-panel version).
    """
    data, results, rhmf_objs = _load_results()
    all_noisy_spectra, all_spectra_for_fit, all_ivar, grid = _all_data_arrays(data)
    os_mask = data["os_mask"]

    plot_rhmf, all_state = _plot_model_and_state(
        data, results, rhmf_objs, all_spectra_for_fit, all_ivar
    )

    # Per-pixel robust weights on all data, then per-object mean.
    weights = plot_rhmf.robust_weights(all_spectra_for_fit, all_ivar, state=all_state)
    per_object_weights = np.mean(weights, axis=1)

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
    ax.set_xlabel("Mean Robust Weight per Spectrum")
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


FIGURES = {
    "toy_weights": fig_toy_weights,
    "toy_residuals": fig_toy_residuals,
    "cv": fig_cv,
    "toy_spectra": fig_toy_spectra,
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
