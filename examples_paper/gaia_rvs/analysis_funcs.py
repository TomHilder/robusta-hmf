"""
Reusable analysis functions for Gaia RVS robust matrix factorization.
"""

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bins import build_all_bins
from collect import MatchedData, compute_abs_mag
from rvs_plot_utils import add_line_markers, load_linelists
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState, load_state_from_npz

import json

import gaia_config as cfg


# === Data structures === #


@dataclass
class BinResults:
    """Container for loaded model results for a single bin."""

    i_bin: int
    ranks: list
    q_vals: list
    states: list  # List of RHMFState
    rhmf_objs: list  # List of Robusta objects with states attached


@dataclass
class CVScores:
    """Container for cross-validation scores across (K, Q) grid."""

    std_z: np.ndarray  # std of z-scores (target: 1.0)
    chi2_red: np.ndarray  # reduced chi-squared (target: 1.0)
    rmse: np.ndarray  # weighted RMSE (lower is better)
    mad_z: np.ndarray  # median absolute z-score (target: 0.6745)
    ranks: list
    q_vals: list


@dataclass
class BinAnalysis:
    """All computed results for a single bin. Plotting needs no recomputation."""

    i_bin: int
    bin_data: object  # Bin object (has .bp_rp, .abs_mag_G, .idx, .ids)
    all_Y: np.ndarray  # (n_spectra, n_pixels) flux
    all_W: np.ndarray  # (n_spectra, n_pixels) weights
    train_idx: np.ndarray  # indices into all_Y for training set
    test_idx: np.ndarray  # indices into all_Y for test set
    source_ids: np.ndarray  # Gaia source IDs, same order as all_Y
    λ_grid: np.ndarray  # wavelength grid (n_pixels,)
    cv_scores: CVScores
    best_K: int
    best_Q: float
    best_rhmf: object  # Robusta object for best model
    best_state: object  # RHMFState (inferred on all data)
    all_reconstructions: np.ndarray  # (n_spectra, n_pixels)
    basis: np.ndarray  # (n_pixels, K)
    outlier_scores: np.ndarray  # (n_spectra,) per-object scores
    all_robust_weights: np.ndarray  # (n_spectra, n_pixels)
    outlier_indices: np.ndarray  # indices into all_Y of outliers
    weight_threshold: float  # threshold used to compute outlier_indices
    outliers_df: object  # pd.DataFrame with outlier metadata


# === Data loading and preparation === #


def build_bins_from_config():
    """Build all bins using shared config parameters."""
    data = MatchedData()

    bp_rp = data["bp_rp"]
    abs_mag_G = compute_abs_mag(data["phot_g_mean_mag"], data["parallax"])

    bp_rp_bin_centres, abs_mag_G_bin_centres = cfg.get_bin_centres()
    bp_rp_width, abs_mag_G_width = cfg.get_bin_widths()

    bins = build_all_bins(
        data,
        bp_rp,
        abs_mag_G,
        bp_rp_bin_centres,
        abs_mag_G_bin_centres,
        bp_rp_width,
        abs_mag_G_width,
    )

    return data, bins, bp_rp, abs_mag_G


def get_test_train_split_idx(n_spectra, train_frac, seed=None):
    """Split indices into train and test sets."""
    if seed is None:
        seed = cfg.RNG_SEED
    indices = np.arange(n_spectra)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(n_spectra * train_frac)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return train_indices, test_indices


def clip_edge_pix(flux, u_flux, n_clip=None):
    """Clip edge pixels from spectra."""
    if n_clip is None:
        n_clip = cfg.N_CLIP_PIX
    if isinstance(n_clip, int):
        n_clip_l = n_clip_r = n_clip
    elif isinstance(n_clip, (list, tuple)) and len(n_clip) == 2:
        n_clip_l, n_clip_r = n_clip
    else:
        raise ValueError("n_clip must be an int or a tuple/list of two ints.")
    return flux[:, n_clip_l:-n_clip_r], u_flux[:, n_clip_l:-n_clip_r]


def nans_mask(arrs):
    """Create mask for non-NaN values across multiple arrays."""
    nans = np.isnan(np.array(arrs))
    return np.logical_not(np.any(nans, axis=0))


def prep_data(flux, u_flux):
    """Prepare flux and weights, handling NaNs."""
    weights = 1.0 / (u_flux**2)
    Y, W = flux.copy(), weights.copy()
    mask = nans_mask([Y, W])
    Y[~mask] = 0.0
    W[~mask] = 0.0
    Y = np.nan_to_num(Y)
    W = np.nan_to_num(W)
    return Y, W


def load_all_spectra_for_bin(data, bin_data, train_frac, n_clip=None):
    """
    Load and prepare all spectra (train + test) for a bin.

    Returns
    -------
    all_Y, all_W : arrays
        Combined train+test flux and weights
    train_idx, test_idx : arrays
        Indices into all_Y/all_W for train and test sets
    source_ids : array
        Gaia source IDs for all spectra (in same order as all_Y)
    """
    if n_clip is None:
        n_clip = cfg.N_CLIP_PIX

    train_idx, test_idx = get_test_train_split_idx(bin_data.n_spectra, train_frac)

    # Load all spectra
    all_flux, all_u_flux = clip_edge_pix(
        *data.get_flux_batch(bin_data.idx), n_clip=n_clip
    )
    all_Y, all_W = prep_data(all_flux, all_u_flux)

    # Source IDs in same order
    source_ids = bin_data.ids

    return all_Y, all_W, train_idx, test_idx, source_ids


# === Model loading === #


def load_bin_results(i_bin, ranks, q_vals, results_dir):
    """
    Load saved model states for a bin.

    Parameters
    ----------
    i_bin : int
        Bin index
    ranks : list of int
        Rank (K) values to load
    q_vals : list of float
        Q values to load
    results_dir : Path
        Directory containing saved states

    Returns
    -------
    BinResults
        Container with loaded states and Robusta objects
    """
    results_dir = Path(results_dir)
    Q_grid, Rank_grid = np.meshgrid(q_vals, ranks)

    states = []
    rhmf_objs = []
    missing = []

    for Q, rank in zip(Q_grid.flatten(), Rank_grid.flatten()):
        state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_bin_{i_bin}.npz"
        if not state_file.exists():
            missing.append((rank, Q))
            continue

        state = load_state_from_npz(state_file)
        states.append(state)

        rhmf = Robusta(rank=rank, robust_scale=Q)
        rhmf._state = state
        rhmf_objs.append(rhmf)

    if missing:
        print(f"Warning: Missing {len(missing)} state files for bin {i_bin}: {missing[:5]}...")

    return BinResults(
        i_bin=i_bin,
        ranks=ranks,
        q_vals=q_vals,
        states=states,
        rhmf_objs=rhmf_objs,
    )


# === Cross-validation metrics === #


def compute_std_z(rhmf, state, Y, W):
    """Compute std of z-scores. Target: 1.0"""
    residuals = rhmf.residuals(Y=Y, state=state)
    robust_weights = rhmf.robust_weights(Y, W, state=state)
    z_scores = residuals * np.sqrt(W) * np.sqrt(robust_weights)
    return np.std(z_scores)


def compute_chi2_red(rhmf, state, Y, W):
    """Compute reduced chi-squared. Target: 1.0"""
    residuals = rhmf.residuals(Y=Y, state=state)
    chi2 = (residuals * np.sqrt(W)) ** 2
    return np.mean(chi2)


def compute_rmse(rhmf, state, Y, W):
    """Compute weighted RMSE. Lower is better."""
    residuals = rhmf.residuals(Y=Y, state=state)
    wmse = np.mean(W * residuals**2)
    return np.sqrt(wmse)


def compute_mad_z(rhmf, state, Y, W):
    """Compute median absolute z-score. Target: 0.6745"""
    residuals = rhmf.residuals(Y=Y, state=state)
    robust_weights = rhmf.robust_weights(Y, W, state=state)
    z_scores = residuals * np.sqrt(W) * np.sqrt(robust_weights)
    return np.median(np.abs(z_scores))


def compute_all_cv_scores(bin_results, Y_test, W_test, verbose=True):
    """
    Compute all CV scores for a bin across the (K, Q) grid.

    Parameters
    ----------
    bin_results : BinResults
        Loaded model results
    Y_test, W_test : arrays
        Test set data (for CV scores)
    verbose : bool
        Show progress bar

    Returns
    -------
    CVScores
        Container with all metric arrays
    """
    std_z_scores = []
    chi2_red_scores = []
    rmse_scores = []
    mad_z_scores = []

    iterator = bin_results.rhmf_objs
    if verbose:
        iterator = tqdm(iterator, desc=f"Computing CV scores for bin {bin_results.i_bin}")

    for rhmf in iterator:
        # Infer on test set for CV scoring
        test_state, _ = rhmf.infer(
            Y_infer=Y_test,
            W_infer=W_test,
            max_iter=1000,
            conv_tol=1e-4,
            conv_check_cadence=5,
        )

        std_z_scores.append(compute_std_z(rhmf, test_state, Y_test, W_test))
        chi2_red_scores.append(compute_chi2_red(rhmf, test_state, Y_test, W_test))
        rmse_scores.append(compute_rmse(rhmf, test_state, Y_test, W_test))
        mad_z_scores.append(compute_mad_z(rhmf, test_state, Y_test, W_test))

    # Reshape to (n_ranks, n_q_vals)
    n_ranks = len(bin_results.ranks)
    n_q_vals = len(bin_results.q_vals)

    cv_scores = CVScores(
        std_z=np.array(std_z_scores).reshape(n_ranks, n_q_vals),
        chi2_red=np.array(chi2_red_scores).reshape(n_ranks, n_q_vals),
        rmse=np.array(rmse_scores).reshape(n_ranks, n_q_vals),
        mad_z=np.array(mad_z_scores).reshape(n_ranks, n_q_vals),
        ranks=bin_results.ranks,
        q_vals=bin_results.q_vals,
    )

    return cv_scores


def find_best_model(cv_scores, metric="std_z"):
    """
    Find best (K, Q) based on a chosen CV metric.

    Parameters
    ----------
    cv_scores : CVScores
        Container with all metric arrays
    metric : str
        Which metric to use for selection. Options:
        - "std_z": std of z-scores, target 1.0 (default)
        - "chi2_red": reduced chi-squared, target 1.0
        - "rmse": weighted RMSE, lower is better
        - "mad_z": median absolute z-score, target 0.6745

    Returns
    -------
    best_rank : int
    best_q : float
    best_idx : int
        Flat index into the model list
    """
    # Get the metric array and compute deviation from target
    if metric == "std_z":
        values = cv_scores.std_z
        deviation = np.abs(values - 1.0)
    elif metric == "chi2_red":
        values = cv_scores.chi2_red
        deviation = np.abs(values - 1.0)
    elif metric == "rmse":
        values = cv_scores.rmse
        deviation = values  # Lower is better, no target
    elif metric == "mad_z":
        values = cv_scores.mad_z
        deviation = np.abs(values - 0.6745)  # Theoretical target for Gaussian
    else:
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Options: std_z, chi2_red, rmse, mad_z"
        )

    best_flat_idx = np.argmin(deviation)

    # Convert to (rank_idx, q_idx)
    rank_idx, q_idx = np.unravel_index(best_flat_idx, values.shape)

    best_rank = cv_scores.ranks[rank_idx]
    best_q = cv_scores.q_vals[q_idx]

    return best_rank, best_q, best_flat_idx


# === Outlier detection === #


def default_outlier_score(pixel_weights):
    """Default outlier scoring: median of per-pixel weights."""
    return np.median(pixel_weights)


def compute_outlier_scores(rhmf, Y, W, state, score_func=None):
    """
    Compute outlier score per spectrum using a custom scoring function.

    Parameters
    ----------
    rhmf : Robusta
        Model object
    Y, W : arrays
        Data and weights
    state : RHMFState
        Model state
    score_func : callable, optional
        Function that takes per-pixel weights (1D array) and returns a scalar score.
        Lower score = more outlier-y. Default: np.median

    Returns
    -------
    scores : array
        Outlier score per spectrum
    all_pixel_weights : array
        Full per-pixel weights (n_spectra x n_pixels)
    """
    if score_func is None:
        score_func = default_outlier_score

    all_pixel_weights = rhmf.robust_weights(Y, W, state=state)
    scores = np.array([score_func(all_pixel_weights[i]) for i in range(len(Y))])

    return scores, all_pixel_weights


def get_outlier_indices(scores, threshold):
    """Get indices of spectra with score below threshold."""
    return np.where(scores < threshold)[0]


def batched_infer(rhmf, all_Y, all_W, batch_size=50_000, verbose=True, **infer_kwargs):
    """
    Run rhmf.infer() in batches to limit memory usage.

    Each spectrum's coefficients are independent (G is fixed), so batching
    is mathematically equivalent to processing all at once.

    Returns
    -------
    state : RHMFState
        State with concatenated A matrix and shared G.
    """
    N = all_Y.shape[0]
    if N <= batch_size:
        state, _ = rhmf.infer(Y_infer=all_Y, W_infer=all_W, **infer_kwargs)
        return state

    A_chunks = []
    n_batches = (N + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        if verbose:
            print(f"  Batch {i + 1}/{n_batches} ({start}:{end})...")
        chunk_state, _ = rhmf.infer(
            Y_infer=all_Y[start:end],
            W_infer=all_W[start:end],
            **infer_kwargs,
        )
        A_chunks.append(np.array(chunk_state.A))
        G = np.array(chunk_state.G)
        gc.collect()
        jax.clear_caches()

    A_full = np.concatenate(A_chunks, axis=0)
    return RHMFState(A=A_full, G=G, it=0)


# === Save/load helpers === #


def save_bin_results(analysis, plots_dir, results_dir):
    """
    Save per-bin results to disk for later use by summarise_bins.py
    and replot_outliers.py.

    Saves:
    - {plots_dir}/bin_{i:02d}/outliers.csv — per-bin outlier DataFrame
    - {plots_dir}/bin_{i:02d}/summary.json — summary metadata
    - {results_dir}/inferred_all_data_R{K}_Q{Q:.2f}_bin_{i}.npz — A + G matrices
    - {plots_dir}/bin_{i:02d}/all_outlier_scores.npy — per-object scores for ALL spectra
    - {plots_dir}/bin_{i:02d}/all_source_ids.npy — source IDs for ALL spectra in this bin
    - {plots_dir}/bin_{i:02d}/outlier_data.npz — data needed to re-plot outliers
    """
    a = analysis
    plots_dir = Path(plots_dir)
    results_dir = Path(results_dir)
    bin_plots_dir = plots_dir / f"bin_{a.i_bin:02d}"
    bin_plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Outlier CSV
    a.outliers_df.to_csv(bin_plots_dir / "outliers.csv", index=False)

    # 2. Summary JSON
    summary = {
        "i_bin": a.i_bin,
        "best_K": int(a.best_K),
        "best_Q": float(a.best_Q),
        "metric": "std_z",
        "n_spectra": int(a.all_Y.shape[0]),
        "n_outliers": int(len(a.outlier_indices)),
        "weight_threshold": float(a.weight_threshold),
    }
    with open(bin_plots_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 3. Inferred state (A + G matrices)
    state_path = (
        results_dir
        / f"inferred_all_data_R{a.best_K}_Q{a.best_Q:.2f}_bin_{a.i_bin}.npz"
    )
    np.savez(state_path, A=np.array(a.best_state.A), G=np.array(a.best_state.G))

    # 4. Per-object scores for ALL spectra (for cross-bin histograms etc.)
    np.save(bin_plots_dir / "all_outlier_scores.npy", a.outlier_scores)

    # 5. All source IDs in this bin (for cross-bin membership analysis)
    np.save(bin_plots_dir / "all_source_ids.npy", a.source_ids)

    # 5. Outlier data for re-plotting (flux, reconstruction, weights, scores, ids, λ)
    outlier_flux = a.all_Y[a.outlier_indices]
    outlier_reconstructions = a.all_reconstructions[a.outlier_indices]
    outlier_robust_weights = a.all_robust_weights[a.outlier_indices]
    outlier_source_ids = a.source_ids[a.outlier_indices]
    outlier_scores_arr = a.outlier_scores[a.outlier_indices]
    np.savez(
        bin_plots_dir / "outlier_data.npz",
        flux=outlier_flux,
        reconstructions=outlier_reconstructions,
        robust_weights=outlier_robust_weights,
        source_ids=outlier_source_ids,
        scores=outlier_scores_arr,
        indices=a.outlier_indices,
        lambda_grid=a.λ_grid,
    )


def load_cached_inferred_state(i_bin, best_K, best_Q, results_dir):
    """
    Check for a cached all-data inferred state and load it if available.

    Returns
    -------
    RHMFState or None
        The cached state, or None if not found.
    """
    results_dir = Path(results_dir)
    state_path = (
        results_dir
        / f"inferred_all_data_R{best_K}_Q{best_Q:.2f}_bin_{i_bin}.npz"
    )
    if state_path.exists():
        data = np.load(state_path)
        return RHMFState(A=data["A"], G=data["G"], it=0)
    return None


def load_outlier_data(plots_dir, i_bin):
    """
    Load saved outlier data for a bin.

    Returns
    -------
    dict with keys: flux, reconstructions, robust_weights, source_ids,
                    scores, indices, lambda_grid
    """
    plots_dir = Path(plots_dir)
    path = plots_dir / f"bin_{i_bin:02d}" / "outlier_data.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "flux": data["flux"],
        "reconstructions": data["reconstructions"],
        "robust_weights": data["robust_weights"],
        "source_ids": data["source_ids"],
        "scores": data["scores"],
        "indices": data["indices"],
        "lambda_grid": data["lambda_grid"],
    }


def compute_bin_analysis(
    i_bin,
    data,
    bins,
    ranks,
    q_vals,
    train_frac,
    weight_threshold,
    results_dir,
    best_model_metric="std_z",
    outlier_score_func=None,
    verbose=True,
):
    """
    Run all expensive computation for a single bin.

    Returns
    -------
    BinAnalysis or None
        None if the bin has 0 spectra or no trained models.
    """
    bin_data = bins[i_bin]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Analysing bin {i_bin} | N spectra: {bin_data.n_spectra}")
        print(f"{'=' * 60}")

    if bin_data.n_spectra == 0:
        print(f"Skipping bin {i_bin}: no spectra")
        return None

    # Load all spectra
    if verbose:
        print("Loading spectra...")
    all_Y, all_W, train_idx, test_idx, source_ids = load_all_spectra_for_bin(
        data, bin_data, train_frac
    )

    # Split for CV scoring
    Y_test, W_test = all_Y[test_idx], all_W[test_idx]

    # Load model results
    if verbose:
        print("Loading trained models...")
    bin_results = load_bin_results(i_bin, ranks, q_vals, results_dir)

    if len(bin_results.rhmf_objs) == 0:
        print(f"No models found for bin {i_bin}, skipping")
        return None

    # Compute CV scores on test set
    if verbose:
        print("Computing CV scores...")
    cv_scores = compute_all_cv_scores(
        bin_results, Y_test, W_test, verbose=verbose
    )

    # Find best model
    best_K, best_Q, best_idx = find_best_model(cv_scores, metric=best_model_metric)
    if verbose:
        metric_values = {
            "std_z": cv_scores.std_z.flatten()[best_idx],
            "chi2_red": cv_scores.chi2_red.flatten()[best_idx],
            "rmse": cv_scores.rmse.flatten()[best_idx],
            "mad_z": cv_scores.mad_z.flatten()[best_idx],
        }
        print(
            f"Best model: K={best_K}, Q={best_Q:.2f} "
            f"({best_model_metric}={metric_values[best_model_metric]:.4f})"
        )

    # Infer best model on all data (only the best, not all models)
    best_rhmf = bin_results.rhmf_objs[best_idx]

    # Check for cached inference
    cached_state = load_cached_inferred_state(i_bin, best_K, best_Q, results_dir)
    if cached_state is not None:
        if verbose:
            print("Loaded cached inference for all data")
        best_state = cached_state
    else:
        if verbose:
            print("Inferring best model on all data...")
        best_state = batched_infer(
            best_rhmf, all_Y, all_W,
            batch_size=50_000,
            verbose=verbose,
            max_iter=1000,
            conv_tol=1e-4,
            conv_check_cadence=5,
        )

    # Compute outlier scores using custom or default function
    outlier_scores, all_robust_weights = compute_outlier_scores(
        best_rhmf, all_Y, all_W, best_state, score_func=outlier_score_func
    )

    # Find outliers
    outlier_indices = get_outlier_indices(outlier_scores, weight_threshold)
    if verbose:
        print(f"Found {len(outlier_indices)} outliers with score < {weight_threshold}")

    # Wavelength grid
    λ_grid = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]

    # Reconstructions and basis
    all_reconstructions = best_rhmf.synthesize(state=best_state)
    basis = best_rhmf.basis_vectors(state=best_state)

    # Build outliers DataFrame (cheap metadata packaging)
    outliers_data = []
    for idx in outlier_indices:
        outliers_data.append(
            {
                "bin": i_bin,
                "idx": idx,
                "source_id": source_ids[idx],
                "score": outlier_scores[idx],
                "best_K": best_K,
                "best_Q": best_Q,
                "in_train": idx in train_idx,
            }
        )
    outliers_df = pd.DataFrame(outliers_data)

    return BinAnalysis(
        i_bin=i_bin,
        bin_data=bin_data,
        all_Y=all_Y,
        all_W=all_W,
        train_idx=train_idx,
        test_idx=test_idx,
        source_ids=source_ids,
        λ_grid=λ_grid,
        cv_scores=cv_scores,
        best_K=best_K,
        best_Q=best_Q,
        best_rhmf=best_rhmf,
        best_state=best_state,
        all_reconstructions=all_reconstructions,
        basis=basis,
        outlier_scores=outlier_scores,
        all_robust_weights=all_robust_weights,
        outlier_indices=outlier_indices,
        weight_threshold=weight_threshold,
        outliers_df=outliers_df,
    )


# === Plotting === #


def plot_cv_heatmaps(cv_scores, i_bin, save_dir, show=False):
    """
    Plot all 4 CV metric heatmaps.

    Parameters
    ----------
    cv_scores : CVScores
    i_bin : int
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=100)

    metrics = [
        ("std_z", cv_scores.std_z, "std(z)", 1.0, "|std(z) - 1|"),
        ("chi2_red", cv_scores.chi2_red, "Reduced Chi-Squared", 1.0, "|chi2_red - 1|"),
        ("rmse", cv_scores.rmse, "Weighted RMSE", None, "RMSE"),
        ("mad_z", cv_scores.mad_z, "Median |z|", 0.6745, "|median|z| - 0.6745|"),
    ]

    for ax, (name, scores, title, target, cbar_label) in zip(axes.flatten(), metrics):
        if target is not None:
            plot_data = np.log(np.abs(scores - target) + 1e-10)
            cbar_label = f"log({cbar_label})"
        else:
            plot_data = scores

        # Use imshow instead of pcolormesh (handles single row/column better)
        im = ax.imshow(
            plot_data,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[-0.5, len(cv_scores.q_vals) - 0.5,
                    min(cv_scores.ranks) - 0.5, max(cv_scores.ranks) + 0.5],
        )
        ax.set_xticks(np.arange(len(cv_scores.q_vals)))
        ax.set_xticklabels([f"{q:.1f}" for q in cv_scores.q_vals])
        ax.set_yticks(cv_scores.ranks)
        ax.set_xlabel("Robust Scale Q")
        ax.set_ylabel("Rank K")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label)

    plt.suptitle(f"Cross-Validation Scores (Bin {i_bin})", fontsize=14)
    plt.tight_layout()

    save_path = save_dir / f"cv_heatmaps_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return save_path


def plot_weights_histograms(
    per_object_weights,
    all_pixel_weights,
    weight_threshold,
    i_bin,
    best_K,
    best_Q,
    save_dir,
    show=False,
):
    """
    Plot histograms of robust weights.

    Parameters
    ----------
    per_object_weights : array
        Median robust weight per spectrum (1D)
    all_pixel_weights : array
        All per-pixel robust weights (2D: n_spectra x n_pixels)
    weight_threshold : float
        Threshold used for outlier detection (shown as vertical line)
    i_bin : int
        Bin index
    best_K, best_Q : int, float
        Model parameters
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    # Left: per-object weights (median per spectrum)
    ax = axes[0]
    ax.hist(per_object_weights, bins=50, alpha=0.7, color="C0", density=False)
    ax.axvline(weight_threshold, color="r", linestyle="--", lw=2, label=f"Threshold = {weight_threshold}")
    ax.set_xlabel("Per-Object Weight (median)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(f"Per-Object Weights (K={best_K}, Q={best_Q:.2f})")
    ax.legend()

    # Right: per-pixel weights (all data points)
    ax = axes[1]
    ax.hist(all_pixel_weights.flatten(), bins=50, alpha=0.7, color="C1", density=False)
    ax.set_xlabel("Per-Pixel Robust Weight")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(f"Per-Pixel Weights (K={best_K}, Q={best_Q:.2f})")

    plt.suptitle(f"Robust Weights Distribution (Bin {i_bin})", fontsize=12)
    plt.tight_layout()

    save_path = save_dir / f"weights_histograms_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    return save_path


def plot_all_spectra_hr_by_weight(
    per_object_weights,
    bp_rp,
    abs_mag_G,
    bin_indices,
    weight_threshold,
    i_bin,
    best_K,
    best_Q,
    save_dir,
    show=False,
):
    """
    Plot HR diagram with ALL spectra in a bin colored by per-object weight.

    Parameters
    ----------
    per_object_weights : array
        Median robust weight per spectrum
    bp_rp : array
        BP-RP colors for spectra in this bin
    abs_mag_G : array
        Absolute G magnitudes for spectra in this bin
    bin_indices : array
        Indices into full dataset (for this bin)
    weight_threshold : float
        Threshold for outlier detection (for reference)
    i_bin : int
        Bin index
    best_K, best_Q : int, float
        Model parameters
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot all spectra colored by weight
    scatter = ax.scatter(
        bp_rp,
        abs_mag_G,
        c=per_object_weights,
        cmap="viridis",
        s=5,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Per-object robust weight")

    # Mark outliers with red edge
    outlier_mask = per_object_weights < weight_threshold
    if np.any(outlier_mask):
        ax.scatter(
            bp_rp[outlier_mask],
            abs_mag_G[outlier_mask],
            c=per_object_weights[outlier_mask],
            cmap="viridis",
            s=20,
            alpha=1.0,
            vmin=0,
            vmax=1,
            edgecolors="red",
            linewidths=1,
        )

    n_outliers = np.sum(outlier_mask)
    ax.set_ylim(15, -5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("Color (BP - RP)")
    ax.set_ylabel("G-Band Absolute Magnitude")
    ax.set_title(
        f"Bin {i_bin} | K={best_K}, Q={best_Q:.2f} | "
        f"N={len(per_object_weights)}, outliers={n_outliers} (threshold={weight_threshold})"
    )

    plt.tight_layout()

    save_path = save_dir / f"hr_by_weight_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    return save_path


def plot_best_model_vs_bin(best_models, save_path, show=False):
    """
    Plot optimal K and Q as a function of bin index.

    Parameters
    ----------
    best_models : dict
        {bin_index: (best_K, best_Q)} mapping
    save_path : Path
    show : bool
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(best_models) == 0:
        print("No best models to plot")
        return None

    bins = sorted(best_models.keys())
    Ks = [best_models[b][0] for b in bins]
    Qs = [best_models[b][1] for b in bins]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=100, sharex=True)

    # Top: K vs bin
    ax = axes[0]
    ax.plot(bins, Ks, "o-", color="C0", markersize=8, linewidth=2)
    ax.set_ylabel("Optimal Rank (K)")
    ax.set_ylim(min(Ks) - 0.5, max(Ks) + 0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # Bottom: Q vs bin
    ax = axes[1]
    ax.plot(bins, Qs, "s-", color="C1", markersize=8, linewidth=2)
    ax.set_ylabel("Optimal Robust Scale (Q)")
    ax.set_xlabel("Bin Index")
    ax.grid(True, alpha=0.3)

    # If Q values are all the same, note it
    if len(set(Qs)) == 1:
        ax.set_title(f"(Q fixed at {Qs[0]:.2f} during training)", fontsize=10)

    plt.suptitle("Optimal Model Parameters vs Bin", fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    return save_path


def plot_random_spectra_reconstructions(
    λ_grid,
    Y,
    reconstructions,
    i_bin,
    best_K,
    best_Q,
    save_dir,
    n_spectra=10,
    seed=42,
    show=False,
):
    """
    Plot random spectra with their reconstructions for sanity checking.

    Parameters
    ----------
    λ_grid : array
        Wavelength grid
    Y : array
        All flux data (n_spectra x n_pixels)
    reconstructions : array
        Model reconstructions
    i_bin : int
    best_K, best_Q : int, float
    save_dir : Path
    n_spectra : int
        Number of random spectra to plot
    seed : int
        Random seed for reproducibility
    show : bool
    """
    save_dir = Path(save_dir)
    rng = np.random.default_rng(seed)

    plot_inds = rng.choice(Y.shape[0], size=min(n_spectra, Y.shape[0]), replace=False)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    for i, (idx, offset) in enumerate(zip(plot_inds, range(len(plot_inds)))):
        ax.plot(λ_grid, Y[idx, :] + offset * 0.5, color=f"C{i % 10}", alpha=0.8, lw=0.8)
        ax.plot(λ_grid, reconstructions[idx, :] + offset * 0.5, color="k", alpha=0.8, lw=0.5)

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Flux + offset")
    ax.set_title(f"Random Spectra (colored) vs Reconstructions (black) | Bin {i_bin}, K={best_K}, Q={best_Q:.2f}")

    save_path = save_dir / f"spectra_reconstructions_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return save_path


def plot_basis_functions(
    λ_grid,
    basis,
    i_bin,
    best_K,
    best_Q,
    save_dir,
    show=False,
):
    """
    Plot the learned basis functions (G matrix columns).

    Parameters
    ----------
    λ_grid : array
        Wavelength grid
    basis : array
        Basis vectors (n_pixels x K)
    i_bin : int
    best_K, best_Q : int, float
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    for k in range(basis.shape[1]):
        offset = k * 0.15
        ax.plot(λ_grid, basis[:, k] + offset, color=f"C{k % 10}", alpha=0.9, lw=1, label=f"Component {k}")

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Basis + offset")
    ax.set_title(f"Learned Basis Functions | Bin {i_bin}, K={best_K}, Q={best_Q:.2f}")
    ax.legend(loc="upper right", fontsize=8)

    save_path = save_dir / f"basis_functions_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return save_path


def plot_weights_heatmap(
    all_robust_weights,
    λ_grid,
    i_bin,
    best_K,
    best_Q,
    save_dir,
    show=False,
):
    """
    Plot 2D heatmap of robust weights (spectra x pixels).

    Parameters
    ----------
    all_robust_weights : array
        Robust weights (n_spectra x n_pixels)
    λ_grid : array
        Wavelength grid
    i_bin : int
    best_K, best_Q : int, float
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

    # Subsample if too many spectra for visualization
    max_spectra = 500
    if all_robust_weights.shape[0] > max_spectra:
        step = all_robust_weights.shape[0] // max_spectra
        weights_to_plot = all_robust_weights[::step, :]
        ylabel = f"Spectrum Index (subsampled 1/{step})"
    else:
        weights_to_plot = all_robust_weights
        ylabel = "Spectrum Index"

    im = ax.imshow(
        weights_to_plot,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[λ_grid[0], λ_grid[-1], 0, weights_to_plot.shape[0]],
    )
    plt.colorbar(im, ax=ax, label="Robust Weight")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Robust Weights Heatmap | Bin {i_bin}, K={best_K}, Q={best_Q:.2f}")

    save_path = save_dir / f"weights_heatmap_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return save_path


def plot_residuals_summary(
    λ_grid,
    Y,
    reconstructions,
    i_bin,
    best_K,
    best_Q,
    save_dir,
    show=False,
):
    """
    Plot summary statistics of residuals across all spectra.

    Parameters
    ----------
    λ_grid : array
    Y : array
        All flux data
    reconstructions : array
    i_bin : int
    best_K, best_Q : int, float
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)

    residuals = Y - reconstructions

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=100, sharex=True)

    # Top: median and percentiles of residuals per pixel
    ax = axes[0]
    median_resid = np.median(residuals, axis=0)
    p16 = np.percentile(residuals, 16, axis=0)
    p84 = np.percentile(residuals, 84, axis=0)

    ax.fill_between(λ_grid, p16, p84, alpha=0.3, color="C0", label="16-84 percentile")
    ax.plot(λ_grid, median_resid, color="C0", lw=1, label="Median")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Residual")
    ax.legend(loc="upper right")
    ax.set_title(f"Residuals Summary | Bin {i_bin}, K={best_K}, Q={best_Q:.2f}")

    # Bottom: RMS of residuals per pixel
    ax = axes[1]
    rms_resid = np.sqrt(np.mean(residuals**2, axis=0))
    ax.plot(λ_grid, rms_resid, color="C1", lw=1)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("RMS Residual")

    plt.tight_layout()

    save_path = save_dir / f"residuals_summary_bin_{i_bin:02d}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return save_path


def plot_outliers_on_hr(
    outliers_df,
    bp_rp_all,
    abs_mag_G_all,
    source_ids_all,
    save_path,
    color_by="score",
    show=False,
):
    """
    Plot detected outliers on the HR diagram.

    Parameters
    ----------
    outliers_df : pd.DataFrame
        Outlier summary with columns: bin, idx, source_id, score, ...
    bp_rp_all : array
        BP-RP colors for all sources in the dataset
    abs_mag_G_all : array
        Absolute G magnitudes for all sources
    source_ids_all : array
        Source IDs for all sources (to match outliers)
    save_path : Path
        Where to save the plot
    color_by : str
        "score" to color by outlier score, "bin" to color by bin index
    show : bool
        Whether to display the plot
    """
    import pandas as pd

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Plot all sources as background (rasterized for smaller file size)
    ax.scatter(
        bp_rp_all,
        abs_mag_G_all,
        s=0.5,
        alpha=0.1,
        c="grey",
        zorder=0,
        marker=".",
        rasterized=True,
    )

    if len(outliers_df) == 0:
        ax.set_title("No outliers detected")
    else:
        # Match outliers to their positions
        # Create lookup from source_id to position
        source_id_to_idx = {sid: i for i, sid in enumerate(source_ids_all)}

        outlier_bp_rp = []
        outlier_abs_mag = []
        outlier_scores = []
        outlier_bins = []

        for _, row in outliers_df.iterrows():
            sid = row["source_id"]
            if sid in source_id_to_idx:
                idx = source_id_to_idx[sid]
                outlier_bp_rp.append(bp_rp_all[idx])
                outlier_abs_mag.append(abs_mag_G_all[idx])
                outlier_scores.append(row["score"])
                outlier_bins.append(row["bin"])

        outlier_bp_rp = np.array(outlier_bp_rp)
        outlier_abs_mag = np.array(outlier_abs_mag)
        outlier_scores = np.array(outlier_scores)
        outlier_bins = np.array(outlier_bins)

        if color_by == "score":
            scatter = ax.scatter(
                outlier_bp_rp,
                outlier_abs_mag,
                c=outlier_scores,
                cmap="viridis_r",
                s=20,
                alpha=0.8,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            cbar = plt.colorbar(scatter, ax=ax, label="Outlier score (lower = more anomalous)")
        elif color_by == "bin":
            scatter = ax.scatter(
                outlier_bp_rp,
                outlier_abs_mag,
                c=outlier_bins,
                cmap="tab20",
                s=20,
                alpha=0.8,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            cbar = plt.colorbar(scatter, ax=ax, label="Bin index")

        # Count duplicates (same source in multiple bins)
        n_unique = outliers_df["source_id"].nunique()
        n_total = len(outliers_df)
        ax.set_title(
            f"Detected Outliers on HR Diagram\n"
            f"{n_total} detections from {n_unique} unique sources"
        )

    ax.set_ylim(15, -5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("Color (BP - RP)")
    ax.set_ylabel("G-Band Absolute Magnitude")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    return save_path


def plot_spectrum_residual(
    λ_grid,
    flux,
    reconstruction,
    robust_weights,
    source_id,
    i_bin,
    idx,
    per_object_weight,
    best_K,
    best_Q,
    save_dir,
    show=False,
):
    """
    Plot spectrum with reconstruction and residuals.

    Parameters
    ----------
    λ_grid : array
        Wavelength grid
    flux : array
        Observed flux (1D)
    reconstruction : array
        Model reconstruction (1D)
    robust_weights : array
        Per-pixel robust weights (1D)
    source_id : int
        Gaia source ID
    i_bin : int
        Bin index
    idx : int
        Spectrum index within bin
    per_object_weight : float
        Median robust weight for this spectrum
    best_K, best_Q : int, float
        Model parameters
    save_dir : Path
    show : bool
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    residual = flux - reconstruction

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

    # Top panel: spectrum and reconstruction
    axes[0].plot(λ_grid, flux, c="k", lw=0.8, label="Observed", alpha=0.8)
    axes[0].plot(λ_grid, reconstruction, c="C2", lw=0.8, label="Reconstruction", alpha=0.8)
    axes[0].set_ylabel("Normalised flux")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"Bin {i_bin} | idx {idx} | source_id {source_id} | "
        f"K={best_K} Q={best_Q:.2f} | median weight={per_object_weight:.3f}"
    )

    # Bottom panel: residual
    ax_resid = axes[1]
    ax_resid.plot(λ_grid, residual, c="k", lw=0.5, label="Residual")
    ax_resid.set_ylabel("Residual")
    ax_resid.set_xlabel("Wavelength [nm]")

    # Add spectral line markers (strong lines only)
    try:
        lines = load_linelists()
        add_line_markers(
            ax=ax_resid,
            lines=lines,
            show_strong=True,
            show_abundance=False,
            show_cn=False,
            show_dib=False,
        )
    except Exception as e:
        print(f"Warning: Could not add line markers: {e}")

    # Fine tick marks for wavelength calibration checking
    from matplotlib.ticker import MultipleLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.tick_params(which="minor", length=3)
        ax.tick_params(which="major", length=6)

    plt.tight_layout()

    # Filename with all relevant info
    filename = (
        f"bin_{i_bin:02d}_K{best_K}_Q{best_Q:.2f}_"
        f"idx_{idx:05d}_srcid_{source_id}_weight_{per_object_weight:.3f}.pdf"
    )
    save_path = save_dir / filename
    plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    return save_path


# === Combined plotting === #


def plot_bin_analysis(
    analysis,
    plots_dir,
    save_residuals=False,
    residuals_dir=None,
    verbose=True,
):
    """
    Generate all plots for a single bin from pre-computed results.

    Parameters
    ----------
    analysis : BinAnalysis
        Pre-computed analysis results (from compute_bin_analysis).
    plots_dir : Path
        Top-level directory for plots. A sub-directory ``bin_{i:02d}`` is created.
    save_residuals : bool
        Whether to save per-outlier residual arrays as .npy files.
    residuals_dir : Path or None
        Directory for residual .npy files. Required when *save_residuals* is True.
    verbose : bool
        Print progress messages.
    """
    plots_dir = Path(plots_dir)
    a = analysis
    bin_plots_dir = plots_dir / f"bin_{a.i_bin:02d}"

    # Clear stale plots from previous runs
    if bin_plots_dir.exists():
        shutil.rmtree(bin_plots_dir)
    bin_plots_dir.mkdir(parents=True, exist_ok=True)

    # CV heatmaps
    if verbose:
        print("Plotting CV heatmaps...")
    plot_cv_heatmaps(a.cv_scores, a.i_bin, bin_plots_dir)

    # Random spectra vs reconstructions
    if verbose:
        print("Plotting diagnostic plots...")
    plot_random_spectra_reconstructions(
        λ_grid=a.λ_grid,
        Y=a.all_Y,
        reconstructions=a.all_reconstructions,
        i_bin=a.i_bin,
        best_K=a.best_K,
        best_Q=a.best_Q,
        save_dir=bin_plots_dir,
    )

    # Basis functions
    plot_basis_functions(
        λ_grid=a.λ_grid,
        basis=a.basis,
        i_bin=a.i_bin,
        best_K=a.best_K,
        best_Q=a.best_Q,
        save_dir=bin_plots_dir,
    )

    # Weights heatmap
    plot_weights_heatmap(
        all_robust_weights=a.all_robust_weights,
        λ_grid=a.λ_grid,
        i_bin=a.i_bin,
        best_K=a.best_K,
        best_Q=a.best_Q,
        save_dir=bin_plots_dir,
    )

    # Residuals summary
    plot_residuals_summary(
        λ_grid=a.λ_grid,
        Y=a.all_Y,
        reconstructions=a.all_reconstructions,
        i_bin=a.i_bin,
        best_K=a.best_K,
        best_Q=a.best_Q,
        save_dir=bin_plots_dir,
    )

    # Weight histograms
    if verbose:
        print("Plotting weight histograms...")
    plot_weights_histograms(
        per_object_weights=a.outlier_scores,
        all_pixel_weights=a.all_robust_weights,
        weight_threshold=a.weight_threshold,
        i_bin=a.i_bin,
        best_K=a.best_K,
        best_Q=a.best_Q,
        save_dir=bin_plots_dir,
    )

    # HR diagram colored by weight
    if verbose:
        print("Plotting HR diagram by weight...")
    plot_all_spectra_hr_by_weight(
        per_object_weights=a.outlier_scores,
        bp_rp=a.bin_data.bp_rp,
        abs_mag_G=a.bin_data.abs_mag_G,
        bin_indices=a.bin_data.idx,
        weight_threshold=a.weight_threshold,
        i_bin=a.i_bin,
        best_K=a.best_K,
        best_Q=a.best_Q,
        save_dir=bin_plots_dir,
    )

    # Individual outlier spectra
    if verbose and len(a.outlier_indices) > 0:
        print("Plotting outlier spectra...")
    for idx in tqdm(
        a.outlier_indices,
        desc=f"Plotting outliers for bin {a.i_bin}",
        disable=not verbose,
    ):
        plot_spectrum_residual(
            λ_grid=a.λ_grid,
            flux=a.all_Y[idx],
            reconstruction=a.all_reconstructions[idx],
            robust_weights=a.all_robust_weights[idx],
            source_id=a.source_ids[idx],
            i_bin=a.i_bin,
            idx=idx,
            per_object_weight=a.outlier_scores[idx],
            best_K=a.best_K,
            best_Q=a.best_Q,
            save_dir=bin_plots_dir,
        )

        if save_residuals:
            residuals_dir = Path(residuals_dir)
            residuals_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                file=residuals_dir
                / f"{a.source_ids[idx]}_residual_bin_{a.i_bin:02d}.npy",
                arr=a.all_Y[idx] - a.all_reconstructions[idx],
            )
