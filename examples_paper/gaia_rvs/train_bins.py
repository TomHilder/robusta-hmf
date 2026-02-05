"""
Train Robusta models on specified bins of Gaia RVS spectra.

Configure the bins to run, ranks, and Q values at the top of this file.
Bin geometry and data processing parameters are imported from gaia_config.py.
"""

from pathlib import Path

import gaia_config as cfg
import numpy as np
from analysis_funcs import clip_edge_pix, get_test_train_split_idx, nans_mask
from bins import build_all_bins
from collect import MatchedData, compute_abs_mag
from tqdm import tqdm

from robusta_hmf import Robusta, save_state_to_npz

# ============================================================================ #
# CONFIGURATION - Modify these to control what gets trained
# ============================================================================ #

# Which bins to train (by index, 0-13 for 14 bins)
# BINS_TO_RUN = [7, 8, 9, 10, 11, 12, 13]
# BINS_TO_RUN = [1, 2, 6]
BINS_TO_RUN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Rank (K) values to try
# RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RANKS = [5]

# Robustness scale (Q) values to try
# Q_VALS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
Q_VALS = [3.0]

# Other training parameters
MAX_ITER = 1000
TRAIN_FRAC = cfg.TRAIN_FRAC  # From shared config

# Output directory
RESULTS_DIR = Path("./gaia_rvs_results")

# ============================================================================ #


def build_bins():
    """Build all bins from the Gaia data using shared config."""
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

    return data, bins


def train_bin(data, bin_data, i_bin, ranks, q_vals, max_iter, train_frac, results_dir):
    """Train models for a single bin over the rank/Q grid."""
    print(f"\n{'=' * 60}")
    print(f"Training bin {i_bin} | N spectra: {bin_data.n_spectra}")
    print(
        f"BP-RP centre: {bin_data.bp_rp_prop.centre:.2f}, Abs Mag G centre: {bin_data.abs_mag_G_prop.centre:.2f}"
    )
    print(f"{'=' * 60}")

    if bin_data.n_spectra == 0:
        print(f"Skipping bin {i_bin}: no spectra")
        return

    # Create grid
    Q_grid, Rank_grid = np.meshgrid(q_vals, ranks)

    # Train/test split (uses cfg.RNG_SEED)
    train_idx, test_idx = get_test_train_split_idx(
        bin_data.n_spectra,
        train_frac=train_frac,
    )

    # Get the train spectra (uses cfg.N_CLIP_PIX)
    train_flux, train_u_flux = clip_edge_pix(*data.get_flux_batch(bin_data.idx[train_idx]))
    train_weights = 1.0 / (train_u_flux**2)

    Y, W = train_flux, train_weights
    spec_nans_mask = nans_mask([Y, W])
    Y[~spec_nans_mask] = np.nan
    W[~spec_nans_mask] = np.nan
    Y = np.nan_to_num(Y)
    W = np.nan_to_num(W)

    # Fit models over the grid, saving each immediately
    results_dir.mkdir(parents=True, exist_ok=True)
    n_trained = 0
    n_skipped = 0
    for Q, rank in tqdm(
        zip(Q_grid.flatten(), Rank_grid.flatten()),
        total=len(Q_grid.flatten()),
        desc=f"Bin {i_bin}",
    ):
        state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_bin_{i_bin}.npz"

        if state_file.exists():
            n_skipped += 1
            continue

        model = Robusta(
            rank=rank,
            robust_scale=Q,
            conv_strategy="max_frac_G",
            conv_tol=1e-4,
            init_strategy="svd",
            rotation="fast",
            target="G",
            whiten=True,
        )
        state, loss = model.fit(
            Y,
            W,
            max_iter=max_iter,
            conv_check_cadence=5,
        )
        save_state_to_npz(state, state_file)
        n_trained += 1

    print(f"Bin {i_bin}: trained {n_trained}, skipped {n_skipped} (already exist)")


def main(
    bins_to_run=BINS_TO_RUN,
    ranks=RANKS,
    q_vals=Q_VALS,
    max_iter=MAX_ITER,
    train_frac=TRAIN_FRAC,
    results_dir=RESULTS_DIR,
):
    """
    Train Robusta models on specified bins.

    Parameters
    ----------
    bins_to_run : list of int
        Indices of bins to train (0-13 for 14 bins).
    ranks : list of int
        Rank (K) values to try.
    q_vals : list of float
        Robustness scale (Q) values to try.
    max_iter : int
        Maximum iterations for training.
    train_frac : float
        Fraction of spectra to use for training (rest for test).
    results_dir : Path
        Directory to save results.

    Note: RNG_SEED and N_CLIP_PIX are imported from gaia_config.py.
    """
    print("Loading data and building bins...")
    data, bins = build_bins()

    print(f"\nBins to train: {bins_to_run}")
    print(f"Ranks: {ranks}")
    print(f"Q values: {q_vals}")
    print(f"Total models per bin: {len(ranks) * len(q_vals)}")

    for i_bin in bins_to_run:
        if i_bin < 0 or i_bin >= len(bins):
            print(f"Warning: bin index {i_bin} out of range (0-{len(bins) - 1}), skipping")
            continue
        train_bin(
            data,
            bins[i_bin],
            i_bin,
            ranks,
            q_vals,
            max_iter,
            train_frac,
            results_dir,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
