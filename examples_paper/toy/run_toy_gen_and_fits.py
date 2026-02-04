from pathlib import Path

import matplotlib.pyplot as plt
import mpl_drip
import numpy as np
from gen_data import (
    absorption_line_outliers,
    column_outliers,
    gen_clean_spectra,
    generate_noise,
    missing_segments,
    pixel_outliers,
    spectrum_outliers,
)
from matplotlib.colors import ListedColormap
from numpy.random import default_rng
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState


def save_state_to_npz(state: RHMFState, filepath: Path):
    np.savez(
        filepath,
        A=state.A,
        G=state.G,
        it=state.it,
    )


rng = default_rng(202012345)
plt.style.use("mpl_drip.custom")

# Configuration
N_SPECTRA = 4000
N_TRAIN = 3500
assert N_TRAIN < N_SPECTRA
N_TEST = N_SPECTRA - N_TRAIN
MAX_ITER = 1000

# Spectra grid and resolution
M_PIXELS = 1200
R_RESOLUTION = 5000

# Noise parameters
NOISE_NOMINAL_SIGMA = 0.2
NOISE_SPECTRUM_SCALE = 2.0
NOISE_WAVELENGTH_SCALE = 2.0
NOISE_RANDOM_SCALE = 0.06

# Outlier spectra parameters
OUTLIER_N = 20
OUTLIER_FREQ_RANGE = (10, 30)
OUTLIER_AMP_RANGE = (0.1, 0.3)

# Outlier pixels parameters
OUTLIER_PIXEL_FRACTION = 0.004
OUTLIER_PIXEL_AMP_RANGE = (0.25, 0.75)
N_OUTLIER_PIXELS = int(OUTLIER_PIXEL_FRACTION * N_SPECTRA * M_PIXELS)
print(
    f"Number of outlier pixels: {N_OUTLIER_PIXELS} out of {N_SPECTRA * M_PIXELS} total pixels for {OUTLIER_PIXEL_FRACTION * 100}%."
)

# Outlier column parameters
N_OUTLIER_COLUMNS = 3
OUTLIER_COLUMN_AMP_RANGE = (0.25, 0.45)
COLUMN_OUTLIER_COLUMN_FRACTION = 0.3

# Outlier absorption line parameters
N_AL_OUTLIER_SPECTRA = 10
N_AL_LINES_PER_SPECTRUM = 3
AL_OUTLIER_WIDTH = 2.0
AL_OUTLIER_AMP_RANGE = (0.3, 0.6)

# Missing segment parameters
MISSING_FRAC_SPECTRA = 0.1
MISSING_N_SEGMENTS_PER_SPECTRUM = 1
MISSING_MIN_LENGTH = 10
MISSING_MAX_LENGTH = 50

# Fit parameters
RANKS = [3, 4, 5, 6, 7]
Q_VALS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
# Q_VALS = [4.0, 5.0]

if __name__ == "__main__":
    # Clean spectra
    spectra, coeffs, basis, grid = gen_clean_spectra(
        N=N_SPECTRA,
        M=M_PIXELS,
        R=R_RESOLUTION,
    )
    # Outlier spectra
    spectra, weird_spectra_idx, os_mask = spectrum_outliers(
        spectra,
        n_outliers=OUTLIER_N,
        freq_range=OUTLIER_FREQ_RANGE,
        amp_range=OUTLIER_AMP_RANGE,
    )
    # Outlier columns
    spectra, outlier_column_idx, oc_mask = column_outliers(
        spectra,
        n_outlier_columns=N_OUTLIER_COLUMNS,
        outlier_amp_range=OUTLIER_COLUMN_AMP_RANGE,
        fraction=COLUMN_OUTLIER_COLUMN_FRACTION,
    )
    # Outlier pixels
    spectra, outlier_pixel_idx, op_mask = pixel_outliers(
        spectra,
        n_outlier_pixels=N_OUTLIER_PIXELS,
        outlier_amp_range=OUTLIER_PIXEL_AMP_RANGE,
    )
    # Absorption line outliers
    spectra, absorption_line_idx, al_mask = absorption_line_outliers(
        spectra,
        n_spectra=N_AL_OUTLIER_SPECTRA,
        n_lines=N_AL_LINES_PER_SPECTRUM,
        line_amp_range=AL_OUTLIER_AMP_RANGE,
        line_width=AL_OUTLIER_WIDTH,
    )
    # Total outlier mask, accounting for possible overlaps
    total_outlier_mask = op_mask | oc_mask | al_mask | os_mask

    # Plot the total outlier mask as a heatmap
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    im = ax.imshow(
        total_outlier_mask,
        aspect="auto",
        cmap="gray_r",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xlabel("Pixel Index")
    ax.set_ylabel("Spectrum Index")
    ax.set_title("Outlier Mask Heatmap")
    plt.colorbar(im, ax=ax, label="Outlier Presence")
    plt.show()

    # # Plot a heatmap with each kind of outlier in a different color
    # # get the first 4 default cycle colors
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # cmap = ListedColormap(
    #     [
    #         (0, 0, 0, 0),  # transparent for label 0
    #         colors[3],  # label 4 -> C3
    #         colors[1],  # label 2 -> C1
    #         colors[2],  # label 3 -> C2
    #         colors[0],  # label 1 -> C0
    #     ]
    # )
    # label_map = np.zeros_like(total_outlier_mask, dtype=int)
    # label_map[op_mask] = 1
    # label_map[oc_mask] = 2
    # label_map[al_mask] = 3
    # label_map[os_mask] = 4
    # fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    # im1 = ax.imshow(
    #     label_map,
    #     aspect="auto",
    #     cmap=cmap,
    #     origin="lower",
    #     interpolation="nearest",
    # )
    # ax.set_xlabel("Pixel Index")
    # ax.set_ylabel("Spectrum Index")
    # ax.set_title("Outlier Types Heatmap")
    # red_patch = plt.Line2D([0], [0], color="red", lw=4, label="Pixel Outliers")
    # green_patch = plt.Line2D([0], [0], color="green", lw=4, label="Column Outliers")
    # blue_patch = plt.Line2D([0], [0], color="blue", lw=4, label="Absorption Line Outliers")
    # purple_patch = plt.Line2D([0], [0], color="yellow", lw=4, label="Weird Spectra Outliers")
    # ax.legend(handles=[red_patch, green_patch, blue_patch, purple_patch], loc="upper right")
    # plt.show()

    # Add noise
    noise, ivar = generate_noise(
        grid,
        spectra,
        nominal_sigma=NOISE_NOMINAL_SIGMA,
        spectrum_scale=NOISE_SPECTRUM_SCALE,
        wavelength_scale=NOISE_WAVELENGTH_SCALE,
        random_scale=NOISE_RANDOM_SCALE,
    )
    noisy_spectra = spectra + noise

    # Inject missing data segments (NaN in spectra, ivar=0)
    noisy_spectra, missing_mask, missing_segment_info = missing_segments(
        noisy_spectra,
        op_mask,
        al_mask,
        frac_spectra=MISSING_FRAC_SPECTRA,
        n_segments_per_spectrum=MISSING_N_SEGMENTS_PER_SPECTRUM,
        min_length=MISSING_MIN_LENGTH,
        max_length=MISSING_MAX_LENGTH,
    )
    ivar[missing_mask] = 0.0
    print(
        f"Injected {len(missing_segment_info)} missing segments "
        f"in {int(MISSING_FRAC_SPECTRA * N_SPECTRA)} spectra."
    )

    # Plot 5 random spectra with a vertical offset for clarity
    # Top panel is noiselss, bottom panel is noisy
    fig, ax = plt.subplots(2, 1, figsize=(12, 16), dpi=100)
    for i, i_off in zip(rng.choice(spectra.shape[0], size=5, replace=False), range(5)):
        ax[0].plot(grid, spectra[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=1)
        ax[1].plot(grid, noisy_spectra[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=1)
    # ax[0].set_xlabel("Wavelength [Å]")
    # ax[0].set_ylabel("Flux + offset")
    # ax[0].set_title("Generated Clean Spectra")
    ax[1].set_xlabel("Wavelength [Å]")
    ax[1].set_ylabel("Flux + offset")
    # ax[1].set_title("Generated Noisy Spectra")
    plt.show()

    # Plot 5 of the weird outlier spectra
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    for i, i_off in zip(weird_spectra_idx[:5], range(5)):
        ax.plot(grid, spectra[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=1)
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Flux + offset")
    ax.set_title("Weird Outlier Spectra")
    plt.show()

    # Plot the basis functions with normalised variance and  vertical offsets
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    for k in range(basis.shape[0]):
        ax.plot(grid, basis[k, :] + k * 2.0, color=f"C{k}", alpha=1.0, lw=1)
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Basis Function + offset")
    ax.set_title("Spectral Basis Functions")
    plt.show()

    # Segregate into training and test sets
    # Replace NaN with 0 for fitting (ivar=0 means these are ignored anyway)
    train_spectra = np.nan_to_num(noisy_spectra[:N_TRAIN, :], nan=0.0)
    train_ivar = ivar[:N_TRAIN, :]
    test_spectra = np.nan_to_num(noisy_spectra[N_TRAIN:, :], nan=0.0)
    test_ivar = ivar[N_TRAIN:, :]

    # Create a grid over q and rank values
    Q_grid, Rank_grid = np.meshgrid(Q_VALS, RANKS)

    # Fit models over the grid
    states = []
    losses = []
    prev_rank = None
    for Q, rank in tqdm(zip(Q_grid.flatten(), Rank_grid.flatten())):
        print(f"Fitting model with rank={rank}, q={Q}")
        model = Robusta(
            rank=rank,
            robust_scale=Q,
            conv_strategy="max_frac_G",
            conv_tol=1e-2,
            init_strategy="svd",
            rotation="fast",
            target="G",
            whiten=True,
        )
        if rank == prev_rank:
            init_state = states[-1]
        else:
            init_state = None
        state, loss = model.fit(
            train_spectra,
            train_ivar,
            # init_state=init_state,
            init_state=None,  # Turn off for now, I think it makes things worse?
            max_iter=MAX_ITER,
            conv_check_cadence=1,
        )
        states.append(state)
        losses.append(loss)
        prev_rank = rank

    # Save everything to disk
    results_dir = Path("./toy_model_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the data
    data_file = results_dir / f"data_N{N_SPECTRA}_M{M_PIXELS}.npz"
    np.savez(
        data_file,
        # Spectra themselves
        grid=grid,
        clean_spectra=spectra,
        noisy_spectra=noisy_spectra,
        ivar=ivar,
        # True coefficients and basis
        true_coeffs=coeffs,
        true_basis=basis,
        # All the indiviudal outlier masks not the indices
        os_mask=os_mask,
        oc_mask=oc_mask,
        op_mask=op_mask,
        al_mask=al_mask,
        missing_mask=missing_mask,
        total_outlier_mask=total_outlier_mask,
        # Outlier indices
        weird_spectra_idx=weird_spectra_idx,
        outlier_column_idx=outlier_column_idx,
        outlier_pixel_idx=outlier_pixel_idx,
        absorption_line_idx=absorption_line_idx,
        missing_segment_info=np.array(missing_segment_info),
        # Train/test split
        train_spectra=train_spectra,
        train_ivar=train_ivar,
        test_spectra=test_spectra,
        test_ivar=test_ivar,
    )

    # Save all the states and losses and the rank/Q value for each
    for i, (Q, rank) in enumerate(zip(Q_grid.flatten(), Rank_grid.flatten())):
        state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_N{N_SPECTRA}_M{M_PIXELS}.npz"
        save_state_to_npz(states[i], state_file)
