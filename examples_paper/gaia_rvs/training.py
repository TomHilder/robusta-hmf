from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from bins import build_all_bins
from collect import MatchedData, compute_abs_mag

from robusta_hmf import Robusta

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)

data = MatchedData()

bp_rp = data["bp_rp"]
abs_mag_G = compute_abs_mag(data["phot_g_mean_mag"], data["parallax"])

# Number of bins
n_bins = 14

# Evenly spaced in BP - RP
bp_rp_min = -0.1
bp_rp_max = 3.0
bp_rp_bin_centres = np.linspace(bp_rp_min, bp_rp_max, n_bins)

# Evenly spaced in Abs. mag G band
abs_mag_G_min = 0
abs_mag_G_max = 11
abs_mag_G_bin_centres = np.linspace(abs_mag_G_min, abs_mag_G_max, n_bins)

abs_mag_G_offsets = np.exp(-0.5 * ((bp_rp_bin_centres - 1.5) / 0.6) ** 2) * 1.5
abs_mag_G_bin_centres += abs_mag_G_offsets

# Bin widths
bp_rp_width = (bp_rp_max - bp_rp_min) / (n_bins - 1) * 1.5
abs_mag_G_width = (abs_mag_G_max - abs_mag_G_min) / (n_bins - 1) * 2.8

# === Build and plot the bins === #

bins = build_all_bins(
    data,
    bp_rp,
    abs_mag_G,
    bp_rp_bin_centres,
    abs_mag_G_bin_centres,
    bp_rp_width,
    abs_mag_G_width,
)

i_bin = 5
print(f"Number of spectra in bin {i_bin}: {bins[i_bin].n_spectra}")

rng_seed = 42


def get_test_train_split_idx(n_spectra, n_train_frac, seed):
    indices = np.arange(n_spectra)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(n_spectra * n_train_frac)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return train_indices, test_indices


def clip_edge_pix(flux, u_flux, n_clip):
    if isinstance(n_clip, int):
        n_clip_l = n_clip_r = n_clip
    elif isinstance(n_clip, (list, tuple)) and len(n_clip) == 2:
        n_clip_l, n_clip_r = n_clip
    else:
        raise ValueError("n_clip must be an int or a tuple/list of two ints.")
    return flux[:, n_clip_l:-n_clip_r], u_flux[:, n_clip_l:-n_clip_r]


def nans_mask(arrs):
    nans = np.isnan(np.array(arrs))
    return np.logical_not(np.any(nans, axis=0))


train_idx, test_idx = get_test_train_split_idx(
    bins[i_bin].n_spectra,
    n_train_frac=0.8,
    seed=rng_seed,
)

n_clip_pix = 40

# Get the train spectra
train_flux, train_u_flux = clip_edge_pix(
    *data.get_flux_batch(bins[i_bin].idx[train_idx]), n_clip=n_clip_pix
)
train_weights = 1.0 / (train_u_flux**2)

Y, W = train_flux, train_weights
spec_nans_mask = nans_mask([Y, W])
Y[~spec_nans_mask] = np.nan
W[~spec_nans_mask] = np.nan
Y = np.nan_to_num(Y)
W = np.nan_to_num(W)

model = Robusta(
    rank=7,
    robust_scale=3,
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
    max_iter=100,
    conv_check_cadence=1,
)

plt.figure(figsize=(6, 4))
plt.plot(loss, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training")
plt.grid()
plt.tight_layout()
plt.show()

basis = model.basis_vectors(state).T
# Plot the basis functions
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
for k in range(basis.shape[0]):
    ax.plot(
        data.λ_grid[n_clip_pix:-n_clip_pix],
        basis[k, :] + k * 0.1,
        color=f"C{k}",
        alpha=1.0,
        lw=1,
    )
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Basis Function + offset")
ax.set_title("Spectral Basis Functions")
plt.show()

# Plot the weights
robust_weights = model.robust_weights(Y, W, state)
object_weights = np.median(robust_weights, axis=1)

# Histogram of object weights
plt.figure(figsize=(6, 4))
plt.hist(object_weights, bins=30, color="C0", alpha=0.7)
plt.yscale("log")
plt.xlabel("Mean Object Robust Weight")
plt.ylabel("Number of Spectra")
plt.title("Histogram of Object Robust Weights")
plt.grid()
plt.tight_layout()
plt.show()

# Get indices of the spectra with 5 lowest weights
weird_spectra_idx = np.argsort(object_weights)[:5]

# Plot 5 of the weird outlier spectra
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
for i, i_off in zip(weird_spectra_idx, range(5)):
    ax.plot(
        data.λ_grid[n_clip_pix:-n_clip_pix],
        Y[i, :] + i_off * 1.0,
        color=f"C{i_off}",
        alpha=1.0,
        lw=1,
    )
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux + offset")
ax.set_title("Weird Outlier Spectra")
plt.tight_layout()
plt.show()

# Heatmap of all the robust weights
plt.figure(figsize=(8, 6))
plt.imshow(
    robust_weights,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
plt.colorbar(label="Robust Weight")
plt.xlabel("Pixel Index")
plt.ylabel("Spectrum Index")
plt.title("Robust Weights Heatmap")
plt.tight_layout()
plt.show()
