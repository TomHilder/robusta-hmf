from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpl_drip  # noqa: F401
import numpy as np
from bins import build_all_bins
from collect import MatchedData, compute_abs_mag
from rvs_plot_utils import add_line_markers, load_linelists
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState, load_state_from_npz

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(42)

# === Configuration (must match training.py) === #

RANKS = [3, 4, 5, 6, 7]
Q_VALS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
i_bin = 12
n_clip_pix = 40
rng_seed = 42

# Output directory for plots
plots_dir = Path(f"./plots_bin_{i_bin}")
plots_dir.mkdir(parents=True, exist_ok=True)


# === Helper functions (from training.py) === #


def get_test_train_split_idx(n_spectra, n_train_frac, seed):
    indices = np.arange(n_spectra)
    rng_split = np.random.default_rng(seed)
    rng_split.shuffle(indices)
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


# === Results dataclass === #


@dataclass(frozen=True)
class Results:
    K: int
    Q: float
    state: RHMFState


# === Load data and rebuild bins === #

print("Loading data...")
data = MatchedData()

bp_rp = data["bp_rp"]
abs_mag_G = compute_abs_mag(data["phot_g_mean_mag"], data["parallax"])

# Bin parameters (from training.py)
n_bins = 14
bp_rp_min, bp_rp_max = -0.1, 3.0
abs_mag_G_min, abs_mag_G_max = 0, 11

bp_rp_bin_centres = np.linspace(bp_rp_min, bp_rp_max, n_bins)
abs_mag_G_bin_centres = np.linspace(abs_mag_G_min, abs_mag_G_max, n_bins)
abs_mag_G_offsets = np.exp(-0.5 * ((bp_rp_bin_centres - 1.5) / 0.6) ** 2) * 1.5
abs_mag_G_bin_centres += abs_mag_G_offsets

bp_rp_width = (bp_rp_max - bp_rp_min) / (n_bins - 1) * 1.5
abs_mag_G_width = (abs_mag_G_max - abs_mag_G_min) / (n_bins - 1) * 2.8

print("Building bins...")
bins = build_all_bins(
    data,
    bp_rp,
    abs_mag_G,
    bp_rp_bin_centres,
    abs_mag_G_bin_centres,
    bp_rp_width,
    abs_mag_G_width,
)

print(f"Bin {i_bin} has {bins[i_bin].n_spectra} spectra")

# === Get train/test split === #

train_idx, test_idx = get_test_train_split_idx(
    bins[i_bin].n_spectra,
    n_train_frac=0.8,
    seed=rng_seed,
)

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# Load train data
print("Loading train spectra...")
train_flux, train_u_flux = clip_edge_pix(
    *data.get_flux_batch(bins[i_bin].idx[train_idx]), n_clip=n_clip_pix
)
train_Y, train_W = prep_data(train_flux, train_u_flux)

# Load test data
print("Loading test spectra...")
test_flux, test_u_flux = clip_edge_pix(
    *data.get_flux_batch(bins[i_bin].idx[test_idx]), n_clip=n_clip_pix
)
test_Y, test_W = prep_data(test_flux, test_u_flux)

# Wavelength grid (clipped)
λ_grid = data.λ_grid[n_clip_pix:-n_clip_pix]

# === Load saved results === #

print("Loading saved states...")
results_dir = Path("./gaia_rvs_results")

Q_grid, Rank_grid = np.meshgrid(Q_VALS, RANKS)
Q_vals = Q_grid.flatten()
K_vals = Rank_grid.flatten()

results = []
for Q, rank in tqdm(zip(Q_vals, K_vals), total=len(Q_vals)):
    state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_bin_{i_bin}.npz"
    state = load_state_from_npz(state_file)
    results.append(Results(K=rank, Q=Q, state=state))

# Create Robusta objects with loaded states
rhmf_objs = [Robusta(rank=r.K, robust_scale=r.Q) for r in results]
for obj, res in zip(rhmf_objs, results):
    obj._state = res.state

# === ANALYSIS === #

# Pick a specific (Q, K) for single-model plots
plot_Q = 5.0
plot_K = 3

result_ind = np.where(
    (np.array([r.Q for r in results]) == plot_Q) & (np.array([r.K for r in results]) == plot_K)
)[0][0]
plot_rhmf: Robusta = rhmf_objs[result_ind]

# === Plot: Spectra and reconstructions === #

print("Plotting spectra and reconstructions...")
plot_inds = rng.choice(train_Y.shape[0], size=5, replace=False)
predictions = plot_rhmf.synthesize(indices=jnp.array(plot_inds))

fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
for i, i_off in zip(plot_inds, range(5)):
    ax.plot(λ_grid, train_Y[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=1)
    ax.plot(λ_grid, predictions[i_off, :] + i_off * 1.0, color="k", alpha=1, lw=0.5)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_title(f"Spectra and Reconstructions (Q={plot_Q}, K={plot_K})")
plt.savefig(plots_dir / "spectra_and_reconstructions.pdf", bbox_inches="tight")
plt.show()

# === Plot: Basis functions === #

print("Plotting basis functions...")
basis = plot_rhmf.basis_vectors()

fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
for k in range(basis.shape[1]):
    ax.plot(λ_grid, basis[:, k] + k * 0.1, color=f"C{k}", alpha=1.0, lw=1)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_title(f"Basis Functions (Q={plot_Q}, K={plot_K})")
plt.savefig(plots_dir / "basis_functions.pdf", bbox_inches="tight")
plt.show()

# === Plot: Robust weights histogram === #

all_Y = np.concatenate([train_Y, test_Y])
all_W = np.concatenate([train_W, test_W])
plot_test_state, _ = plot_rhmf.infer(
    Y_infer=all_Y,
    W_infer=all_W,
    max_iter=1000,
    conv_tol=1e-4,
    conv_check_cadence=5,
)
plot_rhmf_all = plot_rhmf.set_state(plot_test_state)

print("Plotting robust weights histogram...")

# weights = plot_rhmf.robust_weights(train_Y, train_W)
weights = plot_rhmf_all.robust_weights(all_Y, all_W)

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.hist(weights.flatten(), bins=50, alpha=0.7, color="C0", density=False)
ax.set_xlabel("Robust Weight")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.set_title(f"Robust Weights Histogram (Q={plot_Q}, K={plot_K})")
plt.savefig(plots_dir / "robust_weights_histogram.pdf", bbox_inches="tight")
plt.show()

# === Plot: Robust weights heatmap === #

print("Plotting robust weights heatmap...")
plt.figure(figsize=(12, 6), dpi=100)
plt.imshow(weights, aspect="auto", origin="lower", interpolation="nearest")
plt.colorbar(label="Robust Weights")
plt.xlabel("Pixel Index")
plt.ylabel("Spectrum Index")
plt.title(f"Robust Weights Heatmap (Q={plot_Q}, K={plot_K})")
plt.savefig(plots_dir / "robust_weights_heatmap.pdf", bbox_inches="tight")
plt.show()

# === Plot: Lowest-weight spectra (potential outliers) === #

print("Plotting potential outlier spectra...")
object_weights = np.median(weights, axis=1)
n_weird = 5

weird_spectra_idx = np.argsort(object_weights)[:n_weird]
normal_spectra_idx = rng.choice(
    np.setdiff1d(np.arange(all_Y.shape[0]), weird_spectra_idx),
    size=n_weird,
    replace=False,
)

# predictions_weird = plot_rhmf.synthesize(indices=jnp.array(weird_spectra_idx))
# predictions_weird = plot_rhmf_all.synthesize(indices=jnp.array(weird_spectra_idx))
predictions_all = plot_rhmf_all.synthesize()

fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

for i, (idx, i_off) in enumerate(zip(weird_spectra_idx, range(n_weird))):
    # ax.plot(λ_grid, train_Y[idx, :] + i_off * 1.0, color="C1", alpha=1.0, lw=1)
    ax.plot(λ_grid, all_Y[idx, :] + i_off * 1.0, color="C1", alpha=1.0, lw=1)
    # ax.plot(λ_grid, predictions_weird[i, :] + i_off * 1.0, color="k", alpha=1, lw=0.5)
    ax.plot(λ_grid, predictions_all[idx, :] + i_off * 1.0, color="k", alpha=1, lw=0.5)

for i, (idx, i_off) in enumerate(zip(normal_spectra_idx, range(n_weird, 2 * n_weird))):
    ax.plot(λ_grid, all_Y[idx, :] + i_off * 1.0, color="C0", alpha=1.0, lw=1)

add_line_markers(ax=ax, lines=load_linelists())
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_title(f"Low-weight (orange) vs Normal (blue) Spectra (Q={plot_Q}, K={plot_K})")
plt.savefig(plots_dir / "outlier_spectra.pdf", bbox_inches="tight")
plt.show()

# Plot of just the residuals
# Pick just one weird spec for now
print(f"{len(weird_spectra_idx)} weird spectra total (this was set by hand in case you forgot)")
weird_idx_i = weird_spectra_idx[0]
# weird_idx_i = 1067  # Sanity check with random ass spectrum
residual_i = all_Y[weird_idx_i, :] - predictions_all[weird_idx_i, :]

fig, ax = plt.subplots(2, 1, figsize=[12, 8], dpi=300)

# Top panel: spectrum and reconstruction
ax[0].plot(λ_grid, all_Y[weird_idx_i, :], c="k", lw=1, label="Observed")
ax[0].plot(
    λ_grid, predictions_all[weird_idx_i, :], c="C2", lw=1, label="Reconstruction", alpha=0.8
)
ax[0].set_ylabel("Normalised flux")

ax[1].plot(λ_grid, residual_i, c="k", lw=0.5)

ax[0].legend()

add_line_markers(
    ax=ax[1],
    show_strong=True,
    show_abundance=False,
    show_cn=False,
    show_dib=False,
    lines=load_linelists(),
    label_fontsize=14,
)
# Optionally plot the robust weights per-point too
# ax[-1].plot(λ_grid, weights[weird_idx_i], c="r", lw=0.5)
ax[-1].set_xlabel("Wavelength [nm]")
ax[1].set_ylabel("Residual normalised flux")
plt.show()


# === CV: Infer on test set for all models === #

print("Inferring on test set for all models...")
test_states = []

for rhmf in tqdm(rhmf_objs):
    test_set_state, _ = rhmf.infer(
        Y_infer=test_Y,
        W_infer=test_W,
        max_iter=1000,
        conv_tol=1e-4,
        conv_check_cadence=5,
    )
    test_states.append(test_set_state)

# === Plot: Test set score heatmap === #

print("Computing CV scores...")
scores = []
for rhmf, state in zip(rhmf_objs, test_states):
    residuals = rhmf.residuals(Y=test_Y, state=state)
    robust_weights = rhmf.robust_weights(test_Y, test_W, state=state)
    z_scores = residuals * np.sqrt(test_W) * np.sqrt(robust_weights)
    score = np.std(z_scores)
    scores.append(score)

scores = np.array(scores)
scores = scores.reshape(len(RANKS), len(Q_VALS))

plt.figure(figsize=(10, 6), dpi=100)
plt.pcolormesh(
    np.arange(len(Q_VALS)),
    RANKS,
    np.log(np.abs(scores - 1)),
    shading="auto",
    cmap="viridis",
)
plt.xticks(np.arange(len(Q_VALS)))
plt.gca().set_xticklabels([str(q) for q in Q_VALS])
plt.yticks(RANKS)
plt.colorbar(label="log(|score - 1|)")
plt.xlabel("Robust Scale Q")
plt.ylabel("Rank K")
plt.title(f"Test Set Calibration Score: std(z) (Bin {i_bin})")
plt.savefig(plots_dir / "test_set_score_heatmap.pdf", bbox_inches="tight")
plt.show()

# === Alternative metric 1: Reduced chi-squared === #
# chi^2_red = mean((residuals * sqrt(W))^2), should be ~1 for good fit

print("Computing reduced chi-squared...")
chi2_red_scores = []
for rhmf, state in zip(rhmf_objs, test_states):
    residuals = rhmf.residuals(Y=test_Y, state=state)
    chi2 = (residuals * np.sqrt(test_W)) ** 2
    chi2_red = np.mean(chi2)
    chi2_red_scores.append(chi2_red)

chi2_red_scores = np.array(chi2_red_scores).reshape(len(RANKS), len(Q_VALS))

plt.figure(figsize=(10, 6), dpi=100)
plt.pcolormesh(
    np.arange(len(Q_VALS)),
    RANKS,
    np.log(np.abs(chi2_red_scores - 1)),
    shading="auto",
    cmap="viridis",
)
plt.xticks(np.arange(len(Q_VALS)))
plt.gca().set_xticklabels([str(q) for q in Q_VALS])
plt.yticks(RANKS)
plt.colorbar(label="log(|chi2_red - 1|)")
plt.xlabel("Robust Scale Q")
plt.ylabel("Rank K")
plt.title(f"Test Set Calibration Score: Reduced Chi-Squared (Bin {i_bin})")
plt.savefig(plots_dir / "test_set_score_heatmap_chi2.pdf", bbox_inches="tight")
plt.show()

# === Alternative metric 2: Weighted RMSE === #
# RMSE = sqrt(mean(W * residuals^2))

print("Computing weighted RMSE...")
rmse_scores = []
for rhmf, state in zip(rhmf_objs, test_states):
    residuals = rhmf.residuals(Y=test_Y, state=state)
    wmse = np.mean(test_W * residuals**2)
    rmse = np.sqrt(wmse)
    rmse_scores.append(rmse)

rmse_scores = np.array(rmse_scores).reshape(len(RANKS), len(Q_VALS))

plt.figure(figsize=(10, 6), dpi=100)
plt.pcolormesh(
    np.arange(len(Q_VALS)),
    RANKS,
    rmse_scores,
    shading="auto",
    cmap="viridis",
)
plt.xticks(np.arange(len(Q_VALS)))
plt.gca().set_xticklabels([str(q) for q in Q_VALS])
plt.yticks(RANKS)
plt.colorbar(label="Weighted RMSE")
plt.xlabel("Robust Scale Q")
plt.ylabel("Rank K")
plt.title(f"Test Set Score: Weighted RMSE (Bin {i_bin})")
plt.savefig(plots_dir / "test_set_score_heatmap_rmse.pdf", bbox_inches="tight")
plt.show()

# === Alternative metric 3: Median absolute z-score === #
# For standard normal, median(|z|) ≈ 0.6745

print("Computing median absolute z-score...")
mad_z_scores = []
for rhmf, state in zip(rhmf_objs, test_states):
    residuals = rhmf.residuals(Y=test_Y, state=state)
    robust_weights = rhmf.robust_weights(test_Y, test_W, state=state)
    z_scores = residuals * np.sqrt(test_W) * np.sqrt(robust_weights)
    mad_z = np.median(np.abs(z_scores))
    mad_z_scores.append(mad_z)

mad_z_scores = np.array(mad_z_scores).reshape(len(RANKS), len(Q_VALS))
expected_mad = 0.6745  # median(|z|) for standard normal

plt.figure(figsize=(10, 6), dpi=100)
plt.pcolormesh(
    np.arange(len(Q_VALS)),
    RANKS,
    np.log(np.abs(mad_z_scores - expected_mad)),
    shading="auto",
    cmap="viridis",
)
plt.xticks(np.arange(len(Q_VALS)))
plt.gca().set_xticklabels([str(q) for q in Q_VALS])
plt.yticks(RANKS)
plt.colorbar(label=f"log(|median(|z|) - {expected_mad}|)")
plt.xlabel("Robust Scale Q")
plt.ylabel("Rank K")
plt.title(f"Test Set Calibration Score: Median |z| (Bin {i_bin})")
plt.savefig(plots_dir / "test_set_score_heatmap_mad.pdf", bbox_inches="tight")
plt.show()

# === Plot: Coefficient scatter plots === #

print("Plotting coefficient scatter plots...")
train_state = plot_rhmf._state
test_state = test_states[result_ind]
assert train_state is not None
train_coeffs = train_state.A
test_coeffs = test_state.A

fig, axes = plt.subplots(
    plot_rhmf.rank, plot_rhmf.rank, figsize=(15, 15), dpi=100, layout="compressed"
)
for i in range(plot_rhmf.rank):
    for j in range(plot_rhmf.rank):
        ax = axes[i, j]
        if i == j:
            ax.hist(
                np.concatenate([train_coeffs[:, i], test_coeffs[:, i]]),
                bins=30,
                color="gray",
                alpha=0.7,
                density=True,
            )
        else:
            ax.scatter(
                train_coeffs[:, j],
                train_coeffs[:, i],
                color="C0",
                alpha=0.5,
                label="Train",
                s=5,
            )
            ax.scatter(
                test_coeffs[:, j],
                test_coeffs[:, i],
                color="C1",
                alpha=0.5,
                label="Test",
                s=5,
            )
        if i < plot_rhmf.rank - 1:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])
        if i == plot_rhmf.rank - 1:
            ax.set_xlabel(f"Coeff {j}")
        if j == 0:
            ax.set_ylabel(f"Coeff {i}")

axes[0, plot_rhmf.rank - 1].legend(loc="upper right")
plt.suptitle(f"Coefficient Scatter Plots (Q={plot_Q}, K={plot_K}, Bin {i_bin})")
plt.savefig(plots_dir / "coefficient_scatter_plots.pdf", bbox_inches="tight")
plt.show()

print("Done!")
data.close()
