from dataclasses import dataclass
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import mpl_drip
import numpy as np
import seaborn as sns
from numpy.random import default_rng
from r_pca import RobustPCA

# Get the rank and Q vals from other script
from run_toy_gen_and_fits import M_PIXELS, N_SPECTRA, N_TRAIN, Q_VALS, RANKS
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState, load_state_from_npz

rng = default_rng(99202012345)
plt.style.use("mpl_drip.custom")

# Output directory for plots (same directory as this script)
SCRIPT_DIR = Path(__file__).parent
PLOT_DIR = SCRIPT_DIR

PAPER_PLOTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "papers/robust-hmf/687e0587c45a59bcc4a3fe3e/documents/figs"
)
assert PAPER_PLOTS_DIR.exists(), "PAPER_PLOTS_DIR does not exist, please update the path."

# N_SPECTRA and M_PIXELS are imported from run_toy_gen_and_fits

# Load the data itself
results_dir = SCRIPT_DIR / "toy_model_results"
data_file = results_dir / f"data_N{N_SPECTRA}_M{M_PIXELS}.npz"
data = np.load(data_file)


# Dataclasses to keep our stuff
@dataclass(frozen=True)
class Results:
    N: int
    M: int
    K: int
    Q: float
    state: RHMFState


# Create a grid over q and rank values
Q_grid, Rank_grid = np.meshgrid(Q_VALS, RANKS)
Q_vals = Q_grid.flatten()
K_vals = Rank_grid.flatten()

# Load all the results
results = []

for i, (Q, rank) in enumerate(tqdm(zip(Q_vals, K_vals), total=len(Q_vals))):
    state_file = results_dir / f"converged_state_R{rank}_Q{Q:.2f}_N{N_SPECTRA}_M{M_PIXELS}.npz"
    state = load_state_from_npz(state_file)
    results.append(Results(N=N_SPECTRA, M=M_PIXELS, K=rank, Q=Q, state=state))

rhmf_objs = [Robusta(rank=r.K, robust_scale=r.Q) for r in results]
# Manually override the states of the objects with our results
for obj, res in zip(rhmf_objs, results):
    obj._state = res.state

# ==== ANALYSIS ==== #

# Load all spectra (train + test combined) for visualizations
# NaN replaced with 0 for inference, NaN preserved for plotting
all_noisy_spectra = data["noisy_spectra"]  # With NaN for plotting
# For RHMF, NaN values don't matter since ivar=0 there. For PCA/RPCA, use mean imputation.
_spectra_mean = 0.0  # np.nanmean(data["noisy_spectra"])
all_spectra_for_fit = np.nan_to_num(data["noisy_spectra"], nan=_spectra_mean)
all_ivar = data["ivar"]
grid = data["grid"]

# Full masks for all data (not just training portion)
outlier_mask = data["total_outlier_mask"]
os_mask = data["os_mask"]
op_mask = data["op_mask"]
oc_mask = data["oc_mask"]
al_mask = data["al_mask"]

# === Plots of the toy spectra (NOT outlier spectra) and reconstructions for some Q, K === #

plot_Q = 5
plot_K = 5

result_ind = np.where(
    (np.array([r.Q for r in results]) == plot_Q) & (np.array([r.K for r in results]) == plot_K)
)[0][0]
plot_rhmf: Robusta = rhmf_objs[result_ind]

# Infer on all data for visualizations
print("Inferring on all data for visualizations...")
all_state, _ = plot_rhmf.infer(
    Y_infer=all_spectra_for_fit,
    W_infer=all_ivar,
    max_iter=1000,
    conv_tol=1e-2,
    conv_check_cadence=1,
)

# Do a PCA for comparison
U, S, Vh = np.linalg.svd(all_spectra_for_fit, full_matrices=False)
V = Vh.T
pca_basis = V[:, :plot_K]

# Do Robust PCA for comparison
print("Running Robust PCA (this may take a while)...")
rpca = RobustPCA(all_spectra_for_fit)
rpca_L, rpca_S = rpca.fit(max_iter=500, iter_print=1, tol=1e-4)
print("Robust PCA complete.")

plot_inds = rng.choice(all_noisy_spectra.shape[0], size=5, replace=False)

predictions = plot_rhmf.synthesize(indices=plot_inds, state=all_state)

spec_plot = np.nan_to_num(all_noisy_spectra[plot_inds, :], nan=_spectra_mean)
pca_coeffs_plot = spec_plot @ pca_basis
pca_recon_plot = pca_coeffs_plot @ pca_basis.T

plt.figure()
for i in range(5):
    plt.plot(pca_recon_plot[i, :])
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=100)
for i, i_off in zip(plot_inds, range(5)):
    ax.plot(
        grid / 10,
        all_noisy_spectra[i, :] + i_off * 1.0,
        color="k",
        alpha=1.0,
        lw=2,
        label="Toy Data" if i_off == 0 else None,
    )
    ax.plot(
        grid / 10,
        predictions[i_off, :] + i_off * 1.0,
        color="tab:green",
        alpha=1,
        lw=2.0,
        label="RHMF Fit" if i_off == 0 else None,
        ls="-",
    )
    # ax.plot(grid / 10, pca_recon_plot[i_off, :] + i_off * 1.0, color="tab:red", alpha=1, lw=2.0)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_xlim(grid.min() / 10 - 3, grid.max() / 10 + 3)
fig.suptitle(
    r"$\textsf{\textbf{Toy Example: Random Spectra}}$",
    fontsize="24",
    c="dimgrey",
    y=1.015,
)
ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
plt.savefig(PLOT_DIR / "spectra_and_reconstructions.pdf", bbox_inches="tight")
plt.show()

# === Plot of the inferred basis functions for some Q, K === #

basis = plot_rhmf.basis_vectors(state=all_state)

# 5 colors from cmasher colormap
cols = cmr.get_sub_cmap("viridis", 0.15, 0.85, N=plot_rhmf.rank)

fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
for k in range(basis.shape[1]):
    # ax.plot(grid / 10, basis[:, k] + k * 0.4, color=cols.colors[k], alpha=1.0, lw=2)
    ax.plot(grid / 10, basis[:, k] + k * 0.4, color=f"C{k}", alpha=1.0, lw=2)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_xlim(grid.min() / 10 - 3, grid.max() / 10 + 3)
plt.savefig(PLOT_DIR / "basis_functions.pdf", bbox_inches="tight")
plt.show()

# === Plot of the Robust PCA basis functions === #

# Get RPCA basis from SVD of the low-rank matrix L
U_rpca, S_rpca, Vh_rpca = np.linalg.svd(rpca_L, full_matrices=False)
rpca_basis = Vh_rpca.T[:, :plot_K]

fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
for k in range(rpca_basis.shape[1]):
    ax.plot(grid / 10, rpca_basis[:, k] + k * 0.4, color=f"C{k}", alpha=1.0, lw=2)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_xlim(grid.min() / 10 - 3, grid.max() / 10 + 3)
plt.savefig(PLOT_DIR / "rpca_basis_functions.pdf", bbox_inches="tight")
plt.show()

# === Plot histogram of robust weights grouped by outlier type === #

# Compute weights on all data using the inferred state
weights = plot_rhmf.robust_weights(all_spectra_for_fit, all_ivar, state=all_state)

# Segregate weights by outlier type
clean_weights = weights[~outlier_mask]
os_weights = weights[os_mask]
op_weights = weights[op_mask]
oc_weights = weights[oc_mask]
al_weights = weights[al_mask]

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.hist(clean_weights, bins=50, alpha=0.7, label="Clean Pixels", color="C0", density=True)
ax.hist(os_weights, bins=50, alpha=0.7, label="Outlier Spectra", color="C1", density=True)
ax.hist(op_weights, bins=50, alpha=0.7, label="Outlier Pixels", color="C2", density=True)
ax.hist(oc_weights, bins=50, alpha=0.7, label="Outlier Columns", color="C3", density=True)
ax.hist(al_weights, bins=50, alpha=0.7, label="Outlier Lines", color="C4", density=True)
ax.set_xlabel("Robust Weight")
ax.set_ylabel("Density")
ax.legend()
plt.savefig(PLOT_DIR / "robust_weights_histogram.pdf", bbox_inches="tight")
plt.show()

# === Simple clean vs outlier histogram === #

# Combine all outlier pixels (total_outlier_mask already combines them)
all_outlier_weights = weights[outlier_mask]

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
bins = np.linspace(0, 1, 26)
ax.hist(clean_weights, bins=bins, alpha=0.7, label="Clean Pixels", color="C0", density=True)
ax.hist(
    all_outlier_weights, bins=bins, alpha=0.7, label="Outlier Pixels", color="C1", density=True
)
ax.set_xlabel("Robust Weight")
ax.set_ylabel("Density")
ax.legend()
plt.savefig(PLOT_DIR / "robust_weights_clean_vs_outlier.pdf", bbox_inches="tight")
plt.show()

# === Per-object vs per-pixel weights histogram (like Gaia example) === #

# Compute per-object weights (median per spectrum)
# per_object_weights = np.median(weights, axis=1)
per_object_weights = np.mean(weights, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

# Left: per-object weights (median per spectrum)
ax = axes[0]
ax.hist(per_object_weights, bins=50, alpha=0.7, color="C0", density=False)
ax.axvline(0.9, color="r", linestyle="--", lw=2, label="Threshold = 0.9")
ax.set_xlabel("Per-Object Weight (mean)")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.set_title(f"Per-Object Weights (K={plot_K}, Q={plot_Q})")
ax.legend()

# Right: per-pixel weights (all data points)
ax = axes[1]
ax.hist(weights.flatten(), bins=50, alpha=0.7, color="C1", density=False)
ax.axvline(0.5, color="r", linestyle="--", lw=2, label="Threshold = 0.5")
ax.set_xlabel("Per-Pixel Robust Weight")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.set_title(f"Per-Pixel Weights (K={plot_K}, Q={plot_Q})")
ax.legend()

plt.suptitle("Robust Weights Distribution", fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_DIR / "weights_per_object_vs_pixel.pdf", bbox_inches="tight")
plt.show()

# === Per-object vs per-pixel weights: clean vs outlier === #

# Use os_mask to identify outlier spectra (the weird sinusoidal ones)
outlier_spectra_mask = os_mask.any(axis=1)
clean_spectra_mask = ~outlier_spectra_mask

# Per-object weights split by clean/outlier spectra
per_object_weights_clean = per_object_weights[clean_spectra_mask]
per_object_weights_outlier = per_object_weights[outlier_spectra_mask]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

# Left: per-object weights - clean vs outlier spectra
ax = axes[0]
bins = np.linspace(0, 1, 50)
ax.hist(per_object_weights_clean, bins=bins, alpha=0.7, color="C0", label="Clean Spectra")
ax.hist(
    per_object_weights_outlier,
    bins=bins,
    alpha=0.7,
    color="C1",
    label="Outlier Spectra",
)
ax.set_yscale("log")
ax.set_xlabel("Per-Object Weight (mean)")
ax.set_ylabel("Density")
ax.set_title(f"Per-Object Weights (K={plot_K}, Q={plot_Q})")
ax.legend()

# Right: per-pixel weights - clean vs outlier pixels (excluding outlier spectra)
ax = axes[1]
# Exclude outlier spectra from per-pixel analysis
non_os_weights = weights[clean_spectra_mask, :]
non_os_outlier_mask = outlier_mask[clean_spectra_mask, :]
non_os_os_mask = os_mask[clean_spectra_mask, :]  # Should be all False
# Clean pixels: not in any outlier mask, from non-outlier spectra
non_os_clean_weights = non_os_weights[~non_os_outlier_mask]
# Outlier pixels: in outlier mask but not os_mask (pixel/column/line outliers only)
non_os_outlier_pixel_mask = non_os_outlier_mask & ~non_os_os_mask
non_os_outlier_weights = non_os_weights[non_os_outlier_pixel_mask]
ax.hist(non_os_clean_weights, bins=bins, alpha=0.7, color="C0", label="Clean Pixels")
ax.hist(non_os_outlier_weights, bins=bins, alpha=0.7, color="C1", label="Outlier Pixels")
ax.set_yscale("log")
ax.set_xlabel("Per-Pixel Robust Weight")
ax.set_ylabel("Density")
ax.set_title(f"Per-Pixel Weights (K={plot_K}, Q={plot_Q})")
ax.legend()

plt.tight_layout()
plt.savefig(PLOT_DIR / "weights_clean_vs_outlier_object_pixel.pdf", bbox_inches="tight")
plt.show()

# Just the left panel
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
bins = np.linspace(0, 1, 51)
ax.hist(per_object_weights_clean, bins=bins, alpha=0.7, color="C0", label="Normal Spectra")
ax.hist(
    per_object_weights_outlier,
    bins=bins,
    alpha=0.7,
    color="C1",
    label="Outlier Spectra",
)
ax.set_yscale("log")
ax.set_xlabel("Mean Robust Weight per Spectrum")
ax.set_ylabel("Count")
# ax.set_title(f"Per-Object Weights (K={plot_K}, Q={plot_Q})")
ax.legend(loc="upper left", borderaxespad=1)
fig.suptitle(
    r"$\textsf{\textbf{Toy Example: Object Weights}}$",
    fontsize="24",
    c="dimgrey",
    y=0.945,
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "weights_per_object_clean_vs_outlier.pdf", bbox_inches="tight")
# plt.savefig(PAPER_PLOTS_DIR / "weights_per_object_clean_vs_outlier.pdf", bbox_inches="tight")
plt.show()


# === Absorption lines === #


def split_by_near_uniform(x, *, factor=3.0, step=None, return_breaks=False):
    """
    Split 1D array x into subarrays where spacing is ~uniform.

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


# We injected absorption lines as outliers in some spectra. Let's see if we can recover them in the residuals.

# Find the spectra idx with absorption lines (using al_mask defined at top)
al_spectra_idx = np.where(np.any(al_mask, axis=1))[0]
al_mask_al_spectra = al_mask[al_spectra_idx, :]

i_al_spec = 0

# Split grid into chunks where absorption lines are present
al_line_chunks = split_by_near_uniform(grid[al_mask_al_spectra[i_al_spec]], factor=2.0)

# Get masks for this spectrum
op_mask_al_spectra = op_mask[al_spectra_idx, :]
oc_mask_al_spectra = oc_mask[al_spectra_idx, :]

# Get the residuals for these spectra using all_state
reconstructions_al = all_state.A[al_spectra_idx, :] @ all_state.G.T
residuals = all_noisy_spectra[al_spectra_idx, :] - reconstructions_al

robust_weights = weights[al_spectra_idx, :]
ivar_al = all_ivar[al_spectra_idx, :]
weighted_residuals = residuals / robust_weights**2

plt.figure()
plt.plot(weighted_residuals[i_al_spec, :])
plt.show()

spec_i = np.nan_to_num(all_noisy_spectra[al_spectra_idx, :][i_al_spec], nan=_spectra_mean)
pca_coeffs = spec_i @ pca_basis
pca_recon = pca_coeffs @ pca_basis.T

# Get Robust PCA reconstruction for this spectrum by projecting onto truncated basis
# (for fair comparison with PCA and RHMF which use rank-K)
rpca_coeffs = spec_i @ rpca_basis
rpca_recon = rpca_coeffs @ rpca_basis.T

plt.figure()
plt.plot(pca_recon)
plt.show()

# Plot the residuals for all these spectra with offset trick again
# Plot grey bands where the absorption lines are
fig, ax = plt.subplots(
    3, 1, figsize=(12, 8), dpi=100, sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
)
ax[0].plot(
    grid / 10,
    all_noisy_spectra[al_spectra_idx, :][i_al_spec],
    c="k",
    lw=2.0,
    zorder=7,
    label="Toy Data",
)
ax[0].plot(
    grid / 10, reconstructions_al[i_al_spec], c="tab:green", lw=2.0, zorder=10, label="RHMF Fit"
)
ax[0].plot(grid / 10, rpca_recon, c="tab:blue", lw=2.0, zorder=9, label="RPCA Fit")
ax[0].plot(grid / 10, pca_recon, c="tab:red", lw=2.0, zorder=8, label="PCA Fit")
ax[1].plot(grid / 10, residuals[i_al_spec, :], color="k", alpha=1.0, lw=2.0)
ax[2].plot(grid / 10, robust_weights[i_al_spec], color="k", alpha=1.0, lw=2.0)
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
            x1=[al_line_chunks[i].min() / 10 - 0.1],
            x2=[al_line_chunks[i].max() / 10 + 0.1],
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
        lw=5,
        zorder=-2,
        label="Outlier Column" if j == 1 else None,
    )
ax[-1].set_xlabel("Wavelength [nm]")
ax[1].set_ylabel("Residual \nFlux")
ax[2].set_ylabel("Robust \nWeight")
ax[0].legend(loc=(0.3, 0.6))
# Place this legend above the whole figure, centered
# ax[1].legend(
#     ncol=3,
#     loc="lower center",
#     bbox_to_anchor=(0.5, 1.02),
#     borderaxespad=0,
# )
# ax[0].legend(
#     *ax[1].get_legend_handles_labels(),
#     ncol=3,
#     loc="lower center",
#     bbox_to_anchor=(0.5, 1.02),
#     borderaxespad=0,
# )
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
plt.savefig(PLOT_DIR / "absorption_line_residuals.pdf", bbox_inches="tight")
plt.savefig(PAPER_PLOTS_DIR / "absorption_line_residuals.pdf", bbox_inches="tight")
plt.show()

# === Heatmap of the robust weights for === #

plt.figure(figsize=(12, 6), dpi=100)
plt.imshow(weights, aspect="auto", origin="lower", interpolation="nearest")
plt.colorbar(label="Robust Weights")
plt.xlabel("Pixel Index")
plt.ylabel("Spectrum Index")
plt.title("Heatmap of Robust Weights for All Spectra and Pixels")
plt.savefig(PLOT_DIR / "robust_weights_heatmap.pdf", bbox_inches="tight")
plt.show()

# === The outlier spectra themselves and their reconstructions === #

weird_spectra_idx = np.where(np.any(os_mask, axis=1))[0]
predictions_weird = plot_rhmf.synthesize(indices=weird_spectra_idx, state=all_state)

spec_weird = np.nan_to_num(all_noisy_spectra[weird_spectra_idx[:5], :], nan=_spectra_mean)
pca_coeffs_weird = spec_weird @ pca_basis
pca_recon_weird = pca_coeffs_weird @ pca_basis.T

fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
for i, i_off in zip(weird_spectra_idx[:5], range(5)):
    ax.plot(
        grid / 10,
        all_noisy_spectra[i, :] + i_off * 1.0,
        color="k",
        alpha=1.0,
        lw=2,
        label="Outlier Spectra" if i_off == 0 else None,
    )
    # ax.plot(grid / 10, pca_recon_weird[0, :] + i_off * 1.0, color="tab:red")
    ax.plot(
        grid / 10,
        predictions_weird[i_off, :] + i_off * 1.0,
        color="tab:green",
        alpha=1,
        lw=2.0,
        label="RHMF Fit" if i_off == 0 else None,
    )
# ax.set_xlabel("Wavelength [Å]")
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.set_xlim(grid.min() / 10 - 3, grid.max() / 10 + 3)
# plt.suptitle("Outlier Spectra and Reconstructions", fontsize=12)
fig.suptitle(r"$\textsf{\textbf{Toy Example: Outlier Spectra}}$", fontsize=24, c="dimgrey", y=1.01)
plt.savefig(PLOT_DIR / "weird_outlier_spectra_and_reconstructions.pdf", bbox_inches="tight")
plt.show()

# === Infer on the test set for each model and calculate an error metric of some kind to plot === #

test_spectra = data["test_spectra"]
test_ivar = data["test_ivar"]

test_states = []

for rhmf in rhmf_objs:
    test_set_state, _ = rhmf.infer(
        Y_infer=test_spectra,
        W_infer=test_ivar,
        max_iter=1000,
        conv_tol=1e-2,
        conv_check_cadence=1,
    )
    test_states.append(test_set_state)


# Calculate z-score metric and outlier detection metrics for each model
from sklearn.metrics import f1_score

# Get test set outlier masks (ground truth)
test_outlier_mask = data["total_outlier_mask"][N_TRAIN:]  # per-pixel
# Per-object: only count "spectrum outliers" (weird sinusoidal spectra), not every spectrum with a pixel outlier
test_os_mask = data["os_mask"][N_TRAIN:]
test_outlier_object = test_os_mask.any(axis=1)

scores = []
f1_pixel_scores = []
f1_object_scores = []
for rhmf, state in zip(rhmf_objs, test_states):
    residuals = rhmf.residuals(Y=test_spectra, state=state)
    robust_weights = rhmf.robust_weights(test_spectra, test_ivar, state=state)

    # Z-score metric (existing) - all pixels
    z_scores = residuals * np.sqrt(test_ivar) * np.sqrt(robust_weights)
    score = np.std(z_scores)
    scores.append(score)

    # Per-pixel F1 score: treat weight < 0.5 as "predicted outlier"
    y_true_pixel = test_outlier_mask.flatten().astype(int)
    y_pred_pixel = (robust_weights.flatten() < 0.5).astype(int)
    f1_pixel = f1_score(y_true_pixel, y_pred_pixel)
    f1_pixel_scores.append(f1_pixel)

    # Per-object F1 score: median weight per spectrum, threshold 0.9
    median_weights = np.median(robust_weights, axis=1)
    y_true_object = test_outlier_object.astype(int)
    y_pred_object = (median_weights < 0.9).astype(int)
    f1_object = f1_score(y_true_object, y_pred_object)
    f1_object_scores.append(f1_object)

scores = np.array(scores).reshape(len(RANKS), len(Q_VALS))
f1_pixel_scores = np.array(f1_pixel_scores).reshape(len(RANKS), len(Q_VALS))
f1_object_scores = np.array(f1_object_scores).reshape(len(RANKS), len(Q_VALS))

# Three-panel figure: CV score on top (centred), F1 scores on bottom row
# Use two separate gridspecs for independent layout control per row
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

# Tick labels
q_labels = [str(q) for q in Q_VALS]
rank_labels = [str(r) for r in RANKS]

top_cmap = "viridis"

text_bbox = dict(
    boxstyle="square",
    facecolor="white",
    alpha=0.7,
    edgecolor="none",
)
text_loc = (0.08, 0.85)  # relative to axes

# Top panel: CV score (z-score std)
im0 = ax_cv.imshow(
    np.log(np.abs(scores - 1)),
    origin="lower",
    cmap=top_cmap,
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
fig.colorbar(im0, cax=cax_cv, label="CV Score \n(Lower is Better)")

bottom_cmap = "magma_r"

# Bottom left: Per-pixel F1 score (threshold 0.5)
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

# Bottom right: Per-object F1 score (median weight, threshold 0.9)
im2 = ax_f1_obj.imshow(
    f1_object_scores,
    origin="lower",
    cmap=bottom_cmap,
    vmin=0,
    vmax=1,
    aspect="auto",
)
ax_f1_obj.set_xticks(np.arange(len(Q_VALS)), labels=q_labels)
# ax_f1_obj.set_yticks([])
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

# Shared colorbar for the two F1 panels
fig.colorbar(im1, cax=cax_f1, label="F1 Score \n(Higher is Better)")

fig.suptitle(
    r"$\textsf{\textbf{Toy Example: Hyperparameters}}$",
    fontsize="24",
    c="dimgrey",
    y=0.98,
)
plt.savefig(PLOT_DIR / "test_set_score_heatmap.pdf", bbox_inches="tight")
# plt.savefig(PAPER_PLOTS_DIR / "test_set_score_heatmap.pdf", bbox_inches="tight")
plt.show()

# Single panel CV plot
plt.figure(figsize=(8, 4.5), dpi=100)
im = plt.pcolormesh(
    np.arange(len(Q_VALS)),
    RANKS,
    np.log(np.abs(scores - 1)),
    shading="auto",
    cmap="viridis",
)
plt.xticks(np.arange(len(Q_VALS)), [str(q) for q in Q_VALS])
plt.yticks(RANKS)
plt.xlabel("Robust Scale Q")
plt.ylabel("Rank K")
plt.colorbar(im, label="Score")
plt.savefig(PLOT_DIR / "test_set_cv_score_heatmap.pdf", bbox_inches="tight")
plt.show()

# === Scatter plot of coefficients for fitted and inferred coefficients for some model === #
# Reuse the existing plot_Q, plot_K
# Specifically we are going to make a 5 x 5 grid of scatter plots, each showing the coefficients for two basis functions
# over all the spectra. Training spectra in blue, test spectra in orange.

# Use all_state to get coefficients for all spectra, then split by train/test
all_coeffs = all_state.A
train_coeffs = all_coeffs[:N_TRAIN, :]
test_coeffs = all_coeffs[N_TRAIN:, :]

# We'll also plot the coefficients from the outlier spectra in a different colour
os_mask_train = os_mask[:N_TRAIN, :]
os_mask_test = os_mask[N_TRAIN:, :]
train_os_coeffs = train_coeffs[os_mask_train.any(axis=1), :]
train_clean_coeffs = train_coeffs[~os_mask_train.any(axis=1), :]
test_os_coeffs = test_coeffs[os_mask_test.any(axis=1), :]
test_clean_coeffs = test_coeffs[~os_mask_test.any(axis=1), :]

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
            # ax.hist(
            #     np.concatenate([train_os_coeffs[:, i], test_os_coeffs[:, i]]),
            #     bins=3,
            #     color="red",
            #     alpha=0.7,
            #     density=True,
            # )
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
            # ax.scatter(
            #     train_os_coeffs[:, j],
            #     train_os_coeffs[:, i],
            #     color="red",
            #     alpha=0.7,
            #     label="Train Outlier",
            #     s=5,
            # )
            # ax.scatter(
            #     test_os_coeffs[:, j],
            #     test_os_coeffs[:, i],
            #     color="orange",
            #     alpha=0.7,
            #     label="Test Outlier",
            #     s=5,
            # )
        if i < plot_rhmf.rank - 1:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])
        if i == plot_rhmf.rank - 1:
            ax.set_xlabel(f"Coeff {j}")
        if j == 0:
            ax.set_ylabel(f"Coeff {i}")
# Only add legend to top-right plot
axes[0, plot_rhmf.rank - 1].legend(loc="upper right")
plt.suptitle(f"Coefficient Scatter Plots for Q={plot_Q}, K={plot_K}")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig("coefficient_scatter_plots.pdf", bbox_inches="tight")
plt.show()

# === Combined plot: normal spectra on top, outlier spectra on bottom === #

N_CLEAN_PLOT = 3
N_OUTLIER_PLOT = 3

# Randomly select from all data, filtering out very noisy spectra
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

data_lw = 2.0
fit_lw = 1.5

# Plot outlier spectra on bottom
for i_off, idx in enumerate(outlier_plot_idx):
    pred_i = N_CLEAN_PLOT + i_off  # index into combined_predictions
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
        label="RHMF Fits" if i_off == 0 else None,
    )
    offset += offset_step

# Small gap between clean and outlier groups
offset += offset_step * 0.0

# Plot clean spectra on top
for i_off, idx in enumerate(clean_plot_idx):
    pred_i = i_off  # index into combined_predictions
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
        lw=fit_lw,
    )
    offset += offset_step

ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
ax.set_xlim(grid.min() / 10 - 4, grid.max() / 10 + 4)
# Manual legend order: Clean, Outlier, RHMF
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
plt.savefig(PLOT_DIR / "normal_vs_outlier_spectra_reconstructions.pdf", bbox_inches="tight")
plt.savefig(PAPER_PLOTS_DIR / "normal_vs_outlier_spectra_reconstructions.pdf", bbox_inches="tight")
plt.show()
