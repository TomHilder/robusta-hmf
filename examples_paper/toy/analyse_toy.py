from pathlib import Path

import matplotlib.pyplot as plt
import mpl_drip
import numpy as np
from chex import dataclass
from numpy.random import default_rng

# Get the rank and Q vals from other script
from run_toy_gen_and_fits import N_SPECTRA, Q_VALS, RANKS
from tqdm import tqdm

from robusta_hmf import Robusta
from robusta_hmf.state import RHMFState, load_state_from_npz

rng = default_rng(202012345)
plt.style.use("mpl_drip.custom")

# Smaller and potentially shit results
# N_SPECTRA = 100
# M_PIXELS = 300
# Larger results
N_SPECTRA = 4000
M_PIXELS = 1200

# Load the data itself
results_dir = Path("./toy_model_results")
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


# === Plots of the toy spectra (NOT outlier spectra) and reconstructions for some Q, K === #

# spectra = data["clean_spectra"]
noisy_spectra = data["train_spectra"]
grid = data["grid"]

plot_Q = 4
plot_K = 5

result_ind = np.where(
    (np.array([r.Q for r in results]) == plot_Q) & (np.array([r.K for r in results]) == plot_K)
)[0][0]
plot_rhmf: Robusta = rhmf_objs[result_ind]

plot_inds = rng.choice(noisy_spectra.shape[0], size=5, replace=False)

predictions = plot_rhmf.synthesize(indices=plot_inds)
predictions[3, :]

fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
for i, i_off in zip(plot_inds, range(5)):
    # ax.plot(grid / 10, spectra[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=2)
    ax.plot(grid / 10, noisy_spectra[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=2)
    ax.plot(grid / 10, predictions[i_off, :] + i_off * 1.0, color="k", alpha=1, lw=0.5)
    # ax.plot(grid / 10, predictions[i_off, :], color="k", alpha=1, lw=0.5)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
plt.savefig("spectra_and_reconstructions.pdf", bbox_inches="tight")
plt.show()

# === Plot of the inferred basis functions for some Q, K === #

basis = plot_rhmf.basis_vectors()
basis.shape

fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
for k in range(basis.shape[1]):
    ax.plot(grid / 10, basis[:, k] + k * 0.2, color=f"C{k}", alpha=1.0, lw=1)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Flux + offset")
plt.savefig("basis_functions.pdf", bbox_inches="tight")
plt.show()

# === Plot histogram of robust weights grouped by outlier type === #

ivar = data["train_ivar"]
outlier_mask = data["total_outlier_mask"][: noisy_spectra.shape[0]]
os_mask = data["os_mask"][: noisy_spectra.shape[0]]
op_mask = data["op_mask"][: noisy_spectra.shape[0]]
oc_mask = data["oc_mask"][: noisy_spectra.shape[0]]
al_mask = data["al_mask"][: noisy_spectra.shape[0]]

weights = plot_rhmf.robust_weights(noisy_spectra, ivar)

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
# ax.set_yscale("log")
ax.legend()
plt.savefig("robust_weights_histogram.pdf", bbox_inches="tight")
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

# Mask of all pixels with absorption lines
al_mask = data["al_mask"][: noisy_spectra.shape[0]]

# Find the spectra idx with absorption lines
al_spectra_idx = np.where(np.any(al_mask, axis=1))[0]
al_mask_al_spectra = al_mask[al_spectra_idx, :]

# Split grid into chunks where absorption lines are present
al_line_chunks = split_by_near_uniform(grid[al_mask_al_spectra[0]], factor=2.0)

# Mask of outlier pixels
op_mask = data["op_mask"][: noisy_spectra.shape[0]]
op_mask_al_spectra = op_mask[al_spectra_idx, :]
# Get the residuals for these spectra
residuals = plot_rhmf.A[al_spectra_idx, :] @ plot_rhmf.G.T - noisy_spectra[al_spectra_idx, :]
# residuals /= np.sqrt(1.0 / data["train_ivar"][al_spectra_idx, :])

# Mask of outlier columns
oc_mask = data["oc_mask"][: noisy_spectra.shape[0]]
oc_mask_al_spectra = oc_mask[al_spectra_idx, :]

# Plot the residuals for all these spectra with offset trick again
# Plot grey bands where the absorption lines are
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
# for i, i_off in zip(al_spectra_idx, range(len(al_spectra_idx))):
ax.plot(grid, residuals[0, :], color=f"C{0}", alpha=1.0, lw=2)

# ax.vlines(grid[al_mask_al_spectra[0]], ymin=-1, ymax=1, color="C1", alpha=0.3, lw=4, zorder=-2)
ax.fill_betweenx(
    y=[-1, 1],
    x1=[al_line_chunks[0].min()],
    x2=[al_line_chunks[0].max()],
    color="C1",
    alpha=0.3,
    zorder=-2,
)
ax.fill_betweenx(
    y=[-1, 1],
    x1=[al_line_chunks[1].min()],
    x2=[al_line_chunks[1].max()],
    color="C1",
    alpha=0.3,
    zorder=-2,
)
ax.vlines(grid[op_mask_al_spectra[0]], ymin=-1, ymax=1, color="C2", alpha=0.3, lw=4, zorder=-2)
ax.vlines(grid[oc_mask_al_spectra[0]], ymin=-1, ymax=1, color="C3", alpha=0.3, lw=4, zorder=-2)

ax.set_ylim(-1, 1)

ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Residual")
ax.set_title("Outlier Spectrum with Extra Absorption Lines and Pixel Outliers")
plt.savefig("absorption_line_residuals.pdf", bbox_inches="tight")
plt.show()

# === Heatmap of the robust weights for === #

plt.figure(figsize=(12, 6), dpi=100)
plt.imshow(weights, aspect="auto", origin="lower", interpolation="nearest")
plt.colorbar(label="Robust Weights")
plt.xlabel("Pixel Index")
plt.ylabel("Spectrum Index")
plt.title("Heatmap of Robust Weights for All Spectra and Pixels")
plt.savefig("robust_weights_heatmap.pdf", bbox_inches="tight")
plt.show()

# === The outlier spectra themselves and their reconstructions === #

weird_spectra_idx = np.where(np.any(os_mask, axis=1))[0]
predictions_weird = plot_rhmf.synthesize(indices=weird_spectra_idx)

fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
for i, i_off in zip(weird_spectra_idx[:5], range(5)):
    ax.plot(grid, noisy_spectra[i, :] + i_off * 1.0, color=f"C{i_off}", alpha=1.0, lw=1)
    ax.plot(grid, predictions_weird[i_off, :] + i_off * 1.0, color="k", alpha=1, lw=0.5)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux + offset")
plt.savefig("weird_outlier_spectra_and_reconstructions.pdf", bbox_inches="tight")
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


# Calculate median z-score for each model
scores = []
for rhmf, state in zip(rhmf_objs, test_states):
    residuals = rhmf.residuals(Y=test_spectra, state=state)
    robust_weights = rhmf.robust_weights(test_spectra, test_ivar, state=state)
    z_scores = residuals * np.sqrt(test_ivar) * np.sqrt(robust_weights)
    # z_scores = residuals * np.sqrt(test_ivar)
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
# override the xtick labels
plt.gca().set_xticklabels([str(q) for q in Q_VALS])
plt.yticks(RANKS)
plt.colorbar(label="Score")
plt.xlabel("Robust Scale Q")
plt.ylabel("Rank K")
plt.savefig("test_set_score_heatmap.pdf", bbox_inches="tight")
plt.show()

# === Scatter plot of coefficients for fitted and inferred coefficients for some model === #
# Reuse the existing plot_Q, plot_K
# Specifically we are going to make a 5 x 5 grid of scatter plots, each showing the coefficients for two basis functions
# over all the spectra. Training spectra in blue, test spectra in orange.

# result_ind = np.where(
# (np.array([r.Q for r in results]) == plot_Q) & (np.array([r.K for r in results]) == plot_K)
# )[0][0]
# plot_rhmf: Robusta = rhmf_objs[result_ind]
train_state = plot_rhmf._state
test_state = test_states[result_ind]
train_coeffs = train_state.A
test_coeffs = test_state.A

# We'll also plot the coefficients from the outlier spectra in a different colour
os_mask_test = data["os_mask"][noisy_spectra.shape[0] :]
train_os_coeffs = train_coeffs[os_mask.any(axis=1), :]
train_clean_coeffs = train_coeffs[~os_mask.any(axis=1), :]
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
