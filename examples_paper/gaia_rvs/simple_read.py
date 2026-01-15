from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from collect import MatchedData, compute_abs_mag

from robusta_hmf import Robusta

plt.style.use("mpl_drip.custom")
rng = np.random.default_rng(0)

# === #


# === LOAD DATA === #

data = MatchedData()

bp_rp = data["bp_rp"]
abs_mag_G = compute_abs_mag(data["phot_g_mean_mag"], data["parallax"])

# === #


# === SEGRAGATING ALONG MS === #

# Now, instead of having a target, we are going to form bins of spectra along the main sequence
# and then use RHMF on each bin to look for weird shit. But first we need to be able to segregate
# into bins along the MS and also to split into train/test each of those bins

# Let's do that kind of similarly to before, where we bin in ellipses centred on target values
# I think we'll make all ellipses the same size
# We'll also allow/want them to overlap a little? I think that's not a bad idea


@dataclass(frozen=True)
class BinProperties:
    centre: float
    width: float


@dataclass(frozen=True)
class Bin:
    bp_rp_prop: BinProperties
    abs_mag_G_prop: BinProperties
    bp_rp: np.ndarray
    abs_mag_G: np.ndarray
    idx: np.ndarray
    ids: np.ndarray

    @property
    def n_spectra(self):
        return len(self.ids)


def get_mask_ellipse(
    bp_rp_arr,
    abs_mag_G_arr,
    bp_rp_center,
    abs_mag_G_center,
    bp_rp_width,
    abs_mag_G_width,
):
    mask = ((bp_rp_arr - bp_rp_center) / (bp_rp_width / 2)) ** 2 + (
        (abs_mag_G_arr - abs_mag_G_center) / (abs_mag_G_width / 2)
    ) ** 2 < 1
    return mask


def find_indices(source_id, target_ids):
    return np.searchsorted(source_id, target_ids)


def build_bin(bp_rp_centre, bp_rp_width, abs_mag_G_centre, abs_mag_G_width):
    mask = get_mask_ellipse(
        bp_rp,
        abs_mag_G,
        bp_rp_centre,
        abs_mag_G_centre,
        bp_rp_width,
        abs_mag_G_width,
    )

    bp_rp_prop = BinProperties(centre=bp_rp_centre, width=bp_rp_width)
    abs_mag_G_prop = BinProperties(centre=abs_mag_G_centre, width=abs_mag_G_width)

    indices = np.where(mask)[0]

    if len(indices) > 0:
        return Bin(
            bp_rp_prop=bp_rp_prop,
            abs_mag_G_prop=abs_mag_G_prop,
            bp_rp=bp_rp[mask],
            abs_mag_G=abs_mag_G[mask],
            idx=indices,
            ids=data["source_id"][mask],
        )
    else:
        return Bin(
            bp_rp_prop=bp_rp_prop,
            abs_mag_G_prop=abs_mag_G_prop,
            bp_rp=np.array([]),
            abs_mag_G=np.array([]),
            idx=np.array([]),
            ids=np.array([]),
        )


def build_all_bins(bp_rp_bin_centres, abs_mag_G_bin_centres, bp_rp_width, abs_mag_G_width):
    bins = []
    for bp_rp_centre, abs_mag_G_centre in zip(bp_rp_bin_centres, abs_mag_G_bin_centres):
        b = build_bin(
            bp_rp_centre,
            bp_rp_width,
            abs_mag_G_centre,
            abs_mag_G_width,
        )
        bins.append(b)
    return bins


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
    bp_rp_bin_centres,
    abs_mag_G_bin_centres,
    bp_rp_width,
    abs_mag_G_width,
)

# Setup CMD plot
fig, ax = plt.subplots(dpi=100, figsize=[14, 7], layout="compressed")

# Plot all sources
ax.scatter(bp_rp, abs_mag_G, s=0.1, alpha=0.05, c="grey", zorder=0)

cols = plt.cm.viridis(np.linspace(0, 1, len(bins)))

# Plot bins
for i, b in enumerate(bins):
    if b.n_spectra > 0:
        ax.scatter(
            b.bp_rp,
            b.abs_mag_G,
            color=cols[i],
            s=5,
            alpha=0.01,
            label=f"Bin centre: ({b.bp_rp_prop.centre:.2f}, {b.abs_mag_G_prop.centre:.2f}) | N={b.n_spectra}",
        )

# Reverse the y-axis
ax.set_ylim(ax.get_ylim()[::-1])

ax.set_ylim(15, -5)
ax.set_xlim(-0.5, 3.5)
ax.set_xlabel("BP - RP")
ax.set_ylabel("Abs. Mag. G")
plt.show()


# Plot how many spectra in each bin
# X-axis should be (bp_rp_centre, abs_mag_G_centre) labelled
fig, ax = plt.subplots(dpi=100, figsize=[10, 5], layout="compressed")
n_spectra_in_bins = [b.n_spectra for b in bins]
bin_labels = [f"({b.bp_rp_prop.centre:.2f}, {b.abs_mag_G_prop.centre:.2f})" for b in bins]
ax.bar(bin_labels, n_spectra_in_bins)
ax.set_yscale("log")
ax.set_ylabel("Number of spectra in bin")
ax.set_xlabel("(BP - RP centre, Abs. Mag. G centre)")
ax.set_xticklabels(bin_labels, rotation=45, ha="right")
plt.show()
