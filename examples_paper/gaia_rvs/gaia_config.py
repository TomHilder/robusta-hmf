"""
Shared configuration for Gaia RVS analysis.

All scripts (train_bins.py, analyse_bins.py, plot_bins.py) should import from here
to ensure consistency in bin definitions and data processing parameters.
"""

# === Bin geometry (DO NOT CHANGE without retraining) === #

N_BINS = 14

# BP - RP color range
BP_RP_MIN = -0.1
BP_RP_MAX = 3.0

# Absolute magnitude G range
ABS_MAG_G_MIN = 0
ABS_MAG_G_MAX = 11

# Bin widths (as multiples of spacing)
BP_RP_WIDTH_FACTOR = 1.5
ABS_MAG_G_WIDTH_FACTOR = 2.8

# Gaussian offset parameters for following the main sequence
ABS_MAG_G_OFFSET_CENTER = 1.5
ABS_MAG_G_OFFSET_SIGMA = 0.6
ABS_MAG_G_OFFSET_AMPLITUDE = 1.5

# === Data processing parameters === #

N_CLIP_PIX = 40   # Edge pixels to clip from spectra
RNG_SEED = 42     # Random seed for train/test splits
TRAIN_FRAC = 0.5  # Fraction of spectra for training (rest for test)

# === Derived quantities (computed from above) === #

import numpy as np

def get_bin_centres():
    """Compute bin centres in (BP-RP, Abs Mag G) space."""
    bp_rp_bin_centres = np.linspace(BP_RP_MIN, BP_RP_MAX, N_BINS)
    abs_mag_G_bin_centres = np.linspace(ABS_MAG_G_MIN, ABS_MAG_G_MAX, N_BINS)

    # Apply Gaussian offset to follow main sequence
    abs_mag_G_offsets = (
        np.exp(-0.5 * ((bp_rp_bin_centres - ABS_MAG_G_OFFSET_CENTER) / ABS_MAG_G_OFFSET_SIGMA) ** 2)
        * ABS_MAG_G_OFFSET_AMPLITUDE
    )
    abs_mag_G_bin_centres += abs_mag_G_offsets

    return bp_rp_bin_centres, abs_mag_G_bin_centres


def get_bin_widths():
    """Compute bin widths in (BP-RP, Abs Mag G)."""
    bp_rp_width = (BP_RP_MAX - BP_RP_MIN) / (N_BINS - 1) * BP_RP_WIDTH_FACTOR
    abs_mag_G_width = (ABS_MAG_G_MAX - ABS_MAG_G_MIN) / (N_BINS - 1) * ABS_MAG_G_WIDTH_FACTOR
    return bp_rp_width, abs_mag_G_width
