import matplotlib.pyplot as plt
import mpl_drip
import numpy as np
from numpy.random import default_rng

rng = default_rng(0)
plt.style.use("mpl_drip.custom")


LINE_LIST = [4101.7, 4340.5, 4861.3, 5167.3, 5172.7, 5183.6, 5890.0, 5896.0]
# LLM says: (Balmer + Mg b + Na D-ish)


def wavelength_grid(M, lam_min=4000.0, lam_max=6000.0):
    lam = np.geomspace(lam_min, lam_max, M)  # uniform in log-lambda
    x = np.log(lam)
    dx = x[1] - x[0]
    return lam, x, dx


def _lsf_kernel_sigma_pix(sigma_pix, half_width=6):
    hw = max(1, int(np.ceil(half_width * sigma_pix)))
    t = np.arange(-hw, hw + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_pix) ** 2)
    k /= k.sum()
    return k


def convolve_rows(Y, k):
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, Y)


def build_spectral_basis(M, K_extra=0, R=None):
    """Return G_true (K x M) with continuum (1,z,z^2) + line template T0 (+ derivs if R is not None) + two masks."""
    lam, x, dx = wavelength_grid(M)
    # Continuum (Legendre on [-1,1])
    z = 2 * (x - x.min()) / (x.max() - x.min()) - 1.0
    L0 = np.ones_like(z)
    L1 = z
    L2 = 0.5 * (3 * z**2 - 1)

    # Line template
    ll = LINE_LIST
    T0 = np.zeros(M)
    for c in ll:
        T0 += -0.45 * np.exp(-0.5 * ((lam - c) / 1.4) ** 2)

    # Optional LSF
    if R is not None:
        sigma_lam = lam.mean() / R  # Δλ FWHM / λ ≈ 1/R
        sigma_loglam = sigma_lam / lam.mean()
        sigma_pix = sigma_loglam / dx
        k = _lsf_kernel_sigma_pix(sigma_pix)
        T0 = convolve_rows(T0[None, :], k)[0]

    G_rows = [L0, L1, L2, T0]

    # Optional little extras to keep K similar to your toy
    for j in range(K_extra):
        G_rows.append(np.sin(2 * np.pi * (j + 2) * (lam - lam.min()) / (lam.max() - lam.min())))
    G = np.stack(G_rows, axis=0)
    return G, lam


def gen_clean_spectra(N, M, R=None):
    # Random draws for coefficients
    coeffs = np.zeros((N, 3 + 1 + 1))  # continuum + lines + 1 extra
    coeffs[:, 0] = rng.normal(1.0, 0.1, size=N)  # continuum level
    coeffs[:, 1] = rng.exponential(0.2, size=N)  # slope
    coeffs[:, 2] = rng.normal(0.0, 0.1, size=N)  # curvature
    coeffs[:, 3] = rng.lognormal(0.0, 0.1, size=N)  # line strength
    coeffs[:, 4] = rng.normal(0.1, 0.1, size=N)  # extra
    # Build basis
    basis, grid = build_spectral_basis(M=M, K_extra=1, R=R)
    # Generate spectra
    spectra = coeffs @ basis
    return spectra, coeffs, basis, grid


# Function to generate heteroskedastic Gaussian noise
# We want:
# - a multiplier per spectrum
# - a wavelength-dependent term (e.g., higher noise at blue end)
# - some random variation per pixel
def generate_noise(
    grid,
    spectra,
    nominal_sigma=0.02,
    spectrum_scale=0.1,
    wavelength_scale=0.05,
    random_scale=0.01,
):
    N, M = spectra.shape
    spectrum_scale = rng.lognormal(1.0, spectrum_scale, size=N)
    spectrum_scale /= spectrum_scale.mean()
    wavelength_scale = np.exp(
        -0.5 * ((grid - grid.mean()) / ((grid.max() - grid.min()) / wavelength_scale)) ** 2
    )
    wavelength_scale /= wavelength_scale.max()
    random_scale = rng.lognormal(0.0, random_scale, size=(N, M))
    random_scale /= random_scale.mean()
    # Combine all terms
    total_sigma = (
        spectrum_scale[:, None] * wavelength_scale[None, :] * random_scale * nominal_sigma
    )
    ivar = 1.0 / (total_sigma**2)
    assert ivar.shape == spectra.shape
    return rng.normal(0.0, total_sigma), ivar


# Function to replace some spectra with a whole weird outlier spectrum
# Our weird spectra will be high frequency sinusoids with somewhat random frequency/amplitude
def spectrum_outliers(
    spectra,
    n_outliers=5,
    freq_range=(5, 20),
    amp_range=(0.05, 0.2),
):
    N, M = spectra.shape
    # Sample some frequencies and amplitudes
    frequencies = rng.uniform(freq_range[0], freq_range[1], size=n_outliers)
    amplitudes = rng.uniform(amp_range[0], amp_range[1], size=n_outliers)
    # Build weird spectra
    weird_spectra = (
        amplitudes[:, None]
        * np.sin(
            frequencies[:, None] * np.linspace(0, 2 * np.pi, M),
        )
        + 1.0
    )
    # Weird spectra indices
    weird_spectra_indices = rng.choice(N, size=n_outliers, replace=False)
    # Insert weird spectra
    spectra[weird_spectra_indices, :] = weird_spectra
    outlier_mask = np.zeros_like(spectra, dtype=bool)
    outlier_mask[weird_spectra_indices, :] = True
    return spectra, weird_spectra_indices, outlier_mask


# Function to replace some pixels with outliers that don't follow the noise model
# Our outlier pixels will be random spikes added to the spectra
# We will need to keep track of the indices of the outlier pixels for evaluation later
def pixel_outliers(
    spectra,
    n_outlier_pixels=50,
    outlier_amp_range=(0.5, 1.0),
):
    N, M = spectra.shape
    outlier_idx = []
    outlier_mask = np.zeros_like(spectra, dtype=bool)
    for _ in range(n_outlier_pixels):
        i_spectrum = rng.integers(0, N)
        i_pixel = rng.integers(0, M)
        outlier_amp = rng.uniform(outlier_amp_range[0], outlier_amp_range[1])
        spectra[i_spectrum, i_pixel] += outlier_amp * (rng.choice([-1, 1]))
        outlier_mask[i_spectrum, i_pixel] = True
        outlier_idx.append((i_spectrum, i_pixel))
    return spectra, outlier_idx, outlier_mask


# Function to add near-column-level outliers by adding spikes to some % of the spectra at the same pixel locations
def column_outliers(
    spectra,
    n_outlier_columns=5,
    outlier_amp_range=(0.5, 1.0),
    fraction=0.3,
):
    N, M = spectra.shape
    outlier_idx = []
    outlier_mask = np.zeros_like(spectra, dtype=bool)

    i_columns = rng.choice(M, size=n_outlier_columns, replace=False)

    for i_column in i_columns:
        n_affected_spectra = int(fraction * N)
        affected_spectra_indices = rng.choice(N, size=n_affected_spectra, replace=False)
        for i_spectrum in affected_spectra_indices:
            outlier_amp = rng.uniform(outlier_amp_range[0], outlier_amp_range[1])
            spectra[i_spectrum, i_column] += outlier_amp * (rng.choice([-1, 1]))
            outlier_idx.append((i_spectrum, i_column))
            outlier_mask[i_spectrum, i_column] = True
    return spectra, outlier_idx, outlier_mask


# Function to add absorption lines at random locations to random spectra
# with the locations not identical across the outlier spectra
def absorption_line_outliers(
    spectra,
    n_spectra=5,
    n_lines=3,
    line_amp_range=(0.1, 0.3),
    line_width=1.0,
):
    N, M = spectra.shape
    # Keep track of only where the actual pixels changed, not the whole spectra
    # Keep track of the pixels that changed, not just the line centers
    outlier_idx = []
    outlier_mask = np.zeros_like(spectra, dtype=bool)

    i_spectra = rng.choice(N, size=n_spectra, replace=False)
    for i_spectrum in i_spectra:
        line_centers = rng.integers(0, M, size=n_lines)
        for center in line_centers:
            line_amp = rng.uniform(line_amp_range[0], line_amp_range[1])
            line_profile = line_amp * np.exp(-0.5 * ((np.arange(M) - center) / line_width) ** 2)
            spectra[i_spectrum, :] -= line_profile
            # Record outlier pixels (where line_profile is significant)
            affected_pixels = np.where(line_profile > 0.01 * line_amp)[0]
            for pixel in affected_pixels:
                outlier_mask[i_spectrum, pixel] = True
                outlier_idx.append((i_spectrum, pixel))
    return spectra, outlier_idx, outlier_mask


def missing_segments(
    spectra,
    op_mask,
    al_mask,
    frac_spectra=0.1,
    n_segments_per_spectrum=1,
    min_length=5,
    max_length=50,
):
    """
    Inject missing data segments into spectra as NaN values.

    Missing segments are placed in contiguous regions that don't overlap with
    pixel outliers (op_mask) or absorption line outliers (al_mask).

    Parameters
    ----------
    spectra : ndarray (N, M)
        Input spectra array (modified in place).
    op_mask : ndarray (N, M)
        Boolean mask of pixel outliers to avoid.
    al_mask : ndarray (N, M)
        Boolean mask of absorption line outliers to avoid.
    frac_spectra : float
        Fraction of spectra to add missing segments to.
    n_segments_per_spectrum : int
        Number of missing segments per affected spectrum.
    min_length : int
        Minimum segment length in pixels.
    max_length : int
        Maximum segment length in pixels.

    Returns
    -------
    spectra : ndarray (N, M)
        Spectra with NaN values where data is missing.
    missing_mask : ndarray (N, M)
        Boolean mask indicating missing pixels.
    missing_segment_info : list
        List of (spectrum_idx, start_pixel, end_pixel) tuples.
    """
    N, M = spectra.shape
    n_affected = int(frac_spectra * N)

    missing_mask = np.zeros_like(spectra, dtype=bool)
    missing_segment_info = []

    # Combined mask of pixels to avoid
    avoid_mask = op_mask | al_mask

    # Select spectra to affect (can include outlier spectra)
    affected_spectra_idx = rng.choice(N, size=n_affected, replace=False)

    for i_spectrum in affected_spectra_idx:
        # Find valid regions for this spectrum (not in avoid_mask)
        valid_pixels = ~avoid_mask[i_spectrum, :]

        # Find contiguous valid regions
        # Pad with False to detect edges
        padded = np.concatenate([[False], valid_pixels, [False]])
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        # Filter regions that are large enough for a segment
        valid_regions = [
            (s, e) for s, e in zip(starts, ends) if (e - s) >= min_length
        ]

        if len(valid_regions) == 0:
            continue

        # Place n_segments_per_spectrum segments
        for _ in range(n_segments_per_spectrum):
            if len(valid_regions) == 0:
                break

            # Pick a random region
            region_idx = rng.integers(0, len(valid_regions))
            region_start, region_end = valid_regions[region_idx]
            region_length = region_end - region_start

            # Pick a random segment length (uniform)
            seg_length = rng.integers(min_length, min(max_length, region_length) + 1)

            # Pick a random start position within the region
            max_start = region_end - seg_length
            seg_start = rng.integers(region_start, max_start + 1)
            seg_end = seg_start + seg_length

            # Mark as missing
            spectra[i_spectrum, seg_start:seg_end] = np.nan
            missing_mask[i_spectrum, seg_start:seg_end] = True
            missing_segment_info.append((i_spectrum, seg_start, seg_end))

            # Update valid_regions to exclude the placed segment
            # Split the region if segment is in the middle, or shrink it
            new_regions = []
            for s, e in valid_regions:
                if s == region_start and e == region_end:
                    # This is the region we placed in - split it
                    if seg_start - s >= min_length:
                        new_regions.append((s, seg_start))
                    if e - seg_end >= min_length:
                        new_regions.append((seg_end, e))
                else:
                    new_regions.append((s, e))
            valid_regions = new_regions

    return spectra, missing_mask, missing_segment_info
