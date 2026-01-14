# collect.py
# Get the stuff and things

from pathlib import Path

import h5py as h5
import numpy as np
import polars as pl

# Get the files, check existence
DATA_LOC = Path("../../data")
SPECTRA = DATA_LOC / "dr3-rvs-all.hdf5"
META = DATA_LOC / "dr3-source-meta.csv"
assert SPECTRA.is_file()
assert META.is_file()

TARGET_ID = 4264967400467007232


def nans_mask(arrs):
    nans = np.isnan(np.array(arrs))
    return np.logical_not(np.any(nans, axis=0))


def read_meta(filter_nans=True, filter_neg_parallax=True):
    # Read the csv
    lf_meta = pl.scan_csv(META)
    # _ = [print(x) for x in lf_meta.columns]
    lf_meta_selected = lf_meta.select(
        [
            "source_id",
            "parallax",
            "bp_rp",
            "phot_g_mean_mag",
        ]
    )
    df_met = lf_meta_selected.collect()

    source_id = df_met["source_id"].to_numpy()
    bp_rp = df_met["bp_rp"].to_numpy()
    phot_g_mean_mag = df_met["phot_g_mean_mag"].to_numpy()
    parallax = df_met["parallax"].to_numpy()

    if filter_nans:
        any_nans = nans_mask([bp_rp, phot_g_mean_mag, parallax])
        source_id = source_id[any_nans]
        bp_rp = bp_rp[any_nans]
        phot_g_mean_mag = phot_g_mean_mag[any_nans]
        parallax = parallax[any_nans]

    if filter_neg_parallax:
        neg_parallaxes = parallax < 0
        source_id = source_id[~neg_parallaxes]
        bp_rp = bp_rp[~neg_parallaxes]
        phot_g_mean_mag = phot_g_mean_mag[~neg_parallaxes]
        parallax = parallax[~neg_parallaxes]

    return source_id, bp_rp, phot_g_mean_mag, parallax


def find_source_index(source_id, target_id=TARGET_ID):
    return np.where(source_id == target_id)[0][0]


def read_spectra():
    f_spec = h5.File(SPECTRA, "r")
    λ_grid = np.linspace(846, 870, 2401)  # nm
    return (
        λ_grid,
        f_spec["source_id"][:],
        f_spec["flux"],
        f_spec["flux_error"],
    )


def compute_abs_mag(phot_g_mean_mag, parallax):
    abs_mag_G = phot_g_mean_mag + 5 * np.log10(parallax / 1000) + 5
    return abs_mag_G
