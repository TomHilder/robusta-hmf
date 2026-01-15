# collect.py
# Get the stuff and things

from pathlib import Path

import h5py as h5
import numpy as np
import polars as pl

# Get the files, check existence
DATA_LOC = Path(".")
SPECTRA = DATA_LOC / "dr3-rvs-all.hdf5"
META = DATA_LOC / "dr3-source-meta.csv"
assert SPECTRA.is_file()
assert META.is_file()


def read_meta(filter_nans=True, filter_neg_parallax=True):
    """Read metadata CSV and return as a lazy Polars DataFrame."""
    lf_meta = pl.scan_csv(META).select(
        [
            "source_id",
            "parallax",
            "bp_rp",
            "phot_g_mean_mag",
        ]
    )

    if filter_nans:
        lf_meta = lf_meta.drop_nulls()

    if filter_neg_parallax:
        lf_meta = lf_meta.filter(pl.col("parallax") > 0)

    return lf_meta.collect()


def read_spectra_ids():
    """Just read source_ids from HDF5 - this should be fast (~24MB for 3M)."""
    with h5.File(SPECTRA, "r") as f:
        return pl.DataFrame(
            {
                "source_id": f["source_id"][:],
            }
        ).with_row_index("spectra_idx")


def load_matched_metadata():
    """Join to get matched metadata + HDF5 indices, without loading flux."""
    df_meta = read_meta()
    df_spectra = read_spectra_ids()

    df_matched = df_spectra.join(df_meta, on="source_id", how="inner")
    return df_matched


class MatchedData:
    """Lazy access to matched spectra + metadata."""

    def __init__(self):
        self.df = load_matched_metadata()
        self.spectra_indices = self.df["spectra_idx"].to_numpy()
        self.λ_grid = np.linspace(846, 870, 2401)
        self._f_spec = None

    @property
    def f_spec(self):
        if self._f_spec is None:
            self._f_spec = h5.File(SPECTRA, "r")
        return self._f_spec

    def get_flux(self, idx):
        """Get flux for a single matched index."""
        hdf5_idx = self.spectra_indices[idx]
        return self.f_spec["flux"][hdf5_idx], self.f_spec["flux_error"][hdf5_idx]

    def get_flux_batch(self, indices):
        """Get flux for multiple indices - sorts for faster HDF5 access."""
        hdf5_indices = self.spectra_indices[indices]
        order = np.argsort(hdf5_indices)
        sorted_indices = hdf5_indices[order].tolist()

        # Read in sorted order (much faster)
        flux = self.f_spec["flux"][sorted_indices]
        flux_error = self.f_spec["flux_error"][sorted_indices]

        # Restore original order
        inv_order = np.argsort(order)
        return flux[inv_order], flux_error[inv_order]

    def __getitem__(self, col):
        return self.df[col].to_numpy()

    def close(self):
        if self._f_spec:
            self._f_spec.close()


def compute_abs_mag(phot_g_mean_mag, parallax):
    return phot_g_mean_mag + 5 * np.log10(parallax / 1000) + 5
