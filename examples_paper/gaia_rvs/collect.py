# collect.py
# Get the stuff and things

from pathlib import Path

import h5py as h5
import numpy as np
import polars as pl

# Get the files, check existence
DATA_LOC = Path(".")
SPECTRA = DATA_LOC / "gaia-dr3-rvs-all.hdf5"
META = DATA_LOC / "gaia-dr3-source-meta.csv"
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


def load_matched_metadata(filter_nans=True, filter_neg_parallax=True):
    """Join to get matched metadata + HDF5 indices, without loading flux.

    The two filters exist for the colour-magnitude binning, which needs a
    finite BP-RP and a positive parallax to place a star on the HR diagram;
    they drop 5735 of the 999645 spectra. Anything fitting spectra directly
    should pass ``filter_nans=False, filter_neg_parallax=False`` and keep the
    whole catalogue: source_id is 1:1 between the CSV and the HDF5, so with
    both filters off the join returns every spectrum, with null metadata where
    the CSV has none.
    """
    df_meta = read_meta(filter_nans=filter_nans, filter_neg_parallax=filter_neg_parallax)
    df_spectra = read_spectra_ids()

    df_matched = df_spectra.join(df_meta, on="source_id", how="inner")
    return df_matched


class MatchedData:
    """Lazy access to matched spectra + metadata."""

    def __init__(self, filter_nans=True, filter_neg_parallax=True):
        self.df = load_matched_metadata(
            filter_nans=filter_nans, filter_neg_parallax=filter_neg_parallax
        )
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

    def get_flux_batch(self, indices, block=4096):
        """Get flux for multiple indices - sorts for faster HDF5 access.

        Reads in sorted, contiguous slabs rather than handing the whole index
        list to h5py. h5py builds a fancy selection by unioning one hyperslab
        per element, which is quadratic in len(indices): ~34 s for 3.2e4 rows
        and hours for the ~5e5 rows of the full RVS sample. Slab reads keep it
        linear and let HDF5 decompress each chunk exactly once.
        """
        hdf5_indices = self.spectra_indices[indices]
        order = np.argsort(hdf5_indices)
        sorted_indices = hdf5_indices[order]

        d_flux, d_flux_error = self.f_spec["flux"], self.f_spec["flux_error"]
        n_pix = d_flux.dtype.shape[0]
        flux = np.empty((len(sorted_indices), n_pix), dtype=d_flux.dtype.base)
        flux_error = np.empty_like(flux)

        for start in range(0, len(sorted_indices), block):
            sel = sorted_indices[start : start + block]
            lo, hi = int(sel[0]), int(sel[-1]) + 1
            stop = start + len(sel)
            if hi - lo > 4 * len(sel):
                # Sparse span: a slab read would waste most of what it decompresses,
                # and the list is short enough that fancy indexing is still cheap.
                flux[start:stop] = d_flux[sel.tolist()]
                flux_error[start:stop] = d_flux_error[sel.tolist()]
            else:
                local = sel - lo
                flux[start:stop] = d_flux[lo:hi][local]
                flux_error[start:stop] = d_flux_error[lo:hi][local]

        # Restore original order
        inv_order = np.argsort(order)
        return flux[inv_order], flux_error[inv_order]

    def __getitem__(self, col):
        return self.df[col].to_numpy()

    def close(self):
        if self._f_spec:
            self._f_spec.close()


def compute_abs_mag(phot_g_mean_mag, parallax):
    # Unfiltered samples carry null/non-positive parallaxes, which are NaN here
    # rather than an error: the HR plots mask on isfinite.
    with np.errstate(divide="ignore", invalid="ignore"):
        return phot_g_mean_mag + 5 * np.log10(parallax / 1000) + 5
