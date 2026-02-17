"""
One-off script to generate all_source_ids.npy for each bin.

Only needs the metadata CSV (no HDF5 spectra). Takes seconds.
Run this once, then delete it — future runs of analyse_bins.py
will save these automatically.
"""

from pathlib import Path

import numpy as np
from analysis_funcs import build_bins_from_config

PLOTS_DIR = Path("./plots_analysis")
BEST_MODEL_METRIC = "std_z"

if __name__ == "__main__":
    print("Building bins from config (metadata only, no spectra)...")
    data, bins, _, _ = build_bins_from_config()
    data.close()

    plots_dir = PLOTS_DIR / BEST_MODEL_METRIC
    for i_bin, bin_data in enumerate(bins):
        bin_dir = plots_dir / f"bin_{i_bin:02d}"
        if not bin_dir.exists():
            print(f"  Bin {i_bin:2d}: no plot directory, skipping")
            continue
        out_path = bin_dir / "all_source_ids.npy"
        np.save(out_path, bin_data.ids)
        print(f"  Bin {i_bin:2d}: saved {len(bin_data.ids)} source IDs")

    print("Done.")
