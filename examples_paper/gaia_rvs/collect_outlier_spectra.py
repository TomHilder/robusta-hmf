"""
Copy all outlier spectra plots from a metric's results into a single folder.

Usage:
    python collect_outlier_spectra.py [metric]

    metric: std_z or mad_z (default: std_z)
"""

import shutil
import sys
from pathlib import Path

PLOTS_DIR = Path("./plots_analysis")
OUTPUT_DIR = Path("./outlier_spec")


def main():
    metric = sys.argv[1] if len(sys.argv) > 1 else "std_z"

    source_dir = PLOTS_DIR / metric
    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist")
        print(f"Available metrics: {[d.name for d in PLOTS_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)

    # Create output directory
    output_dir = OUTPUT_DIR / metric
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all outlier spectrum plots (in bin_XX subdirectories)
    # Pattern: bin_XX_KY_QZ.ZZ_idx_*_srcid_*_weight_*.pdf
    pattern = "bin_*_K*_Q*_idx_*.pdf"
    copied = 0

    for bin_dir in sorted(source_dir.glob("bin_*")):
        for spectrum_file in bin_dir.glob(pattern):
            dest = output_dir / spectrum_file.name
            shutil.copy2(spectrum_file, dest)
            copied += 1

    print(f"Copied {copied} outlier spectra from {metric} to {output_dir}")


if __name__ == "__main__":
    main()
