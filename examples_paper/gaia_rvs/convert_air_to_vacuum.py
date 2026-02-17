"""
One-off script to convert line list CSV wavelengths from air to vacuum.

The CSV files (rvs_strong_features.csv, rvs_gspspec_lines.csv, rvs_cn_band_regions.csv,
rvs_dib_features.csv) contain air wavelengths mislabelled as 'lambda_vac_nm'. This was
confirmed by checking 10 lines across Ca II, N I, Fe I, and Si I against NIST air values.

This script applies the IAU standard Edlen (1966) / Morton (2000) air-to-vacuum conversion,
then overwrites the CSV files with the corrected vacuum wavelengths.

Run once, then delete this script.
"""

import csv
from pathlib import Path


def air_to_vacuum_nm(lambda_air_nm):
    """Convert air wavelength (nm) to vacuum wavelength (nm).

    Uses the IAU standard formula (Edlen 1966 / Morton 2000):
        (n - 1) * 1e8 = 8342.54 + 2406147 / (130 - sigma^2) + 15998 / (38.9 - sigma^2)
    where sigma = 1 / lambda_vac in micrometers.

    Iterative solution since formula is in terms of lambda_vac.
    """
    lv = lambda_air_nm
    for _ in range(10):
        sigma = 1.0 / (lv / 1000.0)  # nm -> um -> um^-1
        sigma2 = sigma * sigma
        n = 1.0 + (8342.54 + 2406147.0 / (130.0 - sigma2) + 15998.0 / (38.9 - sigma2)) * 1e-8
        lv = lambda_air_nm * n
    return lv


def convert_csv(filepath, wavelength_columns):
    """Read a CSV, convert specified wavelength columns from air to vacuum, overwrite."""
    filepath = Path(filepath)
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            for col in wavelength_columns:
                air_val = float(row[col])
                vac_val = air_to_vacuum_nm(air_val)
                # Round to same precision as original
                # Determine decimal places from original string
                original_str = row[col]
                if "." in original_str:
                    n_decimals = len(original_str.split(".")[1])
                else:
                    n_decimals = 0
                row[col] = f"{vac_val:.{n_decimals}f}"
            rows.append(row)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    base_dir = Path(__file__).parent

    # Show a calibration check first
    print("Calibration check (Ca II triplet):")
    for name, air_nm in [("CaT 8498", 849.802), ("CaT 8542", 854.209), ("CaT 8662", 866.214)]:
        vac_nm = air_to_vacuum_nm(air_nm)
        shift_pm = (vac_nm - air_nm) * 1000
        print(f"  {name}: {air_nm:.3f} nm (air) -> {vac_nm:.3f} nm (vac), shift = {shift_pm:.1f} pm")
    print()

    # Convert each CSV
    files_and_columns = {
        "rvs_strong_features.csv": ["lambda_vac_nm"],
        "rvs_gspspec_lines.csv": ["lambda_vac_nm"],
        "rvs_dib_features.csv": ["lambda_vac_nm"],
        "rvs_cn_band_regions.csv": ["lambda_vac_nm_start", "lambda_vac_nm_end"],
    }

    for filename, columns in files_and_columns.items():
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"WARNING: {filename} not found, skipping")
            continue
        n_rows = convert_csv(filepath, columns)
        print(f"Converted {filename}: {n_rows} rows, columns {columns}")

    # Also convert rvs_all_lines_for_plotting.csv if it exists
    all_lines_path = base_dir / "rvs_all_lines_for_plotting.csv"
    if all_lines_path.exists():
        n_rows = convert_csv(all_lines_path, ["lambda_vac_nm"])
        print(f"Converted rvs_all_lines_for_plotting.csv: {n_rows} rows")

    print("\nDone. All wavelengths converted from air to vacuum.")
    print("The 'lambda_vac_nm' column headers are now correct.")
