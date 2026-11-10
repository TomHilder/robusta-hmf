"""One figure per star for the s-process search, sorted into subfolders.

``sprocess_line_residuals.py`` and ``sprocess_stack.py`` reduce the search to a
number per star: the Ce II 853.276 nm index, in units of that star's own
off-line control scatter. This module draws the spectra behind those numbers, so
a candidate can be looked at rather than trusted.

SUBFOLDERS, under the output directory:

    new_candidates/  uncatalogued stars with Ce II z > 3 -- the actual result
    known_S/         S stars in the outlier set, catalogued as such: the
                     positive control, drawn whatever they scored, so the
                     misses are as visible as the hits
    known_Ba/        barium stars in the outlier set
    known_C/         carbon stars in the outlier set

Filenames lead with the rank inside the folder and then the z, so an
alphabetical listing is ranked best-first:

    rank_001_z5.97_srcid_5053856547979769728.png

WHAT IS PLOTTED, four panels:

    1. the full window: observed flux over the rank-K reconstruction, with the
       Ce II, Nd II and Zr I lines marked and Ce II 853.276 picked out;
    2. the residual over the +/- 1 sigma band from the catalogue flux errors,
       with the star's own control band (+/- 1 control sigma) drawn flat -- a
       feature that leaves that band is one the rest of the spectrum does not
       explain;
    3. a zoom on Ce II 853.276: flux and model, where a real detection is the
       model sitting visibly above the data over the core;
    4. the same zoom on the residual, with the core and sideband windows the
       index is built from shaded, so the number can be read off the picture.

The subtitle carries the identifiers and the numbers: source id, outlier score,
Ce II index and z, BP - RP, absolute G magnitude, and the catalogue label.

USAGE
    uv run python plot_sprocess_spectra.py
    uv run python plot_sprocess_spectra.py --z-min 4 --limit 20

Reads the spectra, so it runs where the fit ran. Note the sample: the saved
full-RVS fit is 993,910 rows, so this uses ``--sample all_filtered`` by default
(see ``train_full_ms.build_sample``); switch to ``all`` once the fit is redone.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_final_full_rvs import check_text_rendering
from plot_final_outlier_spectra import DEFAULT_WEIGHTS, load_outlier_inputs
from sprocess_line_residuals import (
    HALF_CORE,
    SIDE_INNER,
    SIDE_OUTER,
    control_wavelengths,
    line_index,
    load_crossmatch,
    load_lines,
)

try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# The one line that survives the control test; see sprocess_line_residuals.
CE_LINE = 853.276

# Candidate cut, in units of the star's own off-line control scatter.
Z_MIN = 3.0

# Half-width of the zoom panels, in nm.
ZOOM_HALF = 0.35

DPI = 130
FIGSIZE = (14, 10)

SPECIES_COLOUR = {"Ce II": "tab:purple", "Nd II": "tab:green", "Zr I": "tab:orange"}

# ============================================================================ #


def compute_z(λ_grid, residual, every, cn_edges):
    """Ce II index per star, and its significance against that star's controls.

    Uniform pixel weights, matching ``sprocess_stack``: the index is a plain
    window mean and the significance comes from the control scatter, so the
    catalogue errors never enter and a star with optimistic errors cannot buy
    itself a detection.
    """
    ivar = np.ones_like(residual)
    λ_ctrl = control_wavelengths(λ_grid, every, cn_edges)
    ctrl = np.column_stack([line_index(λ_grid, residual, ivar, λ0)[0] for λ0 in λ_ctrl])
    ce = line_index(λ_grid, residual, ivar, CE_LINE)[0]
    scatter = ctrl.std(axis=1)
    return ce, (ce - np.median(ctrl, axis=1)) / scatter, scatter


def _mark_lines(ax, lines, focus=CE_LINE, alpha=0.16):
    """Shade every measured line, with the one being measured picked out."""
    for _, r in lines.iterrows():
        λ0, sp = float(r["lambda_vac_nm"]), r["species"]
        is_focus = abs(λ0 - focus) < 1e-6
        ax.axvspan(
            λ0 - HALF_CORE, λ0 + HALF_CORE,
            color=SPECIES_COLOUR.get(sp, "grey"),
            alpha=0.32 if is_focus else alpha, lw=0, zorder=0,
        )


def plot_star(λ_grid, Y, ivar, recon, residual, row, lines, scatter, out,
              line=CE_LINE, line_label="Ce II", index_col="ce853_index", z_col="ce853_z"):
    """The four-panel figure for one star.

    *line* is the wavelength the zoom panels centre on and the one the index in
    the subtitle refers to, so the same figure serves any line the index
    machinery can measure, not only Ce II.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=FIGSIZE, dpi=DPI,
        gridspec_kw={"width_ratios": [2.2, 1.0], "hspace": 0.28, "wspace": 0.24},
    )
    good = ivar > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(good, 1.0 / np.sqrt(np.where(good, ivar, 1.0)), np.nan)
    flux = np.where(good, Y, np.nan)

    # -- full window, flux and model
    ax = axes[0, 0]
    _mark_lines(ax, lines, focus=line)
    ax.plot(λ_grid, flux, c="k", lw=0.8, zorder=3, label="Data")
    ax.plot(λ_grid, recon, c="tab:red", lw=0.9, ls=(0, (5, 1)), zorder=4, label="Model")
    ax.set_ylabel("Flux")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim(λ_grid[0], λ_grid[-1])

    # -- full window, residual
    ax = axes[1, 0]
    _mark_lines(ax, lines, focus=line)
    ax.fill_between(λ_grid, -sigma, sigma, color="tab:blue", alpha=0.2, lw=0,
                    label=r"$\pm 1\sigma$ (catalogue)")
    ax.axhspan(-scatter, scatter, color="grey", alpha=0.22, lw=0,
               label=r"$\pm 1\sigma$ (control)")
    ax.plot(λ_grid, np.where(good, residual, np.nan), c="k", lw=0.8, zorder=3)
    ax.axhline(0, c="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Residual")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim(λ_grid[0], λ_grid[-1])

    # -- zoom, flux and model
    m = np.abs(λ_grid - line) <= ZOOM_HALF
    ax = axes[0, 1]
    ax.axvspan(-HALF_CORE, HALF_CORE, color="tab:purple", alpha=0.22, lw=0)
    ax.plot(λ_grid[m] - line, flux[m], c="k", lw=1.3, zorder=3)
    ax.plot(λ_grid[m] - line, recon[m], c="tab:red", lw=1.3, ls=(0, (5, 1)), zorder=4)
    ax.set_title(f"{line_label} {line} nm", fontsize=10)
    ax.set_ylabel("Flux")
    ax.set_xlim(-ZOOM_HALF, ZOOM_HALF)

    # -- zoom, residual, with the windows the index is built from
    ax = axes[1, 1]
    ax.axvspan(-HALF_CORE, HALF_CORE, color="tab:purple", alpha=0.22, lw=0, label="core")
    for s in (-1, 1):
        ax.axvspan(s * SIDE_INNER, s * min(SIDE_OUTER, ZOOM_HALF), color="tab:olive",
                   alpha=0.16, lw=0, label="sideband" if s == 1 else None)
    ax.fill_between(λ_grid[m] - line, -sigma[m], sigma[m], color="tab:blue", alpha=0.2, lw=0)
    ax.axhspan(-scatter, scatter, color="grey", alpha=0.22, lw=0)
    ax.plot(λ_grid[m] - line, np.where(good, residual, np.nan)[m], c="k", lw=1.3, zorder=3)
    ax.axhline(0, c="k", lw=0.6, alpha=0.5)
    ax.set_xlabel(rf"$\lambda - {line}$ [nm]")
    ax.set_ylabel("Residual")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_xlim(-ZOOM_HALF, ZOOM_HALF)

    pop = row["population"] or "uncatalogued"
    title = (
        f"Gaia DR3 {int(row['source_id'])}   [{pop}]\n"
        f"{line_label} index {row[index_col]:+.4f} ({row[z_col]:+.2f}$\\,\\sigma$ vs control)   "
        f"outlier score {row['score']:.3f}   "
        f"BP$-$RP {row['bp_rp']:.2f}   $M_G$ {row['abs_mag_G']:.2f}"
    )
    fig.suptitle(title, fontsize=12, y=0.97)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    p.add_argument(
        "--sample", default="all_filtered", choices=("ms", "all", "all_filtered"),
        help="default matches the saved 993,910-row fit",
    )
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--z-min", type=float, default=Z_MIN)
    p.add_argument("--limit", type=int, default=None, help="cap the figures per subfolder")
    p.add_argument("--no-latex", action="store_true")
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False
    if not args.weights.exists():
        raise SystemExit(f"{args.weights} does not exist -- run fit_final_full_rvs.py first.")

    d = np.load(args.weights)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    state_file = args.state or (
        args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    if not state_file.exists():
        raise SystemExit(f"{state_file} does not exist -- pass --state explicitly.")
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/sprocess/spectra")
    out_dir.mkdir(parents=True, exist_ok=True)

    check_text_rendering()

    λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
        args.weights, state_file, threshold, None, args.sample
    )
    residual = Y - recon
    meta_df = pd.DataFrame(meta)

    lines, every, cn_edges = load_lines(λ_grid)
    ce, z, scatter = compute_z(λ_grid, residual, every, cn_edges)
    meta_df["ce853_index"], meta_df["ce853_z"], meta_df["control_scatter"] = ce, z, scatter
    pop = load_crossmatch(Path(__file__).parent)
    meta_df["population"] = meta_df["source_id"].astype(int).map(pop).fillna("")

    p_str = meta_df["population"]
    groups = {
        "new_candidates": (p_str == "") & (meta_df["ce853_z"] > args.z_min),
        "known_S": p_str.str.contains("S", regex=False),
        "known_Ba": p_str.str.contains("Ba", regex=False),
        "known_C": p_str.str.fullmatch("C").fillna(False),
    }

    index_rows = []
    for name, sel in groups.items():
        sub = meta_df[sel.to_numpy()].sort_values("ce853_z", ascending=False)
        if args.limit:
            sub = sub.head(args.limit)
        folder = out_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {len(sub)} stars -> {folder}", flush=True)
        for rank, (i, row) in enumerate(sub.iterrows(), start=1):
            fn = folder / f"rank_{rank:03d}_z{row['ce853_z']:+.2f}_srcid_{int(row['source_id'])}.png"
            plot_star(
                λ_grid, Y[i], W[i], recon[i], residual[i], row, lines,
                float(row["control_scatter"]), fn,
            )
            index_rows.append({"group": name, "rank": rank, "file": fn.name, **row.to_dict()})
            if rank % 20 == 0:
                print(f"  {rank}/{len(sub)}", end="\r", flush=True)
        print(f"  {len(sub)} figures written" + " " * 20, flush=True)

    idx_csv = out_dir / "spectra_index.csv"
    pd.DataFrame(index_rows).to_csv(idx_csv, index=False)
    print(f"\nWrote {idx_csv}")
    print(f"Done. {len(index_rows)} figures in {out_dir}")


if __name__ == "__main__":
    main()
