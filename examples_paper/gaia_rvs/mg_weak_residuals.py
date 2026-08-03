"""Search the outlier residuals for Mg-weak stars.

Same machinery as ``sprocess_line_residuals.py`` with the sign reversed: a
Mg-weak star has *less* absorption than the model, so the data sit above the
reconstruction and the sideband-minus-core index goes negative.

WHICH LINES ARE ACTUALLY AVAILABLE. Of the four Mg I positions asked about, one
is measurable. The RVS grid runs 846.4-869.6 nm after the edge clip, and the
requested wavelengths are NIST air values, which the line lists here are not
(``convert_air_to_vacuum.py`` converted them):

    given (air)   vacuum    in the grid?
    847.3         847.533   yes
    871.0         871.239   no, past the red end
    871.2         871.439   no
    871.7         871.939   no
    873.6         873.840   no

So the Mg I triplet and the stronger 873.6 nm line are outside the RVS window
entirely and nothing here can say anything about them.

AND THE ONE THAT IS IN RANGE IS NOT USABLE EITHER. This is the result. The
air/vacuum choice matters -- 0.233 nm is six times the core window -- but no
choice rescues it, because none of the candidate positions is an absorption
line in these spectra. ``locate_feature`` asks the data instead of the list,
and in the cool-star median spectrum:

    847.300 (given, read as vacuum)   on a flank; nearest minimum 847.410
    847.533 (8473.0 air -> vacuum)    on a local MAXIMUM; nearest min 847.450
    847.603 (8473.7 air -> vacuum)    on a local MAXIMUM; nearest min 847.450
    848.416 (the line list's "Mg I")  on a local MAXIMUM; depth ~0

The only real feature nearby is a broad trough at 847.450 that is absent in
warm stars (BP-RP < 1.5 median flux stays above 0.98 across the whole region,
where the Paschen 16 wing is what little structure there is). It sits in the
middle of the 846.7-848.7 nm CN band, is far too broad for an unresolved atomic
line at R ~ 11500, and is 0.08-0.11 nm from every candidate Mg position. It is
not the Mg I line and this module does not claim it is.

WHAT THAT MEANS. A Mg-weak search cannot be done with the given wavelengths on
RVS data: three are outside the window, and the fourth has no measurable Mg I
line. The index below is still computed and the figures still drawn, but they
measure *the 847.5 nm region*, which is why nothing here is named for Mg. Any
non-detection is a statement about the diagnostic, not about the stars.

For the record, since they were computed before the identification failed: the
region index reaches z < -3 in 0 of 1543 outliers where chance alone predicts
~12, and 81 stars reach z < -2 against ~46 expected. The region is uncorrelated
with CN strength measured in the other CN bands (r = -0.04 for cool stars,
against -0.21 for the Ce II index), so the excess at -2 is not CN either. The
Fe I lines are measured alongside because "Mg-weak" means low Mg/Fe rather than
low metallicity, and a metal-poor star is weak in everything.

OUTPUTS (in ./plots_<tag>_final/mg_weak by default)
    mg_wavelength_audit.csv    -- the table above: in range, depth, and what the
                                  position sits on. Read this first
    mg_line_identification.pdf -- the 847 nm region, warm and cool, with every
                                  candidate marked. The figure this turns on
    mg_indices.csv             -- a row per star: region index, z, Fe reference
    mg_stacks.pdf              -- median residual of the extremes vs the rest
    mg_summary.pdf             -- index distribution against the control
    spectra/                   -- per-star figures, named for the wavelength

USAGE
    uv run python mg_weak_residuals.py
    uv run python mg_weak_residuals.py --no-figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from convert_air_to_vacuum import air_to_vacuum_nm
from plot_final_full_rvs import check_text_rendering
from plot_final_outlier_spectra import DEFAULT_WEIGHTS, load_outlier_inputs
from plot_sprocess_spectra import plot_star
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

# The Mg I positions asked about, as given (air, nm).
MG_AIR = {
    "Mg I 8473 (weak)": 847.3,
    "Mg I 8710 (triplet)": 871.0,
    "Mg I 8712 (triplet)": 871.2,
    "Mg I 8717 (triplet)": 871.7,
    "Mg I 8736 (stronger)": 873.6,
}

# The one that lands in the grid, in vacuum. Everything below measures this.
MG_LINE = 847.533

# The line list's own "Mg I", kept only so the audit can report that it is empty.
REPO_MG = 848.416

# Sensitivity is colour-dependent: the line is absent in warm stars, so a
# candidate list only means anything redward of this.
COOL_BP_RP = 3.0

# Figures drawn per subfolder.
N_FIGURES = 30

# ============================================================================ #


def load_cached(out_dir, weights, state, threshold, sample):
    """Y, ivar, reconstruction and metadata for the outliers, cached on disk.

    The cache lives beside the s-process outputs because it is the same 1543
    spectra; building it costs a pass over the HDF5, and every analysis after
    the first should not pay that again.
    """
    cache = out_dir.parent / "sprocess" / "outlier_full_cache.npz"
    if cache.exists():
        c = np.load(cache)
        meta = pd.DataFrame(
            {k: c[k] for k in ("source_id", "score", "ra", "dec", "bp_rp", "abs_mag_G")}
        )
        print(f"Reusing {cache}")
        return (c["lambda_grid"], c["Y"].astype(np.float64), c["ivar"].astype(np.float64),
                c["recon"].astype(np.float64), meta)

    λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
        weights, state, threshold, None, sample
    )
    meta_df = pd.DataFrame(meta)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, lambda_grid=λ_grid, Y=Y.astype(np.float32), ivar=W.astype(np.float32),
        recon=recon.astype(np.float32), robust=robust.astype(np.float32),
        **{k: meta_df[k].to_numpy() for k in meta_df.columns},
    )
    print(f"Wrote {cache}")
    return λ_grid, Y, W, recon, meta_df


def measured_depth(λ_grid, Y, ivar, λ0, sel):
    """Depth of the line in the *data*: sideband continuum minus core.

    Positive means there is absorption there. This is what decides whether a
    wavelength is a line at all, independently of any model or residual.
    """
    d = np.abs(λ_grid - λ0)
    core, side = d <= HALF_CORE, (d >= SIDE_INNER) & (d <= SIDE_OUTER)
    f = np.where(ivar > 0, Y, np.nan)[sel]
    med = np.nanmedian(f, axis=0)
    return float(np.nanmean(med[side]) - np.nanmean(med[core]))


def locate_feature(λ_grid, Y, ivar, λ0, sel, search=0.12):
    """Where the nearest absorption minimum actually is, and how deep.

    A wavelength taken from a line list is a prediction; this asks the data. It
    returns the position of the deepest local minimum within *search* nm of
    *lambda0*, its offset, and whether *lambda0* itself sits on a minimum, a
    flank, or a local maximum. A candidate position that lands on a maximum is
    not a line, whatever the list says.
    """
    f = np.nanmedian(np.where(ivar > 0, Y, np.nan)[sel], axis=0)
    m = np.abs(λ_grid - λ0) <= search
    λw, fw = λ_grid[m], f[m]
    if len(fw) < 3:
        return np.nan, np.nan, "off grid"
    i = int(np.nanargmin(fw))
    # Local curvature at the requested position: negative second difference
    # means a peak, positive means a trough.
    j = int(np.argmin(np.abs(λ_grid - λ0)))
    curv = f[j - 1] - 2 * f[j] + f[j + 1]
    if abs(λw[i] - λ0) <= 0.015:
        where = "on a minimum"
    elif curv < 0:
        where = "on a local MAXIMUM"
    else:
        where = "on a flank"
    return float(λw[i]), float(λw[i] - λ0), where


def audit_wavelengths(λ_grid, Y, ivar, bp_rp, out):
    """Which requested wavelengths are usable, and is there a line there.

    Prints and returns the table; this is the first thing to read, because four
    of the five rows are "not in the RVS window" and no amount of analysis
    changes that.
    """
    warm, cool = bp_rp < 1.5, bp_rp > COOL_BP_RP
    rows = []
    for name, air in list(MG_AIR.items()) + [("Mg I 848.416 (line list)", None)]:
        vac = REPO_MG if air is None else air_to_vacuum_nm(air)
        in_grid = bool(λ_grid[0] + SIDE_OUTER <= vac <= λ_grid[-1] - SIDE_OUTER)
        row = {
            "line": name,
            "given_air_nm": air,
            "vacuum_nm": round(vac, 3),
            "in_grid": in_grid,
            "depth_warm": measured_depth(λ_grid, Y, ivar, vac, warm) if in_grid else np.nan,
            "depth_cool": measured_depth(λ_grid, Y, ivar, vac, cool) if in_grid else np.nan,
        }
        if in_grid:
            λmin, off, where = locate_feature(λ_grid, Y, ivar, vac, cool)
            row["nearest_min_cool_nm"], row["offset_nm"], row["sits"] = λmin, off, where
        rows.append(row)
    # The as-given value too, to record that it is not where the line is.
    for name, λ0 in (("Mg I 8473 taken as vacuum (wrong)", 847.300),
                     ("Mg I 8473.7 air -> vacuum", 847.603)):
        λmin, off, where = locate_feature(λ_grid, Y, ivar, λ0, cool)
        rows.append({
            "line": name, "given_air_nm": None, "vacuum_nm": λ0, "in_grid": True,
            "depth_warm": measured_depth(λ_grid, Y, ivar, λ0, warm),
            "depth_cool": measured_depth(λ_grid, Y, ivar, λ0, cool),
            "nearest_min_cool_nm": λmin, "offset_nm": off, "sits": where,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\nWavelength audit. 'sits' is where the requested position falls in the "
              "cool-star\nmedian spectrum: a position on a local maximum is not a line, "
              "whatever the list says.")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Wrote {out}")
    return df


def plot_line_identification(λ_grid, Y, ivar, bp_rp, out, label):
    """The 847 nm region in warm and cool stars, with every candidate marked.

    This is the figure the whole analysis turns on. If a Mg I line were usable,
    one of the dashed markers would sit in an absorption trough that deepens
    from warm to cool in the way an atomic line does. None does.
    """
    warm, cool = bp_rp < 1.5, bp_rp > COOL_BP_RP
    f = np.where(ivar > 0, Y, np.nan)
    med_cool = np.nanmedian(f[cool], axis=0)
    med_warm = np.nanmedian(f[warm], axis=0)
    m = (λ_grid > 847.05) & (λ_grid < 848.6)

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=140)
    ax.plot(λ_grid[m], med_cool[m], c="tab:red", lw=1.6,
            label=f"cool median, BP-RP > {COOL_BP_RP} (n={int(cool.sum())})")
    ax.plot(λ_grid[m], med_warm[m], c="tab:blue", lw=1.6,
            label=f"warm median, BP-RP < 1.5 (n={int(warm.sum())})")
    for λ0, name, c in (
        (847.300, "8473 as vacuum", "grey"),
        (847.533, "8473 air -> vac", "k"),
        (847.603, "8473.7 air -> vac", "tab:green"),
        (848.416, "line-list Mg I", "tab:purple"),
    ):
        if λ_grid[m][0] <= λ0 <= λ_grid[m][-1]:
            ax.axvline(λ0, c=c, lw=1.2, ls="--", alpha=0.9)
            # In axes coordinates, so the labels sit inside the frame rather
            # than climbing into the title.
            ax.annotate(name, (λ0, 0.985), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=8, ha="right", va="top", color=c)
    ax.axvspan(846.7, 848.7, color="tab:olive", alpha=0.08, lw=0)
    ax.annotate("CN band 846.7-848.7", (0.015, 0.03), xycoords="axes fraction",
                fontsize=8.5, color="olive")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Median normalised flux")
    ax.set_xlim(λ_grid[m][0], λ_grid[m][-1])
    # Headroom so the rotated labels clear the traces.
    lo, hi = np.nanmin(med_cool[m]), max(np.nanmax(med_warm[m]), np.nanmax(med_cool[m]))
    ax.set_ylim(lo - 0.02, hi + 0.06)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"{label}: is there a usable Mg I line at 847 nm?")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def measure(λ_grid, residual, every, cn_edges):
    """Mg and Fe indices per star, each with a control-calibrated z."""
    uni = np.ones_like(residual)
    λ_ctrl = control_wavelengths(λ_grid, every, cn_edges)
    ctrl = np.column_stack([line_index(λ_grid, residual, uni, λ0)[0] for λ0 in λ_ctrl])
    cmed, cstd = np.median(ctrl, axis=1), ctrl.std(axis=1)

    def z_of(λ0):
        i = line_index(λ_grid, residual, uni, λ0)[0]
        return i, (i - cmed) / cstd

    mg_i, mg_z = z_of(MG_LINE)
    fe_lines = pd.read_csv("rvs_gspspec_lines.csv")
    fe_lines = fe_lines[fe_lines["species"] == "Fe I"]["lambda_vac_nm"].to_numpy()
    fe_z = np.mean([z_of(λ0)[1] for λ0 in fe_lines], axis=0)
    return mg_i, mg_z, fe_z, ctrl, cmed, cstd


def cn_check(λ_grid, residual, mg_z, cool):
    """Is the Mg index really CN? Correlate it with the other CN bands.

    The band containing the Mg line is excluded, so this is CN strength measured
    somewhere the Mg line cannot contribute.
    """
    bands = [(859.2, 862.2), (865.2, 868.2), (850.2, 852.2)]
    cn = np.mean(
        [residual[:, (λ_grid >= a) & (λ_grid <= b)].mean(axis=1) for a, b in bands], axis=0
    )
    r_all = float(np.corrcoef(mg_z, cn)[0, 1])
    r_cool = float(np.corrcoef(mg_z[cool], cn[cool])[0, 1])
    print(f"\nCN cross-check: r(Mg index, CN-band residual) = {r_all:+.3f} all, "
          f"{r_cool:+.3f} cool only")
    return cn, r_all, r_cool


def plot_stacks(λ_grid, residual, weak, rest, out, label):
    """Median residual of the most Mg-weak stars against the rest.

    If there were a population of Mg-weak stars, the weak stack would sit above
    zero at the line and the comparison stack would not.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=140)
    for sel, colour, name in (
        (weak, "tab:blue", f"most Mg-weak (n={int(weak.sum())})"),
        (rest, "grey", f"other cool outliers (n={int(rest.sum())})"),
    ):
        med = np.median(residual[sel], axis=0)
        axes[0].plot(λ_grid, med, color=colour, lw=0.9, label=name)
        m = np.abs(λ_grid - MG_LINE) <= 0.35
        axes[1].plot(λ_grid[m] - MG_LINE, med[m], color=colour, lw=1.6, label=name)
    for ax in axes:
        ax.axhline(0, c="k", lw=0.6, alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_ylabel("Median residual")
    axes[0].set_xlim(λ_grid[0], λ_grid[-1])
    axes[0].set_xlabel("Wavelength [nm]")
    axes[0].axvline(MG_LINE, c="tab:red", lw=0.8, ls="--")
    axes[0].set_title("Full window")
    axes[1].axvspan(-HALF_CORE, HALF_CORE, color="tab:red", alpha=0.14, lw=0)
    axes[1].set_xlabel(rf"$\lambda - {MG_LINE}$ [nm]")
    axes[1].set_title(f"Mg I {MG_LINE} nm (positive = less absorption than the model)")
    fig.suptitle(f"{label}: Mg I stacked residuals", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_summary(mg_z, ctrl, cmed, cstd, out, label):
    """The Mg z distribution over the off-line control, both tails visible."""
    z_ctrl = ((ctrl - cmed[:, None]) / cstd[:, None]).ravel()
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    bins = np.linspace(-6, 6, 100)
    ax.hist(z_ctrl, bins=bins, density=True, color="lightgrey", label="off-line control")
    ax.hist(mg_z, bins=bins, density=True, histtype="step", lw=1.7, color="tab:blue",
            label=f"Mg I {MG_LINE}")
    for s in (-3, 3):
        ax.axvline(s, c="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("z (negative = Mg weaker than the model)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title(f"{label}: Mg I index against the control")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    p.add_argument("--sample", default="all_filtered", choices=("ms", "all", "all_filtered"))
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-figures", type=int, default=N_FIGURES)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-latex", action="store_true")
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False

    d = np.load(args.weights)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    label = "Full Main Sequence" if tag == "full_ms" else "Full RVS Sample"
    state_file = args.state or (
        args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/mg_weak")
    out_dir.mkdir(parents=True, exist_ok=True)

    check_text_rendering()

    λ_grid, Y, ivar, recon, meta_df = load_cached(
        out_dir, args.weights, state_file, threshold, args.sample
    )
    residual = Y - recon
    bp_rp = meta_df["bp_rp"].to_numpy()
    cool = bp_rp > COOL_BP_RP

    audit_wavelengths(λ_grid, Y, ivar, bp_rp, out_dir / "mg_wavelength_audit.csv")
    plot_line_identification(
        λ_grid, Y, ivar, bp_rp, out_dir / "mg_line_identification.pdf", label
    )

    lines, every, cn_edges = load_lines(λ_grid)
    mg_i, mg_z, fe_z, ctrl, cmed, cstd = measure(λ_grid, residual, every, cn_edges)
    cn, _, _ = cn_check(λ_grid, residual, mg_z, cool)

    meta_df["mg_index"], meta_df["mg_z"], meta_df["fe_z"] = mg_i, mg_z, fe_z
    meta_df["mgfe"] = mg_z - fe_z
    meta_df["control_scatter"] = cstd
    meta_df["cn_residual"] = cn
    pop = load_crossmatch(Path(__file__).parent)
    meta_df["population"] = meta_df["source_id"].astype(int).map(pop).fillna("")
    meta_df.to_csv(out_dir / "mg_indices.csv", index=False)
    print(f"Wrote {out_dir / 'mg_indices.csv'}")

    depth = measured_depth(λ_grid, Y, ivar, MG_LINE, cool)
    need = 3 * np.median(cstd[cool])
    print(f"\nSensitivity: line depth {depth:.4f} in cool stars, median control scatter "
          f"{np.median(cstd[cool]):.4f}")
    print(f"  a 3-sigma change is {need:.4f}, i.e. {100 * need / depth:.0f}% of the line "
          f"(~{np.log10(1 + need / depth):.2f} dex); warm stars have no line and no sensitivity")

    chance_lo = float(np.mean(ctrl < (cmed[:, None] - 3 * cstd[:, None])))
    print("\n847.5 nm region counts (negative z = less absorption than the model):")
    print(f"  z < -3: {int((mg_z < -3).sum())} of {len(mg_z)} "
          f"(chance would give ~{chance_lo * len(mg_z):.0f})")
    print(f"  z < -2: {int((mg_z < -2).sum())} "
          f"(chance ~{float(np.mean(ctrl < (cmed[:, None] - 2 * cstd[:, None]))) * len(mg_z):.0f})")
    print(f"  cool stars only, z < -3: {int((mg_z[cool] < -3).sum())} of {int(cool.sum())}")
    print(f"  most extreme z: {mg_z.min():+.2f}")
    print(f"  mgfe proxy < -3 (cool): {int((meta_df['mgfe'].to_numpy()[cool] < -3).sum())}")

    n = min(args.n_figures, int(cool.sum()))
    weak_rank = np.argsort(np.where(cool, mg_z, np.inf))[:n]
    weak = np.zeros(len(mg_z), bool)
    weak[weak_rank] = True
    plot_stacks(λ_grid, residual, weak, cool & ~weak, out_dir / "mg_stacks.pdf", label)
    plot_summary(mg_z, ctrl, cmed, cstd, out_dir / "mg_summary.pdf", label)

    if args.no_figures:
        print(f"\nDone (no per-star figures). Output in {out_dir}")
        return

    # Named for the wavelength, not for Mg: the audit shows the index is not
    # attributable to a Mg I line, so calling these folders "Mg-weak" would put
    # a claim in the filename that the data does not support.
    groups = {
        "region8475_least_absorption_cool": meta_df.index.isin(weak_rank),
        "region8475_most_absorption_cool": meta_df.index.isin(
            np.argsort(np.where(cool, -mg_z, np.inf))[:n]
        ),
    }
    spec_dir = out_dir / "spectra"
    rows = []
    for name, sel in groups.items():
        # Ranked most Mg-weak first, except the contrast folder.
        asc = "least" in name
        key = "mg_z"
        sub = meta_df[sel].sort_values(key, ascending=asc)
        folder = spec_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {len(sub)} stars -> {folder}", flush=True)
        for rank, (i, row) in enumerate(sub.iterrows(), start=1):
            fn = folder / f"rank_{rank:03d}_z{row['mg_z']:+.2f}_srcid_{int(row['source_id'])}.png"
            plot_star(
                λ_grid, Y[i], ivar[i], recon[i], residual[i], row, lines,
                float(row["control_scatter"]), fn,
                line=MG_LINE, line_label="847.5 nm region", index_col="mg_index", z_col="mg_z",
            )
            rows.append({"group": name, "rank": rank, "file": fn.name, **row.to_dict()})
        print(f"  {len(sub)} figures written", flush=True)

    pd.DataFrame(rows).to_csv(spec_dir / "spectra_index.csv", index=False)
    print(f"\nWrote {spec_dir / 'spectra_index.csv'}")
    print(f"Done. {len(rows)} figures in {spec_dir}")


if __name__ == "__main__":
    main()
