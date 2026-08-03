"""Search the outlier residuals for stars enhanced in the s-process lines.

The RVS window carries four Ce II lines, three Nd II lines and four Zr I lines
(``rvs_gspspec_lines.csv``). A star with an s-process excess has more absorption
in those lines than a normal star of its colour and magnitude -- which is
exactly what the factorisation cannot reproduce, so it should be left in the
residual as extra *absorption* at those wavelengths and nowhere else.

MEASUREMENT, per line and per star. The residual is ``flux - reconstruction``,
so negative means the data sit below the model: absorption the model could not
make. For a line at lambda0 the index is

    index = mean(residual in the sidebands) - mean(residual in the core)

with the core ``|lambda - lambda0| <= HALF_CORE`` and the sidebands an annulus
around it. A *positive* index means extra absorption localised at the line. The
sideband term removes any broad residual trend under the line, so what is left
is a feature at the line's own width rather than a continuum error; without it,
every badly-fit star scores highly on every line at once.

Uncertainties come from the catalogue flux errors, propagated through the same
two means, so ``index / sigma`` is how surprising the feature is rather than how
big. Per-species indices are inverse-variance means over that species' lines.

WHY Zr I IS HERE. Ce II and Nd II are the question, Zr I is the check: the three
are made by the same process, so a genuine s-process star should show all three,
while a wavelength-calibration slip, a blend, or a template mismatch has no
reason to respect that. A candidate with Ce and Nd but no Zr is worth less than
one with all three, and the summary reports them separately for that reason.

BLENDS. This is a crowded region at R ~ 11500 (FWHM ~ 0.074 nm), and some of the
lines have a neighbour inside the sidebands: the ``nearest_other_nm`` column of
the per-line output gives the distance to the closest line of another species in
the full list, so a suspicious index can be checked against it. Nd II 855.933 is
the worst case, 0.11 nm from H I Pa13, and is flagged in the output.

CONTROL, AND WHY IT IS THE WHOLE ANALYSIS. The same statistic is measured at
wavelengths at least CONTROL_GAP nm from every catalogued line. Those indices
are what "no feature" looks like, and every significance here is quoted against
their scatter *in the same star*, never against the formal errors and never
against other stars.

That is not fussiness. The s-process stars in this sample are cool, molecule-
rich giants, and the model fits them badly across the entire window: their
median |residual| is about twice the field's. Compare them to other stars and
every line in the list comes out "enhanced", because the whole spectrum is
darker. Compare each line to random wavelengths *in the same spectrum* and that
pedestal cancels, which is the only version of the test that means anything.
``line_control_report`` prints exactly this, per line, and is the table to read.

WHAT SURVIVES. On the full-RVS outliers, one line does: Ce II 853.276, whose
S-star median index beats all 591 control wavelengths. Nd II 855.933 is
marginal (98th percentile) and sits 0.11 nm from H I Pa13. Every other Ce II,
Nd II and Zr I line in the window is consistent with the control. Zr I offers no
corroboration, though in S stars zirconium is largely locked up in ZrO, so weak
atomic Zr I is expected rather than contradictory.

OUTPUTS (in ./plots_<tag>_final/sprocess by default)
    sprocess_indices.csv       -- a row per star: per-species indices, their
                                  significances, the control scatter, rank
    sprocess_per_line.csv      -- a row per star per line, for the top N
    sprocess_candidates.pdf    -- residuals of the top candidates around each
                                  s-process line, with the sample median
    sprocess_summary.pdf       -- index distributions against the control

USAGE
    uv run python sprocess_line_residuals.py
    uv run python sprocess_line_residuals.py --top 40

    # from the residual cache sprocess_stack.py writes, which needs neither the
    # HDF5 nor the sample definition
    uv run python sprocess_line_residuals.py --from-cache

Needs the same inputs as plot_final_outlier_spectra.py, unless ``--from-cache``.

NOTE ON THE CACHE. As of ``a5d4aef`` the "all" sample is 999,645 spectra, while
``full_rvs_final_weights.npz`` and its converged state hold 993,910 -- the count
with the metadata filters still on. Until the final fit is redone, the row-order
assertion in ``load_outlier_inputs`` fires and the only way to run this is
``--from-cache``, off a cache written before that change. The cache carries no
inverse variances, so the formal per-line errors are unavailable there; that
costs nothing, because every significance quoted here is calibrated against the
control wavelengths rather than the formal errors.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from plot_final_full_rvs import check_text_rendering
from plot_final_outlier_spectra import DEFAULT_WEIGHTS, load_outlier_inputs

try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Half the core window, in nm. The RVS resolving power is ~11500, so a line is
# ~0.074 nm FWHM near 860 nm; half of that keeps the pixels that carry the depth
# and leaves out the wings, where the neighbours start.
HALF_CORE = 0.037

# The sideband annulus, in nm from the line centre: outside the wings, inside
# the distance over which a residual trend is straight.
SIDE_INNER = 0.090
SIDE_OUTER = 0.260

# Control wavelengths must sit at least this far from every catalogued line,
# CN band edge included.
CONTROL_GAP = 0.15

# Species measured. Ce II and Nd II are the question; Zr I is the corroboration.
SPECIES = ("Ce II", "Nd II", "Zr I")

LINELIST = Path(__file__).parent / "rvs_gspspec_lines.csv"
ALL_LINES = Path(__file__).parent / "rvs_all_lines_for_plotting.csv"
CN_BANDS = Path(__file__).parent / "rvs_cn_band_regions.csv"

# Known-population cross-matches, if the CSVs are sitting next to this file.
# Ba/S/C stars are the s-process populations; a candidate landing in one of them
# is the cheapest confirmation available.
CROSSMATCH = {
    "Ba": "ba_stars.csv",
    "S": "sstar.csv",
    "C": "cstar.csv",
    "CEMP": "fulbright.csv",
}

# ============================================================================ #


def load_lines(λ_grid=None):
    """The measured lines, and every catalogued wavelength for the blend check.

    When *lambda_grid* is given, lines whose core window falls off the end of
    the grid are dropped: the RVS grid stops at 869.6 nm, which puts Zr I 869.65
    outside it, and an index built from an empty window is NaN rather than a
    non-detection.
    """
    lines = pd.read_csv(LINELIST)
    lines = lines[lines["species"].isin(SPECIES)].reset_index(drop=True)
    if λ_grid is not None:
        λ0 = lines["lambda_vac_nm"].to_numpy()
        keep = (λ0 - HALF_CORE >= λ_grid[0]) & (λ0 + HALF_CORE <= λ_grid[-1])
        for _, r in lines[~keep].iterrows():
            print(f"  dropping {r['species']} {r['lambda_vac_nm']:.3f}: outside the grid")
        lines = lines[keep].reset_index(drop=True)

    every = pd.read_csv(ALL_LINES)[["lambda_vac_nm", "species"]]
    # CN is a band rather than a line, but for "is this window clean" purposes
    # the band edges are the honest thing to measure a distance to.
    cn = pd.read_csv(CN_BANDS)
    cn_edges = np.concatenate(
        [cn["lambda_vac_nm_start"].to_numpy(), cn["lambda_vac_nm_end"].to_numpy()]
    )
    return lines, every, cn_edges


def nearest_other(λ0, species, every):
    """Distance in nm to the closest catalogued line of a *different* species."""
    other = every[every["species"] != species]["lambda_vac_nm"].to_numpy()
    if not len(other):
        return np.inf
    return float(np.min(np.abs(other - λ0)))


def line_index(λ_grid, residual, ivar, λ0):
    """Sideband-minus-core mean residual at *lambda0*, and its uncertainty.

    Returns ``(index, sigma, n_core, n_side)``; positive index means extra
    absorption localised at the line. ``index`` is NaN when either window has no
    unmasked pixel, which happens at the ends of the grid and for spectra masked
    across the line.
    """
    d = np.abs(λ_grid - λ0)
    core = d <= HALF_CORE
    side = (d >= SIDE_INNER) & (d <= SIDE_OUTER)
    if core.sum() == 0 or side.sum() == 0:
        n = len(residual)
        return np.full(n, np.nan), np.full(n, np.nan), 0, 0

    # Masked pixels carry zero inverse variance and a residual of zero, so they
    # have to be dropped from the means rather than averaged in as agreement.
    def _mean(sel):
        good = ivar[:, sel] > 0
        n = good.sum(axis=1)
        r = np.where(good, residual[:, sel], 0.0).sum(axis=1)
        # Variance of a plain mean of n independent pixels.
        v = np.where(good, 1.0 / np.where(good, ivar[:, sel], 1.0), 0.0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n > 0, r / n, np.nan), np.where(n > 0, v / n**2, np.nan), n

    m_core, v_core, n_core = _mean(core)
    m_side, v_side, n_side = _mean(side)
    return m_side - m_core, np.sqrt(v_core + v_side), n_core, n_side


def control_wavelengths(λ_grid, every, cn_edges):
    """Grid wavelengths at least CONTROL_GAP from any catalogued feature."""
    catalogued = np.concatenate([every["lambda_vac_nm"].to_numpy(), cn_edges])
    d = np.abs(λ_grid[:, None] - catalogued[None, :]).min(axis=1)
    # Keep clear of the grid ends too, so every control has full sidebands.
    inner = (λ_grid > λ_grid[0] + SIDE_OUTER) & (λ_grid < λ_grid[-1] - SIDE_OUTER)
    return λ_grid[(d >= CONTROL_GAP) & inner]


def measure(λ_grid, residual, ivar, lines, every, cn_edges):
    """Per-line and per-species indices, plus the control statistics.

    Returns ``(per_line, species_tables, control)`` where *per_line* is a dict
    keyed by ``(species, lambda0)``, *species_tables* is a dict of species ->
    (index, sigma) inverse-variance combined over that species' lines, and
    *control* is (n_stars, n_control) of the same statistic off-line.
    """
    per_line = {}
    for _, row in lines.iterrows():
        λ0, sp = float(row["lambda_vac_nm"]), row["species"]
        idx, sig, n_core, n_side = line_index(λ_grid, residual, ivar, λ0)
        per_line[(sp, λ0)] = {
            "index": idx,
            "sigma": sig,
            "n_core": n_core,
            "n_side": n_side,
            "log_gf": float(row["log_gf"]),
            "nearest_other_nm": nearest_other(λ0, sp, every),
        }

    species_tables = {}
    for sp in SPECIES:
        cols = [v for (s, _), v in per_line.items() if s == sp]
        idx = np.column_stack([c["index"] for c in cols])
        sig = np.column_stack([c["sigma"] for c in cols])
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / sig**2
        w = np.where(np.isfinite(w) & np.isfinite(idx), w, 0.0)
        sw = w.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            combined = np.where(sw > 0, np.nansum(w * np.nan_to_num(idx), axis=1) / sw, np.nan)
            combined_sig = np.where(sw > 0, 1.0 / np.sqrt(sw), np.nan)
        species_tables[sp] = (combined, combined_sig)

    λ_ctrl = control_wavelengths(λ_grid, every, cn_edges)
    print(f"  {len(λ_ctrl)} control wavelengths at least {CONTROL_GAP} nm from any line")
    control = np.column_stack(
        [line_index(λ_grid, residual, ivar, λ0)[0] for λ0 in λ_ctrl]
    )
    return per_line, species_tables, control, λ_ctrl


def line_control_report(per_line, control, groups):
    """Per line: is it enhanced relative to random wavelengths in the same stars?

    For each group of stars (typically the known s-process ones and a colour-
    matched field), the group's median index at the line is placed in the
    distribution of that same group's median indices at the control
    wavelengths. A line that is genuinely there sits at the top of that
    distribution; a line that only looks enhanced because the whole spectrum is
    poorly fit sits in the middle of it, because the controls are inflated by
    exactly the same amount.

    Returns the table as a DataFrame and prints it.
    """
    rows = []
    for (sp, λ0), v in sorted(per_line.items(), key=lambda kv: kv[0][1]):
        row = {"species": sp, "lambda_vac_nm": λ0, "nearest_other_nm": v["nearest_other_nm"]}
        for name, sel in groups.items():
            med = np.nanmedian(v["index"][sel])
            ctrl_med = np.nanmedian(control[sel], axis=0)
            row[f"{name}_index"] = med
            row[f"{name}_pctile"] = 100.0 * np.mean(ctrl_med < med)
        rows.append(row)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 25):
        print("\nPer-line control test (percentile of the group median among control "
              "wavelengths;\n>95 means the line stands out from random wavelengths in the "
              "same stars):")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return df


def load_crossmatch(base):
    """source_id -> population label, from whichever CSVs are present."""
    out = {}
    for label, name in CROSSMATCH.items():
        f = base / name
        if not f.exists():
            continue
        df = pl.read_csv(f)
        if "id" not in df.columns:
            continue
        ids = (
            df["id"]
            .str.replace("Gaia DR3 ", "", literal=True)
            .cast(pl.Int64, strict=False)
            .drop_nulls()
            .to_list()
        )
        for i in ids:
            out.setdefault(i, []).append(label)
        print(f"  {len(ids)} {label} stars from {name}")
    return {k: "+".join(v) for k, v in out.items()}


def plot_candidates(λ_grid, residual, lines, order, meta_df, out, label, n_show=8):
    """The top candidates' residuals, one column per s-process line.

    Each panel is a narrow window around one line, on a shared vertical scale,
    with the sample's median residual behind in grey. A genuine detection is a
    dip at the centre of most panels in a row; a blend or a calibration slip is
    a dip that wanders off centre, and a badly-fit star is a row that is
    everywhere dark.
    """
    picks = order[:n_show]
    λ0s = lines["lambda_vac_nm"].to_numpy()
    sps = lines["species"].to_numpy()
    med_all = np.median(residual, axis=0)
    half = SIDE_OUTER

    fig, axes = plt.subplots(
        len(picks), len(λ0s),
        figsize=(1.55 * len(λ0s) + 2.0, 1.35 * len(picks) + 1.2),
        dpi=140, sharey=True, squeeze=False,
    )
    span = np.percentile(np.abs(residual[picks]), 99)
    for r, idx in enumerate(picks):
        for c, (λ0, sp) in enumerate(zip(λ0s, sps)):
            ax = axes[r, c]
            m = np.abs(λ_grid - λ0) <= half
            ax.plot(λ_grid[m] - λ0, med_all[m], c="lightgrey", lw=1.0, zorder=1)
            ax.plot(λ_grid[m] - λ0, residual[idx][m], c="k", lw=1.0, zorder=3)
            ax.axhline(0, c="k", lw=0.5, alpha=0.4)
            ax.axvspan(-HALF_CORE, HALF_CORE, color="tab:red", alpha=0.15, lw=0)
            ax.set_xlim(-half, half)
            ax.set_xticks([])
            if r == 0:
                ax.set_title(f"{sp}\n{λ0:.3f}", fontsize=7.5)
            if c == 0:
                row = meta_df.iloc[idx]
                ax.set_ylabel(
                    f"{int(row['source_id'])}\n{row.get('population', '') or ''}",
                    rotation=0, ha="right", va="center", fontsize=6.5,
                )
    axes[0, 0].set_ylim(-span, span)
    fig.suptitle(f"{label}: residuals of the top s-process candidates", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_summary(species_tables, control, out, label):
    """Index distributions per species, over the off-line control in grey."""
    ctrl = control[np.isfinite(control)].ravel()
    fig, axes = plt.subplots(1, len(SPECIES), figsize=(5.0 * len(SPECIES), 4.2), dpi=140)
    lo, hi = np.percentile(ctrl, [0.2, 99.8])
    lo, hi = min(lo, -abs(hi)) * 3, abs(hi) * 3
    bins = np.linspace(lo, hi, 90)
    for ax, sp in zip(np.atleast_1d(axes), SPECIES):
        v = species_tables[sp][0]
        v = v[np.isfinite(v)]
        ax.hist(ctrl, bins=bins, density=True, color="lightgrey", label="off-line control")
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color="tab:red", label=sp)
        ax.axvline(0, c="k", lw=0.7, alpha=0.5)
        ax.set_xlabel("Index (positive = extra absorption)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title(sp)
    np.atleast_1d(axes)[0].set_ylabel("Density")
    fig.suptitle(f"{label}: s-process line indices against the control", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None, help="default: the npz's threshold")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="only the N worst outliers")
    p.add_argument("--top", type=int, default=25, help="candidates to tabulate and draw")
    p.add_argument(
        "--from-cache",
        action="store_true",
        help="read residuals from the cache sprocess_stack.py writes (no HDF5, no sample)",
    )
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
    label = "Full Main Sequence" if tag == "full_ms" else "Full RVS Sample"
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/sprocess")
    out_dir.mkdir(parents=True, exist_ok=True)

    check_text_rendering()

    if args.from_cache:
        cache = out_dir / "sprocess_residual_cache.npz"
        if not cache.exists():
            raise SystemExit(f"{cache} does not exist -- run sprocess_stack.py first.")
        c = np.load(cache)
        λ_grid, residual = c["lambda_grid"], c["residual"].astype(np.float64)
        meta_df = pd.DataFrame(
            {k: c[k] for k in ("source_id", "score", "ra", "dec", "bp_rp", "abs_mag_G")}
        )
        # No inverse variances in the cache: weight every pixel equally, which
        # makes the per-line indices plain window means. The control-calibrated
        # significances are unaffected; only the formal sigmas lose meaning.
        W = np.ones_like(residual)
        robust = np.ones_like(residual)
        print(f"Reusing {cache}: {residual.shape} (uniform weights)")
    else:
        state_file = args.state or (
            args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
        )
        if not state_file.exists():
            raise SystemExit(f"{state_file} does not exist -- pass --state explicitly.")
        λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
            args.weights, state_file, threshold, args.limit, args.sample
        )
        residual = Y - recon
        meta_df = pd.DataFrame(meta)
    print(f"Grid: {λ_grid[0]:.3f}-{λ_grid[-1]:.3f} nm, {len(λ_grid)} pixels, "
          f"{np.median(np.diff(λ_grid)) * 1000:.2f} pm/pixel")

    lines, every, cn_edges = load_lines(λ_grid)
    print(f"Measuring {len(lines)} lines ({', '.join(SPECIES)}) in {len(residual)} residuals...")
    per_line, species_tables, control, λ_ctrl = measure(
        λ_grid, residual, W, lines, every, cn_edges
    )

    # The control scatter is the honest denominator: the formal errors assume
    # the model is right everywhere except the line, and for an outlier it is
    # not. Per-star, so a globally badly-fit spectrum has to clear its own bar.
    ctrl_scatter = np.nanstd(control, axis=1)
    ctrl_median = np.nanmedian(control, axis=1)

    out = meta_df.copy()
    pop = load_crossmatch(Path(__file__).parent)
    out["population"] = out["source_id"].map(pop).fillna("")
    out["resid_rms"] = np.sqrt(np.mean(residual**2, axis=1))
    out["control_scatter"] = ctrl_scatter
    for sp in SPECIES:
        key = sp.replace(" ", "_").lower()
        idx, sig = species_tables[sp]
        out[f"{key}_index"] = idx
        out[f"{key}_sigma_formal"] = sig
        # Significance against the star's own off-line scatter, and centred on
        # its own off-line median so a residual pedestal does not count as a line.
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{key}_snr"] = (idx - ctrl_median) / ctrl_scatter

    # The headline ranking asks for Ce *and* Nd, not either: the minimum of the
    # two significances, so a star has to show both to rank highly. Zr is
    # reported alongside as the independent check rather than folded in, so it
    # can disagree.
    out["ce_nd_snr"] = np.minimum(out["ce_ii_snr"], out["nd_ii_snr"])
    out = out.sort_values("ce_nd_snr", ascending=False).reset_index(drop=True)

    csv = out_dir / "sprocess_indices.csv"
    out.to_csv(csv, index=False)
    print(f"Wrote {csv}")

    # Per-line detail for the top candidates, so a ranking can be checked line
    # by line against the blend distances.
    order_in_orig = out.index.to_numpy()
    src_to_row = {int(s): i for i, s in enumerate(meta_df["source_id"])}
    rows = []
    for _, r in out.head(args.top).iterrows():
        i = src_to_row[int(r["source_id"])]
        for (sp, λ0), v in per_line.items():
            rows.append(
                {
                    "source_id": int(r["source_id"]),
                    "species": sp,
                    "lambda_vac_nm": λ0,
                    "log_gf": v["log_gf"],
                    "index": v["index"][i],
                    "sigma_formal": v["sigma"][i],
                    "snr_vs_control": (v["index"][i] - ctrl_median[i]) / ctrl_scatter[i],
                    "nearest_other_nm": v["nearest_other_nm"],
                    "n_core_pix": v["n_core"],
                }
            )
    per_line_csv = out_dir / "sprocess_per_line.csv"
    pd.DataFrame(rows).to_csv(per_line_csv, index=False)
    print(f"Wrote {per_line_csv}")

    # The test that decides which lines are real. Known s-process stars against
    # a colour-matched field, since the s-process stars here are all cool.
    known_s = out["population"].str.contains("S", regex=False).to_numpy()
    field = ((out["population"] == "") & (out["bp_rp"] > 3.5)).to_numpy()
    rank_to_orig = np.array([src_to_row[int(s)] for s in out["source_id"]])
    ctrl_ranked = control[rank_to_orig]
    line_table = line_control_report(
        {k: {**v, "index": v["index"][rank_to_orig]} for k, v in per_line.items()},
        ctrl_ranked,
        {"s_star": known_s, "field": field},
    )
    line_table.to_csv(out_dir / "sprocess_line_control_test.csv", index=False)
    print(f"Wrote {out_dir / 'sprocess_line_control_test.csv'}")

    cols = ["source_id", "score", "bp_rp", "abs_mag_G", "population",
            "ce_ii_snr", "nd_ii_snr", "zr_i_snr", "ce_nd_snr", "resid_rms"]
    with pd.option_context("display.width", 220, "display.max_columns", 25):
        print(f"\nTop {args.top} by min(Ce II, Nd II) significance:")
        print(out.head(args.top)[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # How many clear the control at all, against how many would by chance.
    for sp in SPECIES:
        key = sp.replace(" ", "_").lower()
        s = out[f"{key}_snr"]
        print(f"  {sp}: {int((s > 3).sum())} stars above 3 sigma, "
              f"{int((s > 5).sum())} above 5 (median {s.median():.2f})")
    print(f"  Ce II and Nd II both above 3: {int((out['ce_nd_snr'] > 3).sum())}")
    print(f"  Ce, Nd and Zr all above 3: "
          f"{int(((out['ce_nd_snr'] > 3) & (out['zr_i_snr'] > 3)).sum())}")

    # Reorder the residual array to match the ranking for the figure.
    rank_rows = np.array([src_to_row[int(s)] for s in out["source_id"]])
    meta_ranked = meta_df.iloc[rank_rows].reset_index(drop=True)
    meta_ranked["population"] = out["population"].to_numpy()
    plot_candidates(
        λ_grid, residual[rank_rows], lines, np.arange(len(rank_rows)),
        meta_ranked, out_dir / "sprocess_candidates.pdf", label,
        n_show=min(8, len(rank_rows)),
    )
    plot_summary(species_tables, control, out_dir / "sprocess_summary.pdf", label)
    print(f"\nDone. Output in {out_dir}")
    _ = order_in_orig, robust  # kept for interactive use


if __name__ == "__main__":
    main()
