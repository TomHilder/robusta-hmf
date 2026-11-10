"""Outliers the model rejects because of the 862 nm diffuse interstellar band.

The RVS window contains one DIB, lambda-8620. It is interstellar: its strength
tracks the reddening along the sightline and has nothing to do with the star, so
a spectrum whose *only* problem is a strong DIB is an outlier caused by where
the star is rather than what it is. This finds those.

WHERE THE DIB ACTUALLY IS. Not where ``rvs_dib_features.csv`` says. That file
gives 862.6 nm, and there is no feature there. Measured on the spectra that show
it, the band is centred at 862.01 nm (8620.1 A) on the RVS grid, which matches
the catalogued lambda-8620.4 A. The line list is about 0.5 nm too red -- roughly
one full DIB width, so a window built on it misses the band completely. The
error looks like ``convert_air_to_vacuum.py`` having been applied to a value
that was already on the RVS (vacuum) scale; the Ca II triplet in the same file
is correct, and lands within 0.2 km/s of the data.

The band is also *broad*: FWHM 0.37 nm here against 0.074 nm for an unresolved
atomic line. Any index built with the narrow core/sideband windows used
elsewhere in this folder is the wrong instrument and will not see it.

HOW A DIB OUTLIER IS IDENTIFIED. Not by the size of the residual, which would
just rank the worst-fit spectra. The per-pixel robust weights say which pixels
the model actually rejected, and the outlier score is a percentile of exactly
those weights, so the question "is this star an outlier *because of* the DIB"
has a direct answer: what fraction of its downweighted pixels lie in the DIB
window. The window is 3.9% of the spectrum, so that is the chance level. The
selection asks for

    at least MIN_LOW downweighted pixels, so the fraction means something
    more than MIN_FRAC of them inside the DIB window
    net extra *absorption* there, not emission
    and ranks on (RMS inside the window) / (RMS everywhere else)

The last is what makes it "only an outlier because of the DIB": a large ratio
says the rest of the spectrum is fit well and the model's complaint is confined
to the band.

THE CHECK THAT MATTERS. DIB carriers sit in the Galactic disk, so if these are
really interstellar the candidates must lie near the plane. They do: four of the
five that survive are within 3.3 degrees of the plane, against a median |b| of
11.7 degrees for the outlier sample. Nothing in the selection knows about
position, so this is independent of everything above.

OUTPUTS (in ./plots_<tag>_final/dib by default)
    dib_candidates.csv  -- the candidates with their DIB centroid, FWHM,
                           equivalent width, downweighted-pixel fraction and
                           Galactic coordinates
    dib_<source_id>.pdf -- per candidate: the full window, and the band with the
                           rejected pixels marked

USAGE
    uv run python dib_outliers.py
    uv run python dib_outliers.py --min-frac 0.25
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mg_weak_residuals import load_cached
from plot_final_full_rvs import check_text_rendering
from plot_final_outlier_spectra import DEFAULT_WEIGHTS

try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Measured from the spectra, not taken from the line list; see the docstring.
DIB_LINE = 862.04

# Half-width of the band window, in nm. The DIB is broad, FWHM ~0.37 nm here.
DIB_HALF = 0.45

# Core used for the depth and equivalent width.
DIB_CORE = 0.25

# A pixel counts as rejected below this robust weight, matching
# plot_final_outlier_spectra.PIXEL_WEIGHT_THRESHOLD.
PIXEL_WEIGHT_THRESHOLD = 0.5

# Selection.
MIN_LOW = 15
MIN_FRAC = 0.35
MIN_DEPTH = 0.005

# The band window unavoidably overlaps a stellar blend at 861.6-861.8 (Ni I
# 861.640, Nd II 861.773), and stars with a strong residual there pass the
# fraction test while having nothing to do with the DIB. Their absorption is
# centred at 861.7; the real band is centred at 862.0. The measured centroid is
# what separates them, so it is a requirement rather than a diagnostic.
CENTROID_TOL = 0.20

# And the band is broad. An unresolved atomic line is 0.074 nm FWHM at this
# resolving power, so anything narrower than this is not a DIB.
MIN_FWHM = 0.15

# ============================================================================ #


def galactic(ra, dec):
    """Galactic l, b. NaN when astropy is missing, which is not fatal."""
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        print("note: astropy not available, so no Galactic coordinates")
        return np.full(len(ra), np.nan), np.full(len(ra), np.nan)
    g = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").galactic
    return g.l.deg, g.b.deg


def band_shape(λ_grid, residual, good, lo=861.4, hi=862.8):
    """Centroid, FWHM and equivalent width of the absorption left in the band.

    Measured on the residual, so this is the part of the DIB the model failed to
    absorb into its components -- a lower limit on the true band, not the band.
    """
    m = (λ_grid > lo) & (λ_grid < hi)
    x = λ_grid[m]
    a = np.clip(-np.where(good, residual, np.nan)[m], 0, None)
    a = np.nan_to_num(a)
    if a.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    centroid = float((x * a).sum() / a.sum())
    peak = float(a.max())
    above = x[a >= peak / 2]
    fwhm = float(above[-1] - above[0]) if len(above) > 1 else np.nan
    ew_mA = float(np.trapezoid(a, x) * 10 * 1000)
    return centroid, fwhm, peak, ew_mA


def select(λ_grid, Y, ivar, recon, robust, args):
    """Stars whose rejected pixels are concentrated in the DIB window."""
    residual = Y - recon
    good = ivar > 0
    win = np.abs(λ_grid - DIB_LINE) <= DIB_HALF
    core = np.abs(λ_grid - DIB_LINE) <= DIB_CORE
    low = (robust < PIXEL_WEIGHT_THRESHOLD) & good

    n_low = low.sum(axis=1)
    frac = low[:, win].sum(axis=1) / np.maximum(n_low, 1)
    with np.errstate(invalid="ignore"):
        depth = np.nanmean(-np.where(good, residual, np.nan)[:, core], axis=1)
        rms_in = np.sqrt(np.nanmean(np.where(good, residual, np.nan)[:, win] ** 2, axis=1))
        rms_out = np.sqrt(np.nanmean(np.where(good, residual, np.nan)[:, ~win] ** 2, axis=1))

    print(f"DIB window {λ_grid[win][0]:.2f}-{λ_grid[win][-1]:.2f} nm: {int(win.sum())} of "
          f"{len(λ_grid)} pixels ({100 * win.sum() / len(λ_grid):.1f}%, the chance level)")
    keep = (n_low >= args.min_low) & (frac > args.min_frac) & (depth > MIN_DEPTH)
    print(f"  {int(((n_low >= args.min_low) & (frac > args.min_frac)).sum())} stars with "
          f">={args.min_low} rejected pixels and >{args.min_frac:.0%} of them in the window")
    print(f"  {int(keep.sum())} of those show net absorption rather than emission")
    return keep, dict(n_low=n_low, frac_dib=frac, dib_depth=depth,
                      rms_dib=rms_in, rms_elsewhere=rms_out, ratio=rms_in / rms_out)


def plot_candidate(λ_grid, Y, ivar, recon, robust, row, out, source_id):
    """The star: full window above, the band below with rejected pixels marked."""
    good = ivar > 0
    residual = Y - recon
    flux = np.where(good, Y, np.nan)
    rej = (robust < PIXEL_WEIGHT_THRESHOLD) & good

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), dpi=140,
        gridspec_kw={"height_ratios": [2.0, 1.3, 1.6], "hspace": 0.30},
    )

    ax = axes[0]
    ax.plot(λ_grid, flux, c="k", lw=0.8, label="Data", zorder=3)
    ax.plot(λ_grid, recon, c="tab:red", lw=0.9, ls=(0, (5, 1)), label="Model", zorder=4)
    ax.axvspan(DIB_LINE - DIB_HALF, DIB_LINE + DIB_HALF, color="tab:blue", alpha=0.13, lw=0)
    ax.set_ylabel("Flux")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(λ_grid[0], λ_grid[-1])
    ax.set_title("Full RVS window", fontsize=10)

    ax = axes[1]
    ax.plot(λ_grid, np.where(good, residual, np.nan), c="k", lw=0.8, zorder=3)
    ax.scatter(λ_grid[rej], residual[rej], s=14, c="tab:orange", zorder=5, linewidths=0,
               label=f"rejected by the model ({int(rej.sum())} pixels)")
    ax.axvspan(DIB_LINE - DIB_HALF, DIB_LINE + DIB_HALF, color="tab:blue", alpha=0.13, lw=0)
    ax.axhline(0, c="k", lw=0.6, alpha=0.5)
    ax.set_ylabel("Residual")
    ax.set_xlabel("Wavelength [nm]")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(λ_grid[0], λ_grid[-1])

    ax = axes[2]
    m = np.abs(λ_grid - DIB_LINE) <= 1.1
    ax.plot(λ_grid[m], flux[m], c="k", lw=1.4, label="Data", zorder=3)
    ax.plot(λ_grid[m], recon[m], c="tab:red", lw=1.4, ls=(0, (5, 1)), label="Model", zorder=4)
    ax.fill_between(λ_grid[m], flux[m], recon[m], where=flux[m] < recon[m],
                    color="tab:blue", alpha=0.25, lw=0, label="DIB absorption")
    r = rej & m
    ax.scatter(λ_grid[r], flux[r], s=22, c="tab:orange", zorder=6, linewidths=0)
    ax.axvline(DIB_LINE, c="tab:blue", lw=1.0, ls=":")
    ax.annotate(r"DIB $\lambda$8620", (DIB_LINE, ax.get_ylim()[1]), fontsize=9,
                color="tab:blue", ha="center", va="top")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Flux")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(λ_grid[m][0], λ_grid[m][-1])
    ax.set_title("The band", fontsize=10)

    fig.suptitle(
        f"Gaia DR3 {source_id}   outlier score {row['score']:.3f}\n"
        f"{row['frac_dib']:.0%} of its {int(row['n_low'])} rejected pixels are in the DIB "
        f"window   |   band RMS / elsewhere = {row['ratio']:.1f}\n"
        f"centroid {row['dib_centroid_nm']:.3f} nm ({10 * row['dib_centroid_nm']:.1f} "
        r"$\AA$)   FWHM " f"{row['dib_fwhm_nm']:.3f} nm   EW {row['dib_ew_mA']:.0f} "
        r"m$\AA$   " f"Galactic $b$ = {row['gal_b']:+.2f}$^\\circ$",
        fontsize=11.5, y=0.99,
    )
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
    p.add_argument("--min-low", type=int, default=MIN_LOW)
    p.add_argument("--min-frac", type=float, default=MIN_FRAC)
    p.add_argument("--no-latex", action="store_true")
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False

    d0 = np.load(args.weights)
    K, Q = int(d0["best_K"]), float(d0["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d0["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    state_file = args.state or (
        args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/dib")
    out_dir.mkdir(parents=True, exist_ok=True)

    check_text_rendering()

    λ_grid, Y, ivar, recon, meta_df = load_cached(
        out_dir, args.weights, state_file, threshold, args.sample
    )
    cache = out_dir.parent / "sprocess" / "outlier_full_cache.npz"
    robust = np.load(cache)["robust"].astype(np.float64)

    keep, cols = select(λ_grid, Y, ivar, recon, robust, args)
    out = meta_df.copy()
    for k, v in cols.items():
        out[k] = v
    out["gal_l"], out["gal_b"] = galactic(out["ra"].to_numpy(), out["dec"].to_numpy())

    residual = Y - recon
    good = ivar > 0
    shape = [band_shape(λ_grid, residual[i], good[i]) for i in range(len(Y))]
    for j, name in enumerate(("dib_centroid_nm", "dib_fwhm_nm", "dib_peak", "dib_ew_mA")):
        out[name] = [s[j] for s in shape]

    on_band = (np.abs(out["dib_centroid_nm"] - DIB_LINE) <= CENTROID_TOL)
    broad = out["dib_fwhm_nm"] >= MIN_FWHM
    print(f"  {int((keep & ~on_band).sum())} rejected: absorption centred away from the band "
          f"(mostly the 861.7 nm stellar blend)")
    print(f"  {int((keep & on_band & ~broad).sum())} rejected: too narrow to be a DIB")
    keep = keep & on_band & broad
    cand = out[keep].sort_values("ratio", ascending=False)
    cand.to_csv(out_dir / "dib_candidates.csv", index=False)
    print(f"Wrote {out_dir / 'dib_candidates.csv'}")

    show = ["source_id", "score", "n_low", "frac_dib", "ratio", "dib_centroid_nm",
            "dib_fwhm_nm", "dib_ew_mA", "gal_b", "bp_rp"]
    with pd.option_context("display.width", 220):
        print(f"\n{len(cand)} candidates:")
        print(cand[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nGalactic latitude: candidates {np.abs(cand['gal_b']).round(2).tolist()}, "
          f"median |b| of all {len(out)} outliers {np.nanmedian(np.abs(out['gal_b'])):.1f} deg")

    # Positional, not by source id: iterrows() casts the row to a common dtype
    # and an int64 Gaia source id does not survive the trip through float64.
    src = {int(s): i for i, s in enumerate(meta_df["source_id"])}
    for pos in range(len(cand)):
        sid = int(cand["source_id"].iloc[pos])
        i = src[sid]
        plot_candidate(λ_grid, Y[i], ivar[i], recon[i], robust[i], cand.iloc[pos],
                       out_dir / f"dib_{sid}.pdf", sid)
    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
