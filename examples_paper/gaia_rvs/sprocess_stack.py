"""Stack the outlier residuals by population to show the s-process lines.

``sprocess_line_residuals.py`` measures a Ce II / Nd II / Zr I index per star and
finds that no single spectrum clears its own noise: the RVS lines are weak and
one spectrum is not enough. Stacking is the way to see them. The known S stars
in the outlier set are the positive control -- if the residual really carries an
s-process signal, their median residual has dips at the Ce II, Nd II and Zr I
wavelengths and the field's does not.

That comparison also calibrates the search. The S stars say what an enhanced
residual looks like; stars with the same signature that are *not* already
catalogued as S, C or Ba are the candidates worth following up, and they are
what this writes out.

The field comparison is colour-matched (``--bp-rp-min``): the S stars are all
cool giants, and cool giants are fit worse than the rest of the sample for
reasons that have nothing to do with the s-process, so comparing them to the
whole outlier list would credit the s-process with a temperature effect.

OUTPUTS (in ./plots_<tag>_final/sprocess by default)
    sprocess_stacks.pdf         -- median residual of S stars vs colour-matched
                                   field, over the full grid and per line
    sprocess_new_candidates.csv -- stars matching the S-star signature that are
                                   not in any catalogue used here
    sprocess_residual_cache.npz -- residuals, ivar and grid, so re-running the
                                   figures does not re-read the HDF5

USAGE
    uv run python sprocess_stack.py
    uv run python sprocess_stack.py --reuse-cache --n-boot 2000
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

# The S stars are cool giants; the field they are compared against is cut to the
# same colour so the contrast is composition rather than temperature.
BP_RP_MIN = 3.5

# Bootstrap resamples for the band on each median stack.
N_BOOT = 1000
SEED = 42

# Candidates are selected on Ce II 853.276 alone. It is the only line in the
# window whose S-star index beats every control wavelength (see the control test
# in sprocess_line_residuals.py): Nd II 855.933 is 0.11 nm from H I Pa13 and its
# strongest "detections" are Paschen emission stars, and the Zr I lines are
# consistent with the control, zirconium in an S star being mostly ZrO.
CE_LINE = 853.276

# Significance against the star's own off-line control scatter.
CANDIDATE_Z = 3.0

# ============================================================================ #


def stack(residual, rows, n_boot=N_BOOT, seed=SEED):
    """Median residual over *rows*, with a bootstrap 16-84 band."""
    r = residual[rows]
    med = np.median(r, axis=0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, r.shape[1]))
    for i in range(n_boot):
        boots[i] = np.median(r[rng.integers(0, len(r), len(r))], axis=0)
    lo, hi = np.percentile(boots, [16, 84], axis=0)
    return med, lo, hi


def plot_stacks(λ_grid, residual, s_rows, f_rows, lines, out, label, n_boot=N_BOOT):
    """The two stacks over the full grid, then zoomed on every measured line.

    Top panel is the whole window with the s-process lines marked; the grid
    below is one panel per line, on a shared scale, which is where a dip either
    is or is not at zero offset.
    """
    s_med, s_lo, s_hi = stack(residual, s_rows, n_boot)
    f_med, f_lo, f_hi = stack(residual, f_rows, n_boot)

    λ0s = lines["lambda_vac_nm"].to_numpy()
    sps = lines["species"].to_numpy()
    ncol = 4
    nrow = int(np.ceil(len(λ0s) / ncol))

    fig = plt.figure(figsize=(15, 3.6 + 2.3 * nrow), dpi=140)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[2.4] + [1] * nrow, hspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(λ_grid, f_lo, f_hi, color="grey", alpha=0.3, lw=0)
    ax.plot(λ_grid, f_med, c="grey", lw=1.0, label=f"field, BP-RP > {BP_RP_MIN} (n={len(f_rows)})")
    ax.fill_between(λ_grid, s_lo, s_hi, color="tab:red", alpha=0.3, lw=0)
    ax.plot(λ_grid, s_med, c="tab:red", lw=1.2, label=f"known S stars (n={len(s_rows)})")
    ax.axhline(0, c="k", lw=0.6, alpha=0.5)
    for λ0, sp in zip(λ0s, sps):
        ax.axvline(λ0, c="tab:blue", lw=0.7, alpha=0.45, zorder=0)
    ax.set_xlim(λ_grid[0], λ_grid[-1])
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Median residual")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Median residual, s-process lines marked")

    axes = []
    for i, (λ0, sp) in enumerate(zip(λ0s, sps)):
        a = fig.add_subplot(gs[1 + i // ncol, i % ncol])
        m = np.abs(λ_grid - λ0) <= SIDE_OUTER
        x = λ_grid[m] - λ0
        a.fill_between(x, f_lo[m], f_hi[m], color="grey", alpha=0.3, lw=0)
        a.plot(x, f_med[m], c="grey", lw=1.1)
        a.fill_between(x, s_lo[m], s_hi[m], color="tab:red", alpha=0.3, lw=0)
        a.plot(x, s_med[m], c="tab:red", lw=1.4)
        a.axhline(0, c="k", lw=0.5, alpha=0.4)
        a.axvspan(-HALF_CORE, HALF_CORE, color="tab:blue", alpha=0.12, lw=0)
        a.set_title(f"{sp} {λ0:.3f}", fontsize=9)
        a.set_xlabel(r"$\lambda - \lambda_0$ [nm]", fontsize=8)
        a.tick_params(labelsize=7)
        axes.append(a)
    # One scale across the zooms, set by the stacks themselves.
    span = 1.15 * max(np.abs(s_med).max(), np.abs(f_med).max())
    for a in axes:
        a.set_ylim(-span, span)
    axes[0].set_ylabel("Median residual", fontsize=8)

    fig.suptitle(f"{label}: stacked residuals at the s-process lines", fontsize=14, y=0.98)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return s_med, f_med


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    p.add_argument("--sample", default="all", choices=("ms", "all"))
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--bp-rp-min", type=float, default=BP_RP_MIN)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--no-latex", action="store_true")
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False

    d = np.load(args.weights)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    label = "Full Main Sequence" if tag == "full_ms" else "Full RVS Sample"
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/sprocess")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "sprocess_residual_cache.npz"

    check_text_rendering()

    if args.reuse_cache and cache.exists():
        c = np.load(cache)
        λ_grid, residual = c["lambda_grid"], c["residual"]
        meta_df = pd.DataFrame(
            {k: c[k] for k in ("source_id", "score", "ra", "dec", "bp_rp", "abs_mag_G")}
        )
        print(f"Reusing {cache}: {residual.shape}")
    else:
        state_file = args.state or (
            args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
        )
        λ_grid, Y, W, recon, robust, meta = load_outlier_inputs(
            args.weights, state_file, threshold, None, args.sample
        )
        residual = Y - recon
        meta_df = pd.DataFrame(meta)
        np.savez_compressed(
            cache, lambda_grid=λ_grid, residual=residual.astype(np.float32),
            **{k: meta_df[k].to_numpy() for k in meta_df.columns},
        )
        print(f"Wrote {cache}")

    lines, _, _ = load_lines()
    pop = load_crossmatch(Path(__file__).parent)
    meta_df["population"] = meta_df["source_id"].astype(int).map(pop).fillna("")

    is_s = meta_df["population"].str.contains("S", regex=False).to_numpy()
    known = (meta_df["population"] != "").to_numpy()
    red = (meta_df["bp_rp"].to_numpy() > args.bp_rp_min)
    s_rows = np.flatnonzero(is_s)
    f_rows = np.flatnonzero(red & ~known)
    print(f"{len(s_rows)} known S stars, {len(f_rows)} colour-matched field outliers")

    plot_stacks(
        λ_grid, residual, s_rows, f_rows, lines,
        out_dir / "sprocess_stacks.pdf", label, args.n_boot,
    )

    # Candidates, on Ce II 853.276 alone. The significance is against each
    # star's own control wavelengths, so a spectrum the model fits badly
    # everywhere has to clear its own bar rather than the sample's.
    ivar = np.ones_like(residual)
    λ_ctrl = control_wavelengths(λ_grid, *load_lines()[1:])
    ctrl = np.column_stack([line_index(λ_grid, residual, ivar, λ0)[0] for λ0 in λ_ctrl])
    ce = line_index(λ_grid, residual, ivar, CE_LINE)[0]
    z = (ce - np.median(ctrl, axis=1)) / ctrl.std(axis=1)
    meta_df["ce853_index"], meta_df["ce853_z"] = ce, z
    meta_df.to_csv(out_dir / "ce853_per_star.csv", index=False)

    # Recovery of the known S stars is the only handle on how well the cut
    # works, so it is reported rather than assumed.
    chance = np.mean(ctrl > (np.median(ctrl, axis=1)[:, None] + CANDIDATE_Z * ctrl.std(axis=1)[:, None]))
    print(f"\nCe II {CE_LINE} nm, z > {CANDIDATE_Z} against each star's own controls:")
    print(f"  known S stars: {int((z[s_rows] > CANDIDATE_Z).sum())}/{len(s_rows)} "
          f"({100 * (z[s_rows] > CANDIDATE_Z).mean():.0f}%)")
    print(f"  colour-matched field: {int((z[f_rows] > CANDIDATE_Z).sum())}/{len(f_rows)} "
          f"({100 * (z[f_rows] > CANDIDATE_Z).mean():.0f}%)")
    print(f"  expected by chance from the controls: {100 * chance:.1f}%")

    uncat = (meta_df["population"] == "").to_numpy()
    cand = meta_df[uncat & (z > CANDIDATE_Z)].sort_values("ce853_z", ascending=False)
    cand.to_csv(out_dir / "sprocess_new_candidates.csv", index=False)
    cols = ["source_id", "score", "bp_rp", "abs_mag_G", "ce853_index", "ce853_z"]
    with pd.option_context("display.width", 200):
        print(f"\n{len(cand)} uncatalogued candidates:")
        print(cand.head(30)[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nWrote {out_dir / 'sprocess_new_candidates.csv'}")


if __name__ == "__main__":
    main()
