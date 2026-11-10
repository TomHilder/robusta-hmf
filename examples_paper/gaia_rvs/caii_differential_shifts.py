"""Ca II triplet lines that disagree with each other about the radial velocity.

A star has one velocity. The three Ca II triplet lines (850.035, 854.444,
866.452 nm vacuum) are the same ion in the same atmosphere, so they must all sit
at the same redshift. When one of them does not -- one line shifted and the
other two not, or two shifted in opposite directions -- the star is not a single
star with a velocity error. Something is adding a second component: a companion
whose lines blend in, chromospheric emission filling one core asymmetrically, a
circumstellar or interstellar absorber, or a blend that only touches one member
of the triplet.

A *common* shift of all three is the uninteresting case -- that is just an error
in the catalogue radial velocity used to put the spectrum in the rest frame --
and it is projected out rather than ranked on.

HOW THE SHIFT IS MEASURED. For a small shift the data are the model displaced,
so to first order the residual is proportional to the model's derivative:

    D(lambda) = M(lambda - dlambda) ~= M(lambda) - dlambda * M'(lambda)
    => residual = D - M = -dlambda * M'

so dlambda comes from projecting the residual onto M'. The fit in each window
carries two nuisance terms alongside M': the model itself, which absorbs a line
that is too deep or too shallow, and a constant for the continuum. Without them
a depth error leaks into the shift. The fit is iterated twice, shifting the
model by the current estimate each time, because the linear step alone
compresses large shifts (~7% at 10 km/s).

Validated by injection: shifting the data by a known velocity and recovering it
returns +1.96 for +2.00 km/s, -4.77 for -5.00, and +9.35 for +10.00 with the
single linear step; the iteration removes most of that compression.
``--injection-test`` reruns this.

ERRORS. The formal errors from the fit are wrong by a factor of ~8: they assume
the model is right everywhere except for the shift, and these are outlier
spectra where it is not. Each window's error is inflated by the square root of
its own reduced chi-square, which is the standard fix and takes the median
chi-square of the "all three agree" hypothesis from 39 down to about 6. The
remainder is calibrated empirically against a control (below) rather than
assumed away.

PER-LINE SYSTEMATICS ARE REMOVED FIRST, AND THEY DEPEND ON COLOUR. Each triplet
member has its own neighbours, and a blend pulls its centroid the same way in
every star of a given temperature -- but by a different amount at a different
temperature. Measured here, the differential offset of CaT 8542 swings by 0.77
km/s between the bluest and reddest stars. A single median per line would leave
that as a bias and invent disagreements at the ends of the colour range, so a
running median in BP - RP is subtracted instead.

CALIBRATION. Even after the per-window inflation the errors are still ~2.4x too
small. A differential shift is unphysical for an ordinary star, so almost every
spectrum here should be consistent with "all three agree"; matching the *median*
chi-square to its 2-dof expectation therefore calibrates on the stars that have
nothing wrong with them and leaves the tail as the result. The threshold is set
where fewer than one star in the sample is expected by chance.

Triplets of strong Fe I lines go through the identical procedure as a
cross-check. They disagree *more* than Ca II does, being weaker and more
blended, so they bound the method rather than calibrate it -- using their
distribution as the cut would throw away everything.

THE LINE HAS TO BE THERE. The triplet is strong in cool stars and weak in warm
ones, and a window with no line still returns a shift, fitted to whatever else
is in it. Every candidate here is warm, so this is not academic: each line must
reach MIN_DEPTH in the data or the star is dropped. That cut removes more than
half of what would otherwise be reported.

EMISSION IS THE OBVIOUS CONFOUND. Ca II triplet cores go into emission in
chromospherically active stars, which shifts a centroid without anything moving.
The core residual of each line is reported so an "emission" star can be told
from a genuinely displaced one, and the per-star figures show the profiles.

OUTPUTS (in ./plots_<tag>_final/caii_shifts by default)
    caii_shifts.csv        -- per star: three velocities and errors, the common
                              velocity, the three differential residuals, their
                              significances, chi2_diff, the pattern, core
                              emission indicators
    caii_summary.pdf       -- chi2_diff against the Fe I control, and the
                              differential velocities against each other
    caii_patterns.pdf      -- the three differential velocities per candidate
    spectra/               -- per-star figures, in subfolders by pattern:
                              single_line_shifted/, opposite_shifts/

USAGE
    uv run python caii_differential_shifts.py
    uv run python caii_differential_shifts.py --injection-test
    uv run python caii_differential_shifts.py --t-min 4 --n-figures 40
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mg_weak_residuals import load_cached
from plot_final_full_rvs import check_text_rendering
from plot_final_outlier_spectra import DEFAULT_WEIGHTS
from sprocess_line_residuals import load_crossmatch

try:
    plt.style.use("mpl_drip.custom")
except OSError:
    print("note: the 'mpl_drip.custom' style is not installed; using matplotlib defaults")

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

C_KMS = 299792.458

# The triplet, vacuum, from rvs_strong_features.csv.
CA_LINES = (850.035, 854.444, 866.452)
CA_NAMES = ("CaT 8498", "CaT 8542", "CaT 8662")

# Strong, relatively clean Fe I lines for the control triplets.
FE_CONTROL = (850.201, 851.80, 858.741, 860.12, 867.715, 869.098)

# Half-width of the fitting window, in nm. Wide enough to hold the core and the
# inner wings, where the derivative signal is, and narrow enough to stay off the
# neighbours.
HALF_WINDOW = 0.20

# Newton iterations on the shift.
N_ITER = 2

# A line counts as discrepant at this significance against its own inflated
# error.
T_MIN = 3.0

# Core half-width for the emission indicator, in nm.
CORE_HALF = 0.03

# A triplet line shallower than this is not measurably present, and a shift
# fitted in a window with no line is fitting whatever else is in it.
MIN_DEPTH = 0.02

# Velocity grid the observed line profiles are resampled onto for the
# model-independent cross-check, in km/s.
VELOCITY_GRID = np.arange(-180.0, 180.01, 1.0)

N_FIGURES = 25

# ============================================================================ #


def _shift_model(λ_grid, M, v_kms):
    """The model displaced by *v_kms*, one velocity per row.

    Redshift convention: a positive velocity moves features to longer
    wavelengths, so the shifted model is sampled at lambda / (1 + v/c).
    """
    out = np.empty_like(M)
    for i in range(len(M)):
        out[i] = np.interp(λ_grid / (1.0 + v_kms[i] / C_KMS), λ_grid, M[i])
    return out


def fit_line_shift(λ_grid, Y, ivar, recon, λ0, half=HALF_WINDOW, n_iter=N_ITER):
    """Velocity of one line relative to the model, per star.

    Returns ``(v, sigma, chi2_red, n_pix)``. The design matrix is
    ``[M', M - <M>, 1]``: the derivative carries the shift, the model absorbs a
    depth error, and the constant a continuum error, so a line the model gets
    too deep is not read as a displacement.
    """
    m = np.abs(λ_grid - λ0) <= half
    λw = λ_grid[m]
    dλ = float(np.median(np.diff(λ_grid)))
    w = ivar[:, m]
    v = np.zeros(len(Y))
    sig = np.full(len(Y), np.nan)
    chi2_red = np.ones(len(Y))

    for it in range(n_iter):
        # Compare the data against the model already displaced by the current
        # estimate, so each pass solves for a small correction and the linear
        # approximation stays valid.
        M_full = recon if it == 0 else _shift_model(λ_grid, recon, v)
        M = M_full[:, m]
        r = Y[:, m] - M
        Mp = np.gradient(M, dλ, axis=1)
        X = np.stack([Mp, M - M.mean(axis=1, keepdims=True), np.ones_like(M)], axis=2)
        XtW = X.transpose(0, 2, 1) * w[:, None, :]
        A = XtW @ X
        # A ridge far below the smallest real curvature, so a fully masked
        # window returns a huge error rather than raising.
        A = A + np.eye(3) * (1e-10 * np.trace(A, axis1=1, axis2=2)[:, None, None] + 1e-30)
        beta = np.linalg.solve(A, XtW @ r[:, :, None])[:, :, 0]
        cov = np.linalg.inv(A)
        resid = r - (X @ beta[:, :, None])[:, :, 0]
        dof = max(int(m.sum()) - 3, 1)
        chi2_red = (w * resid**2).sum(axis=1) / dof
        dv = C_KMS * (-beta[:, 0]) / λ0
        v = v + dv
        # Inflated by this window's own misfit: the formal error assumes the
        # model is right, and for an outlier it is not.
        sig = C_KMS * np.sqrt(np.abs(cov[:, 0, 0])) / λ0 * np.sqrt(np.maximum(chi2_red, 1.0))

    n_good = (w > 0).sum(axis=1)
    sig = np.where(n_good > 10, sig, np.inf)
    return v, sig, chi2_red, n_good, λw


def detrend_by_colour(V, bp_rp, n_bin=120):
    """Remove each line's systematic shift as a function of colour.

    A blend pulls a line's centroid the same way in every star of a given
    temperature, and *differently* at a different temperature: measured on this
    sample the differential offset of CaT 8542 swings by 0.77 km/s from the
    bluest stars to the reddest, and 8498 by 0.6 km/s the other way. Subtracting
    one number per line would leave that as a colour-dependent bias and would
    manufacture disagreements at the ends of the colour range. This subtracts a
    running median in colour instead, so what is left is the star against others
    of its own temperature.
    """
    order = np.argsort(bp_rp)
    out = np.empty_like(V)
    n_knot = max(len(bp_rp) // n_bin, 3)
    edges = np.linspace(0, len(bp_rp), n_knot + 1).astype(int)
    for k in range(V.shape[0]):
        vs = V[k][order]
        centres, meds = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                centres.append(np.median(bp_rp[order][a:b]))
                meds.append(np.median(vs[a:b]))
        out[k] = V[k] - np.interp(bp_rp, np.array(centres), np.array(meds))
    return out


def differential(V, S, bp_rp=None):
    """Split per-line velocities into a common part and disagreements.

    *V* and *S* are (n_lines, n_stars). Each line's systematic is removed first
    -- as a function of colour when *bp_rp* is given, otherwise as a single
    median -- so a blend that pulls one member the same way in every star is not
    read as that star disagreeing. Returns the common velocity, the residuals
    about it, their significances and the chi-square of "all lines agree".
    """
    if bp_rp is not None:
        Vc = detrend_by_colour(V, bp_rp)
    else:
        Vc = V - np.median(V, axis=1, keepdims=True)
    w = 1.0 / S**2
    w = np.where(np.isfinite(w), w, 0.0)
    common = (w * np.nan_to_num(Vc)).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-30)
    d = Vc - common
    t = d / S
    chi2 = np.nansum(np.where(np.isfinite(t), t, 0.0) ** 2, axis=0)
    return Vc, common, d, t, chi2


def classify(t, t_min=T_MIN):
    """Name the pattern of disagreement, or "" when there is none.

    Only the two shapes asked about are named: one line displaced with the
    others quiet, and two lines displaced in opposite directions. Everything
    else, including all three moving together, is left unlabelled.
    """
    out = []
    for col in t.T:
        good = col[np.isfinite(col)]
        if len(good) < 3:
            out.append("incomplete")
            continue
        hot = np.abs(col) > t_min
        if hot.sum() == 1:
            out.append("single_line_shifted")
        elif hot.sum() >= 2 and col[hot].max() > t_min and col[hot].min() < -t_min:
            out.append("opposite_shifts")
        elif hot.sum() >= 2:
            out.append("multi_same_sign")
        else:
            out.append("")
    return np.array(out)


def core_emission(λ_grid, residual, λ0, half=CORE_HALF):
    """Mean residual in the line core: positive means emission filling it in."""
    m = np.abs(λ_grid - λ0) <= half
    return residual[:, m].mean(axis=1)


def line_depth(λ_grid, Y, ivar, λ0, half=CORE_HALF, s0=0.09, s1=0.26):
    """Depth of the line in the data, per star: sideband continuum minus core.

    A shift can only be measured for a line that is there. The triplet is strong
    in cool stars and weak in warm ones, and a window with no line still returns
    a number -- driven by whatever else is in it -- so this is the gate rather
    than a diagnostic.
    """
    d = np.abs(λ_grid - λ0)
    core, side = d <= half, (d >= s0) & (d <= s1)
    f = np.where(ivar > 0, Y, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(f[:, side], axis=1) - np.nanmean(f[:, core], axis=1)


def rescale_errors(chi2, dof=2):
    """The factor the errors are underestimated by, from the bulk of the sample.

    A differential shift is unphysical for an ordinary star, so nearly every
    spectrum here should be consistent with "all three lines agree". Matching
    the *median* chi-square to its expectation under that hypothesis therefore
    calibrates the errors on the stars that have nothing wrong with them, and
    leaves the tail to be the result. The median is used rather than the mean
    because the tail is exactly what must not influence the calibration.
    """
    from scipy import stats

    k2 = float(np.median(chi2) / stats.chi2.median(dof))
    return k2


def line_profiles(λ_grid, Y, rows, v_grid=None):
    """Each triplet line's observed profile on a common velocity grid.

    Continuum from the far wings, then divided by its own depth, so only the
    *shape* survives: the three lines have very different strengths and this
    must not be read as a difference in position.
    """
    v_grid = VELOCITY_GRID if v_grid is None else v_grid
    P = np.empty((len(CA_LINES), len(rows), len(v_grid)))
    for k, λ0 in enumerate(CA_LINES):
        wl = λ0 * (1 + v_grid / C_KMS)
        for j, i in enumerate(rows):
            p = np.interp(wl, λ_grid, Y[i])
            cont = np.percentile(p[np.abs(v_grid) > 120], 90)
            d = cont - p
            P[k, j] = d / max(d.max(), 1e-6)
    return P


def pairwise_line_dv(P, a, b, lag_max=60):
    """Velocity difference between two triplet lines, from the data alone.

    Cross-correlates one line's profile against the other's and interpolates the
    peak. Nothing here refers to the model, which is the point: the ranking
    statistic is built from residuals against a rank-16 reconstruction that fits
    the Ca II profile poorly in warm stars, so a profile mismatch can imitate a
    displacement. This cannot -- if two lines of the same star really do sit at
    different velocities, they disagree here too.
    """
    lags = np.arange(-lag_max, lag_max + 1)
    dv = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        fa, fb = P[a, j], P[b, j]
        cc = np.array([
            np.corrcoef(fa[lag_max:-lag_max], np.roll(fb, int(m))[lag_max:-lag_max])[0, 1]
            for m in lags
        ])
        i = int(np.argmax(cc))
        if 0 < i < len(cc) - 1:
            y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
            dv[j] = lags[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        else:
            dv[j] = lags[i]
    return dv * float(VELOCITY_GRID[1] - VELOCITY_GRID[0])


def confirm_model_independent(λ_grid, Y, cand_rows, control_rows):
    """Do the candidates still disagree when the model is taken out of it?

    Returns the pairwise differences for the candidates, and prints them beside
    a random control drawn from stars that passed the depth cut. A candidate
    sample that matches the control here would mean the ranking is measuring the
    reconstruction rather than the star.
    """
    print("\nModel-independent check: line against line, no reconstruction involved")
    out = {}
    for name, rows in (("candidates", cand_rows), ("random control", control_rows)):
        P = line_profiles(λ_grid, Y, rows)
        print(f"  {name} (n={len(rows)}):")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            dv = pairwise_line_dv(P, a, b)
            mad = 1.4826 * np.median(np.abs(dv - np.median(dv)))
            print(f"    {CA_NAMES[a]} - {CA_NAMES[b]}: median {np.median(dv):+6.2f} km/s, "
                  f"MAD {mad:5.2f}, |dv| > 10 in {int((np.abs(dv) > 10).sum())}/{len(dv)}")
            if name == "candidates":
                out[f"dv_xcorr_{CA_NAMES[a][-4:]}_{CA_NAMES[b][-4:]}"] = dv
    return out


def control_chi2(λ_grid, Y, ivar, recon, lines=FE_CONTROL, max_triplets=10):
    """chi2_diff for triplets of Fe I lines: what "no disagreement" looks like.

    These lines are also one species in one atmosphere, so they should agree.
    Their distribution calibrates the Ca II threshold. It is only approximate --
    Fe I is weaker than Ca II and measured less precisely -- so it is used to
    place a cut, not to quote a probability.
    """
    fits = {λ0: fit_line_shift(λ_grid, Y, ivar, recon, λ0)[:2] for λ0 in lines}
    out = []
    for trip in itertools.islice(itertools.combinations(lines, 3), max_triplets):
        V = np.array([fits[λ0][0] for λ0 in trip])
        S = np.array([fits[λ0][1] for λ0 in trip])
        out.append(differential(V, S)[4])
    return np.concatenate(out)


def injection_test(λ_grid, Y, ivar, recon, λ0=854.444, n=400):
    """Recover a known velocity, to show the estimator is unbiased and signed."""
    print("\nInjection test (redshift convention D(lambda) = Y(lambda / (1 + v/c))):")
    base = fit_line_shift(λ_grid, Y, ivar, recon, λ0)[0]
    for v_true in (2.0, -5.0, 10.0, -20.0):
        Y2 = Y.copy()
        for j in range(n):
            Y2[j] = np.interp(λ_grid / (1.0 + v_true / C_KMS), λ_grid, Y[j])
        v = fit_line_shift(λ_grid, Y2, ivar, recon, λ0)[0]
        rec = np.median((v - base)[:n])
        print(f"  injected {v_true:+7.2f} km/s -> recovered {rec:+7.2f}  "
              f"(bias {100 * (rec - v_true) / v_true:+.1f}%)")


def plot_summary(chi2, ctrl, d, out, label):
    """chi2 against the control, and the disagreements against each other."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=140)

    ax = axes[0]
    bins = np.logspace(-1, 3.2, 70)
    ax.hist(np.clip(ctrl, 1e-1, None), bins=bins, density=True, color="lightgrey",
            label=f"Fe I control triplets (n={len(ctrl)})")
    ax.hist(np.clip(chi2, 1e-1, None), bins=bins, density=True, histtype="step",
            lw=1.7, color="tab:red", label=f"Ca II triplet (n={len(chi2)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\chi^2$ of 'all three lines agree' (2 dof)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title("Disagreement, against the control")

    pairs = ((0, 1), (0, 2))
    for ax, (i, j) in zip(axes[1:], pairs):
        ax.scatter(d[i], d[j], s=7, c=np.log10(np.maximum(chi2, 0.1)), cmap="viridis",
                   linewidths=0)
        lim = np.nanpercentile(np.abs(d), 99.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, c="k", lw=0.6, alpha=0.5)
        ax.axvline(0, c="k", lw=0.6, alpha=0.5)
        ax.set_xlabel(f"{CA_NAMES[i]} $-$ common [km/s]")
        ax.set_ylabel(f"{CA_NAMES[j]} $-$ common [km/s]")
        ax.set_title("Opposite shifts land off-diagonal")
    fig.suptitle(f"{label}: Ca II triplet differential velocities", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_star_lines(λ_grid, Y, ivar, recon, residual, row, V, S, out):
    """One star: each triplet line, data over model, with its fitted shift.

    Three columns, one per line, flux above and residual below. The fitted
    velocity is printed on each panel, so the disagreement the ranking is built
    on can be checked against the profiles that produced it.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=130, sharex="col")
    good = ivar > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(good, 1.0 / np.sqrt(np.where(good, ivar, 1.0)), np.nan)
    flux = np.where(good, Y, np.nan)

    for k, (λ0, name) in enumerate(zip(CA_LINES, CA_NAMES)):
        m = np.abs(λ_grid - λ0) <= 0.45
        x = λ_grid[m] - λ0
        a0, a1 = axes[0, k], axes[1, k]
        a0.plot(x, flux[m], c="k", lw=1.2, label="Data")
        a0.plot(x, recon[m], c="tab:red", lw=1.2, ls=(0, (5, 1)), label="Model")
        a0.axvline(0, c="grey", lw=0.8, ls=":")
        a0.axvspan(-HALF_WINDOW, HALF_WINDOW, color="tab:blue", alpha=0.07, lw=0)
        a0.set_title(f"{name} {λ0} nm\n$v = {V[k]:+.2f} \\pm {S[k]:.2f}$ km/s", fontsize=10)
        if k == 0:
            a0.set_ylabel("Flux")
            a0.legend(fontsize=8, loc="lower right")

        a1.fill_between(x, -sigma[m], sigma[m], color="tab:blue", alpha=0.2, lw=0)
        a1.plot(x, np.where(good, residual, np.nan)[m], c="k", lw=1.2)
        a1.axhline(0, c="k", lw=0.6, alpha=0.5)
        a1.axvline(0, c="grey", lw=0.8, ls=":")
        a1.set_xlabel(rf"$\lambda - {λ0}$ [nm]")
        if k == 0:
            a1.set_ylabel("Residual")

    pop = row.get("population", "") or "uncatalogued"
    fig.suptitle(
        f"Gaia DR3 {int(row['source_id'])}   [{pop}]   {row['pattern']}\n"
        f"$\\chi^2$(agree) = {row['chi2_diff']:.1f}   "
        f"common $v$ = {row['v_common']:+.2f} km/s   "
        f"max pairwise $\\Delta v$ = {row['max_pair_dv']:.2f} km/s   "
        f"score {row['score']:.3f}   BP$-$RP {row['bp_rp']:.2f}",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    p.add_argument("--sample", default="all_filtered", choices=("ms", "all", "all_filtered"))
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--t-min", type=float, default=T_MIN)
    p.add_argument("--min-depth", type=float, default=MIN_DEPTH,
                   help="each triplet line must be at least this deep in the data")
    p.add_argument("--n-figures", type=int, default=N_FIGURES)
    p.add_argument("--injection-test", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-latex", action="store_true")
    args = p.parse_args()

    if args.no_latex:
        plt.rcParams["text.usetex"] = False

    d0 = np.load(args.weights)
    K, Q = int(d0["best_K"]), float(d0["best_Q"])
    threshold = args.threshold if args.threshold is not None else float(d0["threshold"])
    tag = args.weights.name.replace("_final_weights.npz", "")
    label = "Full Main Sequence" if tag == "full_ms" else "Full RVS Sample"
    state_file = args.state or (
        args.weights.parent / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    out_dir = args.out_dir or Path(f"./plots_{tag}_final/caii_shifts")
    out_dir.mkdir(parents=True, exist_ok=True)

    check_text_rendering()

    λ_grid, Y, ivar, recon, meta_df = load_cached(
        out_dir, args.weights, state_file, threshold, args.sample
    )
    residual = Y - recon

    if args.injection_test:
        injection_test(λ_grid, Y, ivar, recon)

    print(f"\nFitting the triplet in {len(Y)} spectra...", flush=True)
    fits = [fit_line_shift(λ_grid, Y, ivar, recon, λ0) for λ0 in CA_LINES]
    V = np.array([f[0] for f in fits])
    S = np.array([f[1] for f in fits])
    for λ0, name, v, s in zip(CA_LINES, CA_NAMES, V, S):
        print(f"  {name} {λ0}: median {np.median(v):+.3f} km/s, "
              f"scatter (MAD) {1.4826 * np.median(np.abs(v - np.median(v))):.3f}, "
              f"median error {np.median(s[np.isfinite(s)]):.3f}")

    Vc, common, d, t, chi2_raw = differential(V, S, meta_df["bp_rp"].to_numpy())

    # Calibrate on the bulk, where the lines should agree, then threshold where
    # fewer than one star is expected by chance.
    k2 = rescale_errors(chi2_raw)
    chi2 = chi2_raw / k2
    S = S * np.sqrt(k2)
    t = t / np.sqrt(k2)
    cut = 2.0 * np.log(len(chi2))
    print(f"\nErrors underestimated by {np.sqrt(k2):.2f}x beyond the per-window inflation "
          f"(median chi2 {np.median(chi2_raw):.2f} -> {np.median(chi2):.2f})")
    print(f"  threshold chi2 > {cut:.1f}, where <1 of {len(chi2)} is expected by chance: "
          f"{int((chi2 > cut).sum())} stars")
    pattern = classify(t, args.t_min)

    # The line has to be there. The triplet is strong in cool stars and weak in
    # warm ones, and every candidate here is warm, so this is not academic.
    depths = np.array([line_depth(λ_grid, Y, ivar, λ0) for λ0 in CA_LINES])
    present = np.all(depths > args.min_depth, axis=0)
    print(f"  Ca II present in all three lines (depth > {args.min_depth}): "
          f"{int(present.sum())} of {len(present)} stars")

    print("Fitting the Fe I control triplets (a cross-check, not the cut)...", flush=True)
    ctrl = control_chi2(λ_grid, Y, ivar, recon) / k2
    print(f"  Fe I control chi2 median {np.median(ctrl):.2f} vs Ca II {np.median(chi2):.2f} -- "
          f"the Fe I lines disagree more,\n  being weaker and more blended, so they bound the "
          f"method rather than calibrate it")

    out = meta_df.copy()
    pop = load_crossmatch(Path(__file__).parent)
    out["population"] = out["source_id"].astype(int).map(pop).fillna("")
    for k, name in enumerate(CA_NAMES):
        key = name.replace(" ", "_").lower()
        out[f"v_{key}"] = V[k]
        out[f"sig_{key}"] = S[k]
        out[f"dv_{key}"] = d[k]
        out[f"t_{key}"] = t[k]
        out[f"core_resid_{key}"] = core_emission(λ_grid, residual, CA_LINES[k])
    out["v_common"] = common
    out["chi2_diff"] = chi2
    out["chi2_diff_raw"] = chi2_raw
    for k, name in enumerate(CA_NAMES):
        out[f"depth_{name.replace(' ', '_').lower()}"] = depths[k]
    out["caii_present"] = present
    out["max_pair_dv"] = np.nanmax(d, axis=0) - np.nanmin(d, axis=0)
    out["n_lines_discrepant"] = (np.abs(t) > args.t_min).sum(axis=0)
    out["pattern"] = pattern
    out["control_cut"] = cut
    out = out.sort_values("chi2_diff", ascending=False).reset_index(drop=True)
    out.to_csv(out_dir / "caii_shifts.csv", index=False)
    print(f"Wrote {out_dir / 'caii_shifts.csv'}")

    wanted = out[out["pattern"].isin(("single_line_shifted", "opposite_shifts"))]
    n_before = int((wanted["chi2_diff"] > cut).sum())
    wanted = wanted[(wanted["chi2_diff"] > cut) & wanted["caii_present"]]
    print(f"\n{n_before - len(wanted)} of {n_before} dropped for having no measurable Ca II line")
    print(f"\nPatterns asked for, above the control cut ({cut:.1f}):")
    for name in ("single_line_shifted", "opposite_shifts"):
        n = int((wanted["pattern"] == name).sum())
        print(f"  {name}: {n}")
    print(f"  (for reference, all three moving together: "
          f"{int(((out['pattern'] == 'multi_same_sign') & (out['chi2_diff'] > cut)).sum())}, "
          f"not ranked)")
    cols = ["source_id", "pattern", "chi2_diff", "max_pair_dv", "v_common",
            "t_cat_8498", "t_cat_8542", "t_cat_8662", "bp_rp", "abs_mag_G"]
    with pd.option_context("display.width", 220):
        print("\nTop 20:")
        print(wanted.head(20)[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    src_to_row_all = {int(sid): i for i, sid in enumerate(meta_df["source_id"])}
    cand_rows = [src_to_row_all[int(sid)] for sid in wanted["source_id"]]
    rng = np.random.default_rng(0)
    pool = np.setdiff1d(np.flatnonzero(present), np.array(cand_rows, int))
    ctrl_rows = list(rng.choice(pool, min(200, len(pool)), replace=False))
    if cand_rows:
        xc = confirm_model_independent(λ_grid, Y, cand_rows, ctrl_rows)
        for k, v in xc.items():
            wanted = wanted.copy()
            wanted[k] = v
        wanted.to_csv(out_dir / "caii_candidates.csv", index=False)
        print(f"Wrote {out_dir / 'caii_candidates.csv'}")

    plot_summary(chi2, ctrl, d, out_dir / "caii_summary.pdf", label)

    if args.no_figures:
        print(f"\nDone (no per-star figures). Output in {out_dir}")
        return

    src_to_row = {int(s): i for i, s in enumerate(meta_df["source_id"])}
    spec_dir = out_dir / "spectra"
    # Cleared, so a rerun with a different cut does not leave the previous
    # run's candidates sitting alongside this one's.
    for stale in spec_dir.glob("*/*.png"):
        stale.unlink()
    rows = []
    for name in ("single_line_shifted", "opposite_shifts"):
        sub = wanted[wanted["pattern"] == name].head(args.n_figures)
        folder = spec_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {len(sub)} figures -> {folder}", flush=True)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            i = src_to_row[int(row["source_id"])]
            fn = folder / (f"rank_{rank:03d}_chi2{row['chi2_diff']:07.1f}"
                           f"_srcid_{int(row['source_id'])}.png")
            plot_star_lines(
                λ_grid, Y[i], ivar[i], recon[i], residual[i], row, V[:, i], S[:, i], fn
            )
            rows.append({"group": name, "rank": rank, "file": fn.name, **row.to_dict()})
        print(f"  {len(sub)} figures written", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(spec_dir / "spectra_index.csv", index=False)
        print(f"\nWrote {spec_dir / 'spectra_index.csv'}")
    print(f"Done. {len(rows)} figures in {spec_dir}")


if __name__ == "__main__":
    main()
