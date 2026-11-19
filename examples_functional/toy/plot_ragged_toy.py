"""Figures and summary numbers for the ragged toy (run ragged_toy.py first).

    uv run --group examples python examples_functional/toy/plot_ragged_toy.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_drip import get_color

HERE = Path(__file__).parent
FIGS = HERE / "figs"
FR, RB, TRUTH = get_color(0), get_color(1), "0.15"
FR_LABEL, RB_LABEL = "FunctionalRobusta", "resample + Robusta"
LO, HI = 0.01, 0.99  # comparison range: the baseline excludes its end grid points


def load():
    d = dict(np.load(HERE / "outputs" / "ragged_toy.npz"))
    fine = d["fine"]
    d["keep"] = (fine >= LO) & (fine <= HI)
    # Baseline basis on the fine grid (linear, as it was built).
    d["rb_fine"] = np.stack(
        [np.interp(fine, d["grid"], np.nan_to_num(d["rb_G"][:, k])) for k in range(int(d["K"]))],
        axis=1,
    )
    d["rb_fine"][~d["keep"]] = np.nan
    return d


def mapped(F, T, keep):
    """``F M`` with ``M`` the least-squares map from the fitted to the true functions."""
    M = np.linalg.lstsq(F[keep], T[keep], rcond=None)[0]
    return F @ M


def principal_cosines(F, T):
    q1, q2 = np.linalg.qr(F)[0], np.linalg.qr(T)[0]
    return np.linalg.svd(q1.T @ q2, compute_uv=False)


def object_models(d, i):
    fine = d["fine"]
    return (
        d["truth_fine"] @ d["a_true"][i],
        d["fr_fine"] @ d["fr_A"][i],
        d["rb_fine"] @ np.nan_to_num(d["rb_A"][i]),
    )


def baseline_point_weights(d):
    """Baseline robust weight at each measurement: the lower of its two bracketing grid points."""
    grid, Wg = d["grid"], d["rb_wgrid"]
    j = np.clip(np.searchsorted(grid, d["x"]) - 1, 0, grid.size - 2)
    return np.minimum(Wg[d["row"], j], Wg[d["row"], j + 1])


def object_scores(d, w_fr, w_rb):
    N = d["a_true"].shape[0]
    fr = np.array([np.median(w_fr[d["row"] == i]) for i in range(N)])
    rb = np.array([np.median(w_rb[d["row"] == i]) for i in range(N)])
    return fr, rb


def f1(flag, truth):
    tp = np.sum(flag & truth)
    return 2 * tp / max(np.sum(flag) + np.sum(truth), 1)


# ----------------------------------------------------------------------------- figures


def fig_examples(d):
    """Three objects: measurements, truth and both fits."""
    normal = ~d["outlier_obj"] & ~d["extra_line"]
    full = [i for i in np.flatnonzero(normal) if d["x"][d["row"] == i][0] < 0.02]
    partial = [i for i in np.flatnonzero(normal) if d["x"][d["row"] == i][0] > 0.2]
    picks = [(full[0], "full coverage"), (partial[0], "partial coverage"),
             (np.flatnonzero(d["extra_line"])[0], "extra emission line")]
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    fine = d["fine"]
    for ax, (i, label) in zip(axes, picks):
        m = d["row"] == i
        truth, fr, rb = object_models(d, i)
        ax.errorbar(d["x"][m], d["y"][m], d["sigma"][m], fmt=".", ms=3, color="0.6", lw=0.6,
                    zorder=1, label="data")
        ax.plot(fine, truth, color=TRUTH, lw=1, ls="--", label="truth")
        ax.plot(fine, fr, color=FR, lw=1.4, label=FR_LABEL)
        ax.plot(fine, rb, color=RB, lw=1.4, label=RB_LABEL)
        ax.text(0.01, 0.92, label, transform=ax.transAxes, va="top")
        ax.set_ylabel("flux")
        lo, hi = d["x"][m].min(), d["x"][m].max()
        ax.axvspan(0, lo, color="0.93", zorder=0)
        ax.axvspan(hi, 1, color="0.93", zorder=0)
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(FIGS / "examples.png", dpi=150)
    plt.close(fig)


def fig_functions(d):
    """True column functions vs each fit, mapped onto the truth by least squares."""
    fine, keep, T = d["fine"], d["keep"], d["truth_fine"]
    fr = mapped(d["fr_fine"], T, keep)
    rb = mapped(d["rb_fine"], T, keep)
    names = ["continuum", "curvature", "absorption lines", "emission lines"]
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(fine[keep], T[keep, k], color=TRUTH, lw=1, ls="--", label="truth")
        ax.plot(fine[keep], fr[keep, k], color=FR, lw=1.3, label=FR_LABEL)
        ax.plot(fine[keep], rb[keep, k], color=RB, lw=1.3, label=RB_LABEL)
        ax.text(0.01, 0.9, names[k], transform=ax.transAxes, va="top")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(FIGS / "functions.png", dpi=150)
    plt.close(fig)
    return fr, rb


def fig_function_residuals(d, fr, rb):
    """Mapped fit minus truth for each function: where each method's basis is wrong."""
    fine, keep, T = d["fine"], d["keep"], d["truth_fine"]
    names = ["continuum", "curvature", "absorption lines", "emission lines"]
    fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
    for k, ax in enumerate(axes):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(fine[keep], fr[keep, k] - T[keep, k], color=FR, lw=1, label=FR_LABEL)
        ax.plot(fine[keep], rb[keep, k] - T[keep, k], color=RB, lw=1, label=RB_LABEL)
        ax.text(0.01, 0.9, names[k], transform=ax.transAxes, va="top")
        ax.set_ylabel("fit $-$ truth")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(FIGS / "function_residuals.png", dpi=150)
    plt.close(fig)


def fig_model_error(d):
    """Per-object RMS distance from the true noiseless model, both methods."""
    normal = np.flatnonzero(~d["outlier_obj"] & ~d["extra_line"])
    fine, keep = d["fine"], d["keep"]
    err = []
    for i in normal:
        truth, fr, rb = object_models(d, i)
        x = d["x"][d["row"] == i]
        cov = keep & (fine >= x.min()) & (fine <= x.max())
        err.append([np.sqrt(np.mean((fr - truth)[cov] ** 2)),
                    np.sqrt(np.mean((rb - truth)[cov] ** 2))])
    err = np.array(err)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lim = [err.min() * 0.8, err.max() * 1.2]
    ax.plot(lim, lim, color="0.6", lw=0.8)
    ax.scatter(err[:, 0], err[:, 1], s=8, color=FR, alpha=0.7)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=f"RMS error, {FR_LABEL}", ylabel=f"RMS error, {RB_LABEL}")
    fig.tight_layout()
    fig.savefig(FIGS / "model_error.png", dpi=150)
    plt.close(fig)
    return err


def fig_spikes(d, w_fr, w_rb):
    """Robust weights of injected spikes and of clean measurements, both methods."""
    spike = d["spike"].astype(bool)
    clean = ~spike & ~d["outlier_obj"][d["row"]] & ~d["extra_line"][d["row"]]
    bins = np.logspace(-4, 0, 41)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for ax, mask, label in [(axes[0], clean, "clean measurements"), (axes[1], spike, "spikes")]:
        ax.hist(w_fr[mask], bins, color=FR, histtype="step", lw=1.5, label=FR_LABEL)
        ax.hist(w_rb[mask], bins, color=RB, histtype="step", lw=1.5, label=RB_LABEL)
        ax.axvline(0.5, color="0.6", lw=0.8, ls="--")
        ax.set(xscale="log", yscale="log", ylabel="measurements")
        ax.text(0.02, 0.9, label, transform=ax.transAxes, va="top")
    axes[0].legend(loc="upper center", fontsize=8)
    axes[-1].set_xlabel("robust weight")
    fig.tight_layout()
    fig.savefig(FIGS / "spikes.png", dpi=150)
    plt.close(fig)


def fig_object_scores(d, s_fr, s_rb):
    """Object score (median robust weight) for normal, extra-line and outlier objects."""
    out, line = d["outlier_obj"].astype(bool), d["extra_line"].astype(bool)
    normal = ~out & ~line
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, s, colour, label in [(axes[0], s_fr, FR, FR_LABEL), (axes[1], s_rb, RB, RB_LABEL)]:
        for pos, mask in enumerate([normal, line, out]):
            ax.scatter(pos + 0.15 * rng.uniform(-1, 1, mask.sum()), s[mask], s=8, color=colour,
                       alpha=0.6)
        ax.set_xticks([0, 1, 2], ["normal", "extra line", "outlier"])
        ax.text(0.02, 0.04, label, transform=ax.transAxes)
    axes[0].set_ylabel("median robust weight")
    fig.tight_layout()
    fig.savefig(FIGS / "object_scores.png", dpi=150)
    plt.close(fig)


def main():
    plt.style.use("mpl_drip.custom")
    FIGS.mkdir(exist_ok=True)
    d = load()
    w_fr, w_rb = d["fr_w"], baseline_point_weights(d)
    s_fr, s_rb = object_scores(d, w_fr, w_rb)

    fig_examples(d)
    fr_map, rb_map = fig_functions(d)
    fig_function_residuals(d, fr_map, rb_map)
    err = fig_model_error(d)
    fig_spikes(d, w_fr, w_rb)
    fig_object_scores(d, s_fr, s_rb)

    keep, T = d["keep"], d["truth_fine"]
    print("principal cosines vs truth (1 = recovered):")
    print("  FR:", np.round(principal_cosines(d["fr_fine"][keep], T[keep]), 6))
    print("  RB:", np.round(principal_cosines(d["rb_fine"][keep], T[keep]), 6))
    print("max |mapped fit - truth| per function:")
    print("  FR:", np.round(np.max(np.abs(fr_map - T)[keep], axis=0), 4))
    print("  RB:", np.round(np.max(np.abs(rb_map - T)[keep], axis=0), 4))
    print(f"median per-object RMS error: FR {np.median(err[:, 0]):.4f}, "
          f"RB {np.median(err[:, 1]):.4f}; FR better for {np.mean(err[:, 0] < err[:, 1]):.0%}")
    out, line = d["outlier_obj"].astype(bool), d["extra_line"].astype(bool)
    normal = ~out[d["row"]] & ~line[d["row"]]
    spike = d["spike"].astype(bool) & normal
    clean = ~d["spike"].astype(bool) & normal
    for lab, w in [("FR", w_fr), ("RB", w_rb)]:
        print(f"{lab} (normal objects): spike recall {np.mean(w[spike] < 0.5):.3f}, "
              f"clean w < 0.5: {np.sum(w[clean] < 0.5)}, F1 {f1(w[normal] < 0.5, spike[normal]):.3f}")
    for lab, s in [("FR", s_fr), ("RB", s_rb)]:
        print(f"{lab} median-weight score: normal min {s[~out & ~line].min():.3f}, "
              f"outlier max {s[out].max():.3f}")


if __name__ == "__main__":
    main()
