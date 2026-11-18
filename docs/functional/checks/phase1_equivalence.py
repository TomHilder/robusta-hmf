"""Phase 1 check: how closely FunctionalRobusta's paths track Robusta, iteration by iteration.

Runs Robusta and FunctionalRobusta (Discrete bases, flat fast path and generic solver) for a
fixed number of iterations (no convergence test), and plots the largest relative difference in
the row and column coefficients (the worse of the two). One panel converges (K equals the true rank); the other is
over-parameterised (K above the true rank) and never converges.

    uv run python docs/functional/checks/phase1_equivalence.py
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
from robusta_hmf import Robusta
from robusta_hmf.functional import Discrete, FunctionalRobusta, Measurements

sys.path.insert(0, str(Path(__file__).parents[3] / "tests"))
from test_functional import problem  # noqa: E402  (the equivalence tests' data)

OUT = Path(__file__).parent / "figs"
N, M = 60, 20
ITERS = [5, 10, 20, 50, 100, 200, 500, 1000]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def rel_diff(state, ref):
    """Worse of the row (A) and column (G) coefficients, relative to the largest entry."""
    return max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))
        for a, b in ((state.row_coeffs, ref.A), (state.col_coeffs, ref.G))
    )


def track(K, Q, Y, W):
    kw = dict(robust_scale=Q, conv_strategy="none")
    rows = np.repeat(np.arange(N), M)
    cols = np.tile(np.arange(M), N)
    flat = Measurements(rows, cols, Y.ravel(), W.ravel())
    out = {"flat": [], "generic": []}
    for it in ITERS:
        ref, _ = Robusta(rank=K, **kw).fit(Y, W, max_iter=it)
        s, _ = FunctionalRobusta(K, Discrete(N), Discrete(M), **kw).fit(flat, max_iter=it)
        out["flat"].append(rel_diff(s, ref))
        s, _ = FunctionalRobusta(K, Discrete(N, False), Discrete(M, False), **kw).fit(
            Measurements.from_matrix(Y, W), max_iter=it
        )
        out["generic"].append(rel_diff(s, ref))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    Y, W = problem(K=3)
    cases = [
        ("K = 3 on rank-3 data (converges)", track(3, 3.0, Y, W)),
        ("K = 5 on rank-3 data (never converges)", track(5, 10.0, Y, W)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, d) in zip(axes, cases):
        ax.plot(ITERS, d["flat"], color=BLUE, lw=2, marker="o", ms=5, label="flat fast path")
        ax.plot(ITERS, d["generic"], color=ORANGE, lw=2, marker="s", ms=5, label="generic solver")
        ax.axhline(1e-10, color=MUTED, lw=1, ls="--")
        ax.text(ITERS[0], 1.4e-10, "test tolerance 1e-10", color=MUTED, fontsize=8)
        ax.set(xscale="log", yscale="log", title=title, xlabel="iterations")
        ax.title.set_fontsize(10)
        ax.grid(color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("max relative coefficient\ndifference vs Robusta")
    axes[0].set_ylim(1e-17, 1e-8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "FunctionalRobusta (Discrete × Discrete) vs Robusta; dense layout is bit-identical (0)",
        fontsize=10, color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "phase1_equivalence.png", dpi=150)


if __name__ == "__main__":
    main()
