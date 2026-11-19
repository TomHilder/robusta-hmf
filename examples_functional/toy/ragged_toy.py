"""Ragged toy: FunctionalRobusta on per-object sampling vs resample-then-Robusta.

Generates N objects from K smooth functions on [0, 1] (continuum, curvature, absorption- and
emission-line templates). Each object is sampled on its own regular grid (own spacing, random
offset, full or partial coverage) with heteroskedastic noise, plus spike outliers, outlier
objects and objects with an extra emission line. Fits

- FunctionalRobusta: Discrete rows, Fourier columns, directly on the measurements;
- the baseline: each object linearly interpolated onto a common grid, then Robusta,

with the same rank and robust scale, and saves data, truth and both fits to
outputs/ragged_toy_nf<n_freq>.npz.

    uv run python examples_functional/toy/ragged_toy.py [--n-freq 60]
"""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from robusta_hmf import Robusta
from robusta_hmf.functional import Discrete, Fourier, FunctionalRobusta, Measurements

OUT = Path(__file__).parent / "outputs"
SEED = 0
N = 400
K = 4
Q = 4.0
PAD = 0.25
GRID = np.linspace(0.0, 1.0, 101)  # the baseline's common grid
FINE = np.linspace(0.0, 1.0, 2001)  # for comparing functions


def gaussians(x, centres, width):
    return sum(np.exp(-0.5 * ((x - c) / width) ** 2) for c in centres)


def true_functions(x):
    """The K = 4 true column functions, shape (len(x), 4)."""
    return np.stack(
        [
            1.0 + 0.3 * x,
            np.sin(np.pi * x),
            -gaussians(x, [0.2, 0.45, 0.7], 0.02),
            gaussians(x, [0.33, 0.85], 0.015),
        ],
        axis=1,
    )


def make_data(rng):
    a = np.stack(
        [
            rng.normal(1.0, 0.1, N),
            rng.normal(0.0, 0.2, N),
            rng.lognormal(np.log(0.5), 0.3, N),
            np.abs(rng.normal(0.0, 0.3, N)),
        ],
        axis=1,
    )
    outlier_obj = np.zeros(N, bool)
    outlier_obj[rng.choice(N, 8, replace=False)] = True
    extra_line = np.zeros(N, bool)
    extra_line[rng.choice(np.flatnonzero(~outlier_obj), 8, replace=False)] = True
    noise_scale = 0.02 * rng.lognormal(0.0, 0.5, N)

    rows, xs, ys, sig, clean, spike = [], [], [], [], [], []
    line_centre = np.full(N, np.nan)
    for i in range(N):
        dx = rng.uniform(0.008, 0.015)
        if rng.random() < 0.3:
            length = rng.uniform(0.5, 0.9)
            lo = rng.uniform(0.0, 1.0 - length)
            hi = lo + length
        else:
            lo, hi = 0.0, 1.0
        x = np.arange(lo + rng.uniform(0, dx), hi, dx)
        if outlier_obj[i]:
            f = 1.0 + rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * rng.uniform(10, 30) * x)
        else:
            f = true_functions(x) @ a[i]
        if extra_line[i]:
            line_centre[i] = rng.uniform(lo + 0.05, hi - 0.05)
            f = f + 0.3 * gaussians(x, [line_centre[i]], 0.01)
        s = noise_scale[i] * (1 + 0.5 * x)
        y = f + s * rng.normal(size=x.size)
        is_spike = rng.random(x.size) < 0.01
        y = y + is_spike * rng.choice([-1, 1], x.size) * rng.uniform(0.3, 0.6, x.size)
        rows.append(np.full(x.size, i))
        xs.append(x)
        ys.append(y)
        sig.append(s)
        clean.append(f)
        spike.append(is_spike)
    cat = np.concatenate
    return dict(
        row=cat(rows), x=cat(xs), y=cat(ys), sigma=cat(sig), clean=cat(clean),
        spike=cat(spike), a_true=a, outlier_obj=outlier_obj, extra_line=extra_line,
        line_centre=line_centre,
    )


def resample_to_grid(d):
    """Linear interpolation of each object onto GRID; zero weight outside its coverage."""
    Y = np.zeros((N, GRID.size))
    W = np.zeros((N, GRID.size))
    for i in range(N):
        m = d["row"] == i
        x, y, v = d["x"][m], d["y"][m], d["sigma"][m] ** 2
        inside = (GRID >= x[0]) & (GRID <= x[-1])
        g = GRID[inside]
        j = np.clip(np.searchsorted(x, g) - 1, 0, x.size - 2)
        t = (g - x[j]) / (x[j + 1] - x[j])
        Y[i, inside] = (1 - t) * y[j] + t * y[j + 1]
        W[i, inside] = 1.0 / ((1 - t) ** 2 * v[j] + t**2 * v[j + 1])
    return Y, W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-freq", type=int, default=60, help="Fourier harmonics (2n+1 modes)")
    N_FREQ = parser.parse_args().n_freq
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = make_data(rng)
    print(f"{N} objects, {d['x'].size} measurements, {d['spike'].sum()} spikes")

    basis = Fourier(n_freq=N_FREQ, domain=(0.0, 1.0), pad=PAD)
    data = Measurements(d["row"], d["x"], d["y"], d["sigma"] ** -2)
    fr = FunctionalRobusta(K, Discrete(N), basis, robust_scale=Q, conv_tol=1e-6)
    fr_state, fr_loss = fr.fit(data, max_iter=2000)
    fr_w = np.asarray(fr.robust_weights(data))
    fr_fine = np.asarray(basis.features(FINE) @ fr_state.col_coeffs)  # (fine, K)

    Y, W = resample_to_grid(d)
    rb = Robusta(rank=K, robust_scale=Q, conv_tol=1e-6)
    rb_state, rb_loss = rb.fit(Y, W, max_iter=2000)
    rb_wgrid = np.asarray(rb.robust_weights(Y, W))

    np.savez(
        OUT / f"ragged_toy_nf{N_FREQ}.npz",
        **d,
        fine=FINE, grid=GRID, truth_fine=true_functions(FINE),
        fr_A=np.asarray(fr_state.row_coeffs), fr_gamma=np.asarray(fr_state.col_coeffs),
        fr_fine=fr_fine, fr_w=fr_w, fr_loss=np.asarray(fr_loss),
        fr_converged=fr.fit_info.converged,
        rb_A=np.asarray(rb_state.A), rb_G=np.asarray(rb_state.G), rb_wgrid=rb_wgrid,
        rb_Y=Y, rb_W=W, rb_loss=np.asarray(rb_loss), rb_converged=rb.fit_info.converged,
        n_freq=N_FREQ, pad=PAD, K=K, Q=Q,
    )
    print(f"FunctionalRobusta: {len(fr_loss)} iterations, converged={fr.fit_info.converged}")
    print(f"Robusta baseline:  {len(rb_loss)} iterations, converged={rb.fit_info.converged}")


if __name__ == "__main__":
    main()
