# test_functional_fourier.py
#
# The Fourier basis, and FunctionalRobusta with a Discrete row basis and a
# Fourier column basis on ragged data.

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from robusta_hmf import RobustaWarning
from robusta_hmf.functional import Discrete, Fourier, FunctionalRobusta, Measurements

jax.config.update("jax_enable_x64", True)

DOMAIN = (0.75, 5.0)


# ----------------------------
# The basis
# ----------------------------
def test_fourier_layout_and_size():
    basis = Fourier(n_freq=3, domain=DOMAIN, pad=0.5)
    assert basis.n_coeffs == 7
    assert basis.period == pytest.approx(4.25 * 1.5)
    x = jnp.array([0.75, 2.0, 5.0])
    Phi = basis.features(x)
    theta = 2 * np.pi * (np.asarray(x) - 0.75) / basis.period
    np.testing.assert_allclose(Phi[:, 0], 1.0)
    np.testing.assert_allclose(Phi[:, 2], np.cos(2 * theta), atol=1e-12)
    np.testing.assert_allclose(Phi[:, 3 + 3], np.sin(3 * theta), atol=1e-12)


def test_fourier_adjoint_is_transpose_of_evaluate():
    rng = np.random.default_rng(0)
    op = Fourier(n_freq=5, domain=DOMAIN).bind(jnp.asarray(rng.uniform(*DOMAIN, size=60)))
    c = jnp.asarray(rng.normal(size=(11, 3)))
    v = jnp.asarray(rng.normal(size=(60, 3)))
    assert abs(float(jnp.sum(op.evaluate(c) * v) - jnp.sum(c * op.adjoint(v)))) < 1e-10


@pytest.mark.parametrize("pad", [0.0, 0.25, 1.0])
def test_fourier_gram_matches_quadrature(pad):
    basis = Fourier(n_freq=6, domain=DOMAIN, pad=pad)
    x = np.linspace(*DOMAIN, 200_001)
    Phi = np.asarray(basis.features(jnp.asarray(x)))
    S_quad = np.trapezoid(Phi[:, :, None] * Phi[:, None, :], x, axis=0)
    np.testing.assert_allclose(basis.gram(), S_quad, atol=1e-8)


def test_fourier_gram_without_padding_is_diagonal():
    basis = Fourier(n_freq=4, domain=DOMAIN, pad=0.0)
    L = basis.period
    np.testing.assert_allclose(basis.gram(), np.diag([L] + [L / 2] * 8), atol=1e-12)


def test_fourier_rejects_points_outside_the_domain():
    with pytest.raises(ValueError, match="domain"):
        Fourier(n_freq=2, domain=DOMAIN).check_points(np.array([1.0, 5.5]))


@pytest.mark.parametrize("kw", [dict(domain=(1.0, 1.0)), dict(n_freq=-1), dict(pad=-0.1)])
def test_fourier_rejects_bad_arguments(kw):
    args = dict(n_freq=2, domain=DOMAIN) | kw
    with pytest.raises(ValueError):
        Fourier(**args)


# ----------------------------
# Fitting ragged data
# ----------------------------
def ragged_problem(seed=0, N=80, K=2, n_freq=4, sigma=0.02):
    """Rows observed at their own random coordinates, from functions in the basis itself."""
    rng = np.random.default_rng(seed)
    basis = Fourier(n_freq=n_freq, domain=DOMAIN)
    G_true = rng.normal(size=(basis.n_coeffs, K)) / np.arange(1, basis.n_coeffs + 1)[:, None]
    A_true = rng.normal(size=(N, K))
    rows, cols = [], []
    for i in range(N):
        n_i = rng.integers(20, 40)
        rows.append(np.full(n_i, i))
        cols.append(np.sort(rng.uniform(*DOMAIN, size=n_i)))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    Phi = np.asarray(basis.features(jnp.asarray(cols)))
    clean = np.sum(A_true[rows] * (Phi @ G_true), axis=1)
    y = clean + sigma * rng.normal(size=clean.size)
    data = Measurements(rows, cols, y, np.full(y.size, sigma**-2))
    return basis, data, A_true, G_true


def test_recovers_functions_from_ragged_data():
    basis, data, A_true, G_true = ragged_problem()
    model = FunctionalRobusta(2, Discrete(80), basis, robust_scale=5.0, conv_tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RobustaWarning)
        state, loss = model.fit(data, max_iter=500)
    assert model.fit_info.converged
    # The model on a dense grid matches the truth for every row, to the noise level.
    x = np.linspace(*DOMAIN, 400)
    Phi = np.asarray(basis.features(jnp.asarray(x)))
    truth = A_true @ (Phi @ G_true).T
    fit = np.asarray(state.row_coeffs) @ (Phi @ np.asarray(state.col_coeffs)).T
    assert np.max(np.abs(fit - truth)) < 0.05
    assert np.sqrt(np.mean((fit - truth) ** 2)) < 0.01


def test_column_functions_are_orthonormal_over_the_domain():
    basis, data, _, _ = ragged_problem(seed=1)
    state, _ = FunctionalRobusta(2, Discrete(80), basis, conv_tol=1e-10).fit(data, max_iter=200)
    gamma = np.asarray(state.col_coeffs)
    np.testing.assert_allclose(gamma.T @ np.asarray(basis.gram()) @ gamma, np.eye(2), atol=1e-6)


def test_predict_matches_the_fitted_model_at_the_measurements():
    basis, data, _, _ = ragged_problem(seed=2)
    model = FunctionalRobusta(2, Discrete(80), basis)
    state, _ = model.fit(data, max_iter=200)
    expected = np.sum(
        np.asarray(state.row_coeffs)[np.asarray(data.row)]
        * np.asarray(basis.features(data.col) @ state.col_coeffs),
        axis=1,
    )
    np.testing.assert_allclose(model.predict(data.row, data.col), expected, rtol=1e-12)


def test_flags_injected_outliers():
    basis, data, _, _ = ragged_problem(seed=3)
    rng = np.random.default_rng(30)
    bad = rng.random(data.n) < 0.02
    y = np.where(bad, np.asarray(data.y) + rng.choice([-1, 1], size=data.n), np.asarray(data.y))
    data = Measurements(data.row, data.col, y, data.w)
    model = FunctionalRobusta(2, Discrete(80), basis, robust_scale=3.0, conv_tol=1e-8)
    model.fit(data, max_iter=500)
    w = np.asarray(model.robust_weights(data))
    assert np.all(w[bad] < 0.1)
    assert np.median(w[~bad]) > 0.9
