# test_functional.py
#
# FunctionalRobusta with Discrete bases: data handling, the generic solver,
# and equivalence with Robusta through every execution path.

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_outlier_problem
from robusta_hmf import DataWarning, PrecisionError, Robusta
from robusta_hmf.functional import (
    Basis,
    Discrete,
    FunctionalRobusta,
    FunctionalState,
    Measurements,
)
from robusta_hmf.functional.engine import DenseLayout, FlatLayout, make_layout
from robusta_hmf.functional.solvers import generic_solve

jax.config.update("jax_enable_x64", True)

TOL = 1e-10  # float64 round-off, relative to the largest entry
N, M = 60, 20


def rel_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def problem(seed=0, K=3, missing_frac=0.1):
    """Low-rank data with outliers and missing entries (Y = W = 0 there)."""
    prob = make_outlier_problem(seed=seed, N=N, M=M, K=K, outlier_frac=0.02)
    rng = np.random.default_rng(seed + 7)
    missing = rng.random((N, M)) < missing_frac
    Y = np.where(missing, 0.0, prob["Y"])
    W = np.where(missing, 0.0, prob["W"])
    return Y, W


def flat(Y, W, keep=None):
    """The entries of (Y, W) as a flat list, without the matrix shape."""
    rows, cols = np.nonzero(np.ones_like(Y, dtype=bool) if keep is None else keep)
    return Measurements(rows, cols, Y[rows, cols], W[rows, cols])


# ----------------------------
# Measurements
# ----------------------------
def test_from_matrix_round_trip():
    Y, W = problem()
    data = Measurements.from_matrix(Y, W)
    assert data.shape == (N, M) and data.n == N * M
    Y2, W2 = data.as_matrices()
    np.testing.assert_array_equal(Y2, Y)
    np.testing.assert_array_equal(W2, W)
    np.testing.assert_array_equal(Y[data.row, data.col], data.y)


def test_measurements_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="entries"):
        Measurements(np.arange(3), np.arange(4), np.zeros(3), np.ones(3))


def test_flat_measurements_are_not_a_matrix():
    Y, W = problem()
    with pytest.raises(ValueError, match="not a rectangular"):
        flat(Y, W).as_matrices()


def test_nonfinite_measurements_warn_and_become_missing():
    data = Measurements(np.arange(3), np.arange(3), np.array([1.0, np.nan, 2.0]), np.ones(3))
    model = FunctionalRobusta(1, Discrete(3), Discrete(3))
    with pytest.warns(DataWarning, match="1 measurements"):
        clean = model._prepare(data)
    np.testing.assert_array_equal(clean.y, [1.0, 0.0, 2.0])
    np.testing.assert_array_equal(clean.w, [1.0, 0.0, 1.0])


def test_nonfinite_measurements_raise_with_nan_error():
    data = Measurements(np.arange(2), np.arange(2), np.array([1.0, np.inf]), np.ones(2))
    model = FunctionalRobusta(1, Discrete(2), Discrete(2), nan_error=True)
    with pytest.raises(ValueError, match="non-finite"):
        model._prepare(data)


def test_negative_weights_raise():
    data = Measurements(np.arange(2), np.arange(2), np.ones(2), np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="negative"):
        FunctionalRobusta(1, Discrete(2), Discrete(2))._prepare(data)


# ----------------------------
# Discrete basis
# ----------------------------
def test_discrete_adjoint_is_transpose_of_evaluate():
    rng = np.random.default_rng(0)
    basis = Discrete(7)
    points = jnp.asarray(rng.integers(0, 7, size=40))
    c = jnp.asarray(rng.normal(size=(7, 3)))
    v = jnp.asarray(rng.normal(size=(40, 3)))
    lhs = jnp.sum(basis.evaluate(c, points) * v)
    rhs = jnp.sum(c * basis.adjoint(v, points))
    assert abs(float(lhs - rhs)) < 1e-12


@pytest.mark.parametrize(
    "points, match",
    [(np.array([0, 5]), r"\[0, 5\)"), (np.array([-1, 0]), r"\[0, 5\)"), (np.array([0.5]), "integer")],
)
def test_discrete_rejects_bad_ids(points, match):
    with pytest.raises(ValueError, match=match):
        Discrete(5).check_points(points)


def test_discrete_fast_solve_matches_generic():
    rng = np.random.default_rng(1)
    basis = Discrete(6)
    points = jnp.asarray(rng.integers(0, 6, size=50))
    other = jnp.asarray(rng.normal(size=(50, 2)))
    y = jnp.asarray(rng.normal(size=50))
    w = jnp.asarray(rng.uniform(0.5, 2.0, size=50))
    fast = basis.solve(points, other, y, w, other_n=50)
    slow = Discrete(6, fast=False).solve(points, other, y, w, other_n=50)
    assert rel_diff(fast, slow) < TOL


# ----------------------------
# Generic solver on singular systems
# ----------------------------
class DuplicatedDiscrete(Basis):
    """A Discrete basis whose last function repeats the first: singular by construction."""

    n: int

    @property
    def n_coeffs(self):
        return self.n + 1

    def evaluate(self, coeffs, points):
        return coeffs[points] + jnp.where((points == 0)[:, None], coeffs[self.n], 0.0)

    def adjoint(self, values, points):
        out = jax.ops.segment_sum(values, points, num_segments=self.n)
        extra = jnp.sum(jnp.where((points == 0)[:, None], values, 0.0), axis=0)
        return jnp.concatenate([out, extra[None]], axis=0)

    def gram(self):
        return None


def test_generic_solve_is_minimum_norm_on_a_duplicated_function():
    rng = np.random.default_rng(2)
    basis = DuplicatedDiscrete(4)
    points = jnp.asarray(rng.integers(0, 4, size=30))
    other = jnp.asarray(rng.normal(size=(30, 2)))
    y = jnp.asarray(rng.normal(size=30))
    w = jnp.asarray(rng.uniform(0.5, 2.0, size=30))
    coeffs = basis.solve(points, other, y, w, other_n=30)
    # The duplicated pair shares its coefficient equally (minimum norm) ...
    np.testing.assert_allclose(coeffs[0], coeffs[4], rtol=1e-10)
    # ... and the fit equals the full-rank fit without the duplicate.
    ref = Discrete(4).solve(points, other, y, w, other_n=30)
    np.testing.assert_allclose(coeffs[0] + coeffs[4], ref[0], rtol=1e-10)
    np.testing.assert_allclose(coeffs[1:4], ref[1:4], rtol=1e-10)
    # It is the minimum-norm least-squares solution.
    X = np.asarray(
        (basis.design(points)[:, :, None] * other[:, None, :]).reshape(30, -1)
    ) * np.sqrt(np.asarray(w))[:, None]
    lstsq = np.linalg.lstsq(X, np.asarray(y) * np.sqrt(np.asarray(w)), rcond=None)[0]
    np.testing.assert_allclose(np.asarray(coeffs).ravel(), lstsq, rtol=1e-8, atol=1e-12)


def test_generic_solve_gives_zero_for_an_untouched_function():
    rng = np.random.default_rng(3)
    points = jnp.asarray(rng.integers(0, 4, size=30))  # category 4 never appears
    other = jnp.asarray(rng.normal(size=(30, 2)))
    y = jnp.asarray(rng.normal(size=30))
    w = jnp.ones(30)
    coeffs = generic_solve(Discrete(5).design(points), other, y, w)
    assert np.all(np.isfinite(coeffs))
    np.testing.assert_array_equal(coeffs[4], 0.0)


# ----------------------------
# Equivalence with Robusta
# ----------------------------
PATHS = ["dense", "flat", "generic", "ragged"]


def functional_fit(path, Y, W, K, **kw):
    if path == "dense":
        data, rb, cb = Measurements.from_matrix(Y, W), Discrete(N), Discrete(M)
    elif path == "flat":
        data, rb, cb = flat(Y, W), Discrete(N), Discrete(M)
    elif path == "generic":
        data, rb, cb = Measurements.from_matrix(Y, W), Discrete(N, False), Discrete(M, False)
    elif path == "ragged":  # missing entries dropped from the list, not zero-weighted
        data, rb, cb = flat(Y, W, keep=W > 0), Discrete(N), Discrete(M)
    else:
        raise ValueError(path)
    model = FunctionalRobusta(K, rb, cb, **kw)
    state, loss = model.fit(data, max_iter=300)
    return model, data, state, loss


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "K, true_K, kw",
    [
        (3, 3, dict(robust_scale=3.0)),
        (2, 3, dict(robust_scale=1.0)),
        # Converging cases only: an over-parameterised fit (K above the true
        # rank) drifts along directions the data barely constrain and does not
        # converge, so round-off differences between paths grow with iterations.
        (5, 5, dict(robust_scale=10.0)),
        (3, 3, dict(robust=False)),
        (3, 3, dict(robust_scale=2.0, robust_nu=4.0, conv_strategy="rel_frac_loss")),
    ],
)
def test_matches_robusta(path, K, true_K, kw):
    Y, W = problem(K=true_K)
    kw = dict(conv_tol=1e-8, **kw)
    ref = Robusta(rank=K, **kw)
    ref_state, ref_loss = ref.fit(Y, W, max_iter=300)
    model, data, state, loss = functional_fit(path, Y, W, K, **kw)

    layout = make_layout(data, model.row_basis, model.col_basis)
    assert isinstance(layout, DenseLayout if path == "dense" else FlatLayout)
    assert len(loss) == len(ref_loss)
    assert model.fit_info.converged and ref.fit_info.converged
    assert rel_diff(state.row_coeffs, ref_state.A) < TOL
    assert rel_diff(state.col_coeffs, ref_state.G) < TOL
    assert rel_diff(loss, ref_loss) < TOL

    ref_w = np.asarray(ref.robust_weights(Y, W))[np.asarray(data.row), np.asarray(data.col)]
    assert rel_diff(model.robust_weights(data), ref_w) < TOL
    ref_model = np.asarray(ref.synthesize())[np.asarray(data.row), np.asarray(data.col)]
    assert rel_diff(model.predict(data.row, data.col), ref_model) < TOL


def test_matches_robusta_with_random_init():
    Y, W = problem()
    ref = Robusta(rank=3, robust_scale=3.0, conv_tol=1e-8, init_strategy="random")
    ref_state, ref_loss = ref.fit(Y, W, max_iter=300, seed=5)
    model = FunctionalRobusta(
        3, Discrete(N), Discrete(M), robust_scale=3.0, conv_tol=1e-8, init_strategy="random"
    )
    state, loss = model.fit(Measurements.from_matrix(Y, W), max_iter=300, seed=5)
    assert rel_diff(state.col_coeffs, ref_state.G) < TOL
    assert rel_diff(loss, ref_loss) < TOL


def test_matches_robusta_with_nan_entries():
    Y, W = problem()
    Y[3, 4] = np.nan
    W[7, 1] = np.inf
    with pytest.warns(DataWarning):
        ref_state, ref_loss = Robusta(rank=3, robust_scale=3.0).fit(Y, W, max_iter=100)
    with pytest.warns(DataWarning):
        state, loss = FunctionalRobusta(3, Discrete(N), Discrete(M), robust_scale=3.0).fit(
            Measurements.from_matrix(Y, W), max_iter=100
        )
    assert rel_diff(state.col_coeffs, ref_state.G) < TOL
    assert rel_diff(loss, ref_loss) < TOL


def test_continues_from_init_state():
    Y, W = problem()
    data = Measurements.from_matrix(Y, W)
    model = FunctionalRobusta(3, Discrete(N), Discrete(M), robust_scale=3.0, conv_tol=1e-8)
    state, _ = model.fit(data, max_iter=20, conv_check_cadence=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again, loss = model.fit(data, max_iter=300, init_state=state)
    assert int(again.it) > int(state.it)
    full, _ = FunctionalRobusta(
        3, Discrete(N), Discrete(M), robust_scale=3.0, conv_tol=1e-8
    ).fit(data, max_iter=300)
    np.testing.assert_allclose(
        model.predict(data.row, data.col, again), model.predict(data.row, data.col, full),
        atol=1e-6,
    )


# ----------------------------
# Errors
# ----------------------------
def test_rank_above_basis_size_raises():
    Y, W = problem()
    with pytest.raises(ValueError, match="rank"):
        FunctionalRobusta(25, Discrete(N), Discrete(M)).fit(Measurements.from_matrix(Y, W))


def test_init_state_with_wrong_shape_raises():
    Y, W = problem()
    bad = FunctionalState(np.zeros((N, 2)), np.zeros((M, 3)))
    with pytest.raises(ValueError, match="row_coeffs"):
        FunctionalRobusta(3, Discrete(N), Discrete(M)).fit(
            Measurements.from_matrix(Y, W), init_state=bad
        )


def test_svd_init_needs_discrete_bases():
    Y, W = problem()
    model = FunctionalRobusta(3, DuplicatedDiscrete(N - 1), Discrete(M))
    data = Measurements.from_matrix(Y[: N - 1], W[: N - 1])
    with pytest.raises(ValueError, match="svd"):
        model.fit(data)


def test_data_must_be_measurements():
    Y, W = problem()
    with pytest.raises(TypeError, match="Measurements"):
        FunctionalRobusta(3, Discrete(N), Discrete(M)).fit(Y)


def test_float64_without_x64_raises():
    Y, W = problem()
    data = Measurements.from_matrix(Y, W)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(PrecisionError):
            FunctionalRobusta(3, Discrete(N), Discrete(M)).fit(data)
    finally:
        jax.config.update("jax_enable_x64", True)
