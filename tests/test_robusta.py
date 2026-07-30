# test_robusta.py
#
# Integration tests for the public Robusta API: fit, infer, synthesize,
# robust_weights, serialization, and error/warning paths.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_low_rank_problem, subspace_alignment
from robusta_hmf import Robusta, load_state_from_npz, save_state_to_npz

jax.config.update("jax_enable_x64", True)

RANK = 3  # true rank of the synthetic problems in conftest


# ----------------------------
# Shared fitted models (fit once, assert many)
# ----------------------------
@pytest.fixture(scope="module")
def gaussian_fit(low_rank_problem):
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-10)
    state, loss_history = model.fit(
        jnp.array(low_rank_problem["Y"]),
        jnp.array(low_rank_problem["W"]),
        max_iter=500,
    )
    return model, state, loss_history


@pytest.fixture(scope="module")
def robust_fit(outlier_problem):
    model = Robusta(rank=RANK, robust=True, robust_scale=3.0, conv_tol=1e-10)
    state, loss_history = model.fit(
        jnp.array(outlier_problem["Y"]),
        jnp.array(outlier_problem["W"]),
        max_iter=500,
    )
    return model, state, loss_history


@pytest.fixture(scope="module")
def gaussian_fit_on_outliers(outlier_problem):
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-10)
    state, loss_history = model.fit(
        jnp.array(outlier_problem["Y"]),
        jnp.array(outlier_problem["W"]),
        max_iter=500,
    )
    return model, state, loss_history


# ----------------------------
# Ground-truth recovery (clean data)
# ----------------------------
def test_recovers_true_subspace(gaussian_fit, low_rank_problem):
    _, state, _ = gaussian_fit
    assert subspace_alignment(state.G, low_rank_problem["G_true"]) > 0.99


def test_zscores_calibrated(gaussian_fit, low_rank_problem):
    model, state, _ = gaussian_fit
    resid = model.residuals(jnp.array(low_rank_problem["Y"]), state=state)
    z = np.asarray(resid) / low_rank_problem["sigma"]
    # Slightly below 1 is expected: K(N+M) parameters absorb some noise
    assert 0.9 < z.std() < 1.05


def test_loss_monotone_gaussian_als(gaussian_fit):
    _, _, loss_history = gaussian_fit
    losses = np.asarray(loss_history)
    assert np.all(np.diff(losses) <= 1e-8 * np.abs(losses[0]))


def test_synthesize_shape_and_indices(gaussian_fit, low_rank_problem):
    model, state, _ = gaussian_fit
    N, M = low_rank_problem["Y"].shape
    full = model.synthesize(state=state)
    assert full.shape == (N, M)
    idx = jnp.array([0, 5, 7])
    sub = model.synthesize(state=state, indices=idx)
    assert sub.shape == (3, M)
    assert jnp.allclose(sub, full[idx])


# ----------------------------
# Robust downweighting of outliers
# ----------------------------
def test_robust_weights_discriminate_outliers(robust_fit, outlier_problem):
    model, state, _ = robust_fit
    w = np.asarray(
        model.robust_weights(
            jnp.array(outlier_problem["Y"]), jnp.array(outlier_problem["W"]), state=state
        )
    )
    mask = outlier_problem["outlier_mask"]
    assert w.shape == outlier_problem["Y"].shape
    assert np.all((w >= 0) & (w <= 1))
    assert np.mean(w[mask]) < 0.1  # injected outliers strongly downweighted
    assert np.median(w[~mask]) > 0.8  # clean pixels barely downweighted


def test_loss_monotone_robust_als(robust_fit):
    _, _, loss_history = robust_fit
    losses = np.asarray(loss_history)
    assert np.all(np.diff(losses) <= 1e-8 * np.abs(losses[0]))


def test_robust_recovers_subspace_despite_outliers(robust_fit, outlier_problem):
    _, state, _ = robust_fit
    assert subspace_alignment(state.G, outlier_problem["G_true"]) > 0.99


def test_robust_beats_gaussian_on_outliers(
    robust_fit, gaussian_fit_on_outliers, outlier_problem
):
    _, robust_state, _ = robust_fit
    _, gaussian_state, _ = gaussian_fit_on_outliers
    G_true = outlier_problem["G_true"]
    robust_align = subspace_alignment(robust_state.G, G_true)
    gaussian_align = subspace_alignment(gaussian_state.G, G_true)
    assert robust_align > gaussian_align


def test_gaussian_weights_are_all_ones(gaussian_fit, low_rank_problem):
    model, state, _ = gaussian_fit
    w = model.robust_weights(
        jnp.array(low_rank_problem["Y"]), jnp.array(low_rank_problem["W"]), state=state
    )
    assert jnp.all(w == 1.0)


# ----------------------------
# infer(): fixed-basis inference
# ----------------------------
def test_infer_reproduces_training_reconstruction(robust_fit, outlier_problem):
    model, state, _ = robust_fit
    Y = jnp.array(outlier_problem["Y"])
    W = jnp.array(outlier_problem["W"])
    infer_state, _ = model.infer(Y, W, state=state, max_iter=100, conv_tol=1e-10)
    assert jnp.allclose(infer_state.G, state.G)  # basis must stay fixed
    recon_fit = model.synthesize(state=state)
    recon_infer = model.synthesize(state=infer_state)
    assert np.max(np.abs(np.asarray(recon_fit - recon_infer))) < 0.05


def test_infer_calibrated_on_new_rows(gaussian_fit, low_rank_problem):
    model, state, _ = gaussian_fit
    rng = np.random.default_rng(99)
    N_new, K = 50, RANK
    A_new = rng.normal(size=(N_new, K))
    sigma = rng.uniform(0.05, 0.15, size=(N_new, low_rank_problem["G_true"].shape[0]))
    Y_new = A_new @ low_rank_problem["G_true"].T + sigma * rng.normal(size=sigma.shape)
    infer_state, _ = model.infer(jnp.array(Y_new), jnp.array(1.0 / sigma**2), state=state)
    z = (Y_new - np.asarray(infer_state.A @ infer_state.G.T)) / sigma
    assert 0.85 < z.std() < 1.1


def test_infer_warns_on_max_frac_G(gaussian_fit, low_rank_problem):
    model, state, _ = gaussian_fit
    with pytest.warns(UserWarning, match="max_frac_G"):
        model.infer(
            jnp.array(low_rank_problem["Y"]),
            jnp.array(low_rank_problem["W"]),
            state=state,
            conv_strategy="max_frac_G",
            max_iter=2,
        )


# ----------------------------
# Determinism
# ----------------------------
def test_fit_deterministic_with_seed(low_rank_problem):
    Y = jnp.array(low_rank_problem["Y"])
    W = jnp.array(low_rank_problem["W"])
    states = []
    for _ in range(2):
        model = Robusta(rank=RANK, init_strategy="random", conv_strategy="none")
        state, _ = model.fit(Y, W, max_iter=30, seed=42)
        states.append(state)
    assert np.array_equal(np.asarray(states[0].A), np.asarray(states[1].A))
    assert np.array_equal(np.asarray(states[0].G), np.asarray(states[1].G))


# ----------------------------
# Missing data and ill-conditioning
# ----------------------------
def test_missing_data_zero_weights(low_rank_problem):
    rng = np.random.default_rng(7)
    W = low_rank_problem["W"].copy()
    missing = rng.random(W.shape) < 0.3
    W[missing] = 0.0
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-10)
    state, _ = model.fit(jnp.array(low_rank_problem["Y"]), jnp.array(W), max_iter=500)
    assert np.all(np.isfinite(np.asarray(state.A)))
    assert np.all(np.isfinite(np.asarray(state.G)))
    resid = np.asarray(model.residuals(jnp.array(low_rank_problem["Y"]), state=state))
    z_obs = (resid / low_rank_problem["sigma"])[~missing]
    assert 0.85 < z_obs.std() < 1.1


def test_fully_masked_column_with_ridge(low_rank_problem):
    W = low_rank_problem["W"].copy()
    W[:, 5] = 0.0  # no information at all about column 5
    model = Robusta(rank=RANK, robust=False, als_ridge=1e-6, conv_tol=1e-10)
    state, _ = model.fit(jnp.array(low_rank_problem["Y"]), jnp.array(W), max_iter=200)
    assert np.all(np.isfinite(np.asarray(state.A)))
    assert np.all(np.isfinite(np.asarray(state.G)))
    # With ridge and zero weight, the dead column's basis row collapses to zero
    assert np.allclose(np.asarray(state.G)[5], 0.0)


# ----------------------------
# Shape extremes
# ----------------------------
@pytest.mark.parametrize(
    "N,M,K",
    [
        (40, 20, 1),  # rank 1
        (20, 40, 2),  # more pixels than objects
        (30, 6, 6),  # K = min(N, M)
    ],
)
def test_fit_shape_extremes(N, M, K):
    prob = make_low_rank_problem(seed=3, N=N, M=M, K=K)
    model = Robusta(rank=K, robust=False, conv_strategy="none")
    state, loss_history = model.fit(jnp.array(prob["Y"]), jnp.array(prob["W"]), max_iter=20)
    assert state.A.shape == (N, K)
    assert state.G.shape == (M, K)
    assert np.all(np.isfinite(np.asarray(loss_history)))


# ----------------------------
# SGD path
# ----------------------------
def test_sgd_fit_reduces_loss(low_rank_problem):
    model = Robusta(rank=RANK, robust=False, method="sgd", conv_strategy="none")
    state, loss_history = model.fit(
        jnp.array(low_rank_problem["Y"]),
        jnp.array(low_rank_problem["W"]),
        max_iter=200,
        rotation_cadence=50,
    )
    losses = np.asarray(loss_history)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert np.all(np.isfinite(np.asarray(state.A)))
    assert np.all(np.isfinite(np.asarray(state.G)))


# ----------------------------
# State handling and serialization
# ----------------------------
def test_set_state_returns_independent_copy(gaussian_fit):
    model, state, _ = gaussian_fit
    other = Robusta(rank=RANK)
    other2 = other.set_state(state)
    assert other._state is None  # original untouched
    assert other2 is not other
    assert jnp.allclose(other2.state.A, state.A)


def test_state_npz_roundtrip_through_robusta(gaussian_fit, tmp_path):
    model, state, _ = gaussian_fit
    path = tmp_path / "state.npz"
    save_state_to_npz(state, path)
    loaded = load_state_from_npz(path)
    model2 = Robusta(rank=RANK).set_state(loaded)
    assert np.array_equal(np.asarray(model2.synthesize()), np.asarray(model.synthesize()))


# ----------------------------
# Warnings and error paths
# ----------------------------
def test_als_rotation_cadence_warns(low_rank_problem):
    model = Robusta(rank=RANK, conv_strategy="none")
    with pytest.warns(UserWarning, match="rotation_cadence"):
        model.fit(
            jnp.array(low_rank_problem["Y"]),
            jnp.array(low_rank_problem["W"]),
            max_iter=3,
            rotation_cadence=2,
        )


@pytest.mark.parametrize("method_name", ["synthesize", "basis_vectors", "coefficients"])
def test_methods_raise_before_fit(method_name):
    model = Robusta(rank=RANK)
    with pytest.raises(ValueError, match="fit"):
        getattr(model, method_name)()


def test_robust_weights_raises_before_fit(low_rank_problem):
    model = Robusta(rank=RANK)
    with pytest.raises(ValueError, match="fit"):
        model.robust_weights(
            jnp.array(low_rank_problem["Y"]), jnp.array(low_rank_problem["W"])
        )


def test_properties_none_before_fit():
    model = Robusta(rank=RANK)
    assert model.A is None
    assert model.G is None
    assert model.state is None
    assert model.loss_history is None
