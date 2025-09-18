# test_likelihoods.py
import jax
import jax.numpy as jnp
import pytest
from robusta_hmf.likelihoods import (
    CauchyLikelihood,
    GaussianLikelihood,
    StudentTLikelihood,
)


def rng(seed=0, N=6, M=5, K=3):
    key = jax.random.key(seed)
    Y = jax.random.normal(key, (N, M))
    A = jax.random.normal(key, (N, K))
    G = jax.random.normal(key, (M, K))
    W = jax.random.uniform(key, (N, M), minval=0.5, maxval=2.0)
    return Y, A, G, W


def resid(Y, A, G):
    return Y - A @ G.T


def wls_surrogate_loss(Wtot, Y, A, G):
    r2 = (Y - A @ G.T) ** 2
    return jnp.sum(jax.lax.stop_gradient(Wtot) * r2)


# ----------------------------
# API & identities
# ----------------------------
@pytest.mark.parametrize(
    "L", [GaussianLikelihood(), CauchyLikelihood(scale=1.3), StudentTLikelihood()]
)
def test_shapes_and_total_equals_data_times_irls(L):
    Y, A, G, W = rng()
    W_irls = L.weights_irls(Y, A, G, W)
    W_tot = (
        W * W_irls if not hasattr(L, "weights_total") else L.weights_total(Y, A, G, W)
    )
    assert W_irls.shape == Y.shape
    assert jnp.allclose(W_tot, W * W_irls)


def test_gaussian_matches_chi2():
    L = GaussianLikelihood()
    Y, A, G, W = rng()
    assert jnp.allclose(L.weights_irls(Y, A, G, W), jnp.ones_like(Y))
    assert jnp.allclose(L.weights_total(Y, A, G, W), W)
    r2 = resid(Y, A, G) ** 2
    assert jnp.allclose(L.loss(Y, A, G, W), jnp.sum(W * r2))


# ----------------------------
# Robust behaviour
# ----------------------------
def test_cauchy_downweights_outliers():
    L = CauchyLikelihood(scale=1.0)
    Y, A, G, W = rng()
    W0 = L.weights_total(Y, A, G, W)
    Y_big = Y.at[0, 0].set(Y[0, 0] + 50.0)
    W1 = L.weights_total(Y_big, A, G, W)
    assert W1[0, 0] < W0[0, 0]


def test_scale_effect_cauchy():
    Y, A, G, W = rng()
    L_small = CauchyLikelihood(scale=0.5)
    L_large = CauchyLikelihood(scale=5.0)
    W_small = L_small.weights_irls(Y, A, G, W)
    W_large = L_large.weights_irls(Y, A, G, W)
    # Larger scale -> less downweighting -> larger IRLS factors
    assert jnp.all(W_large >= W_small - 1e-12)
    # With the rescaled loss, larger scale also increases the loss
    assert L_large.loss(Y, A, G, W) >= L_small.loss(Y, A, G, W) - 1e-12


# ----------------------------
# Student-t special cases
# ----------------------------
def test_student_t_matches_cauchy_at_nu1():
    Y, A, G, W = rng()
    s = 1.7
    C = CauchyLikelihood(scale=s)
    T = StudentTLikelihood(nu=1.0, scale=s)
    # exact equality for both weights_total and loss
    assert jnp.allclose(
        C.weights_total(Y, A, G, W),
        W * T.weights_irls(Y, A, G, W),
        rtol=1e-7,
        atol=1e-8,
    )
    assert jnp.allclose(C.loss(Y, A, G, W), T.loss(Y, A, G, W), rtol=1e-7, atol=1e-8)


def test_student_t_gaussian_limit_large_nu():
    Y, A, G, W = rng()
    T = StudentTLikelihood(nu=1e9, scale=1.0)
    Gaus = GaussianLikelihood()
    Wt = W * T.weights_irls(Y, A, G, W)
    assert jnp.allclose(Wt, W, rtol=1e-6, atol=1e-6)
    gA_t, gG_t = jax.grad(lambda A, G: T.loss(Y, A, G, W), (0, 1))(A, G)
    gA_g, gG_g = jax.grad(lambda A, G: Gaus.loss(Y, A, G, W), (0, 1))(A, G)
    assert jnp.allclose(gA_t, gA_g, rtol=1e-6, atol=1e-5)
    assert jnp.allclose(gG_t, gG_g, rtol=1e-6, atol=1e-5)


# ----------------------------
# IRLS tangent equivalence
# ----------------------------
@pytest.mark.parametrize(
    "L",
    [
        GaussianLikelihood(),
        CauchyLikelihood(scale=1.2),
        StudentTLikelihood(nu=3.5, scale=0.9),
    ],
)
def test_tangent_match_gradients_exact(L):
    Y, A, G, W = rng()
    Wtot = W * L.weights_irls(Y, A, G, W)

    def loss_true(A, G):
        return L.loss(Y, A, G, W)

    def loss_wls(A, G):
        return wls_surrogate_loss(Wtot, Y, A, G)

    gA_true, gG_true = jax.grad(loss_true, (0, 1))(A, G)
    gA_wls, gG_wls = jax.grad(loss_wls, (0, 1))(A, G)

    # with the new prefactors these should now be equal (not just proportional)
    assert jnp.allclose(gA_true, gA_wls, rtol=1e-6, atol=1e-7)
    assert jnp.allclose(gG_true, gG_wls, rtol=1e-6, atol=1e-7)


# ----------------------------
# Numerical hygiene
# ----------------------------
@pytest.mark.parametrize(
    "L", [CauchyLikelihood(scale=1.0), StudentTLikelihood(nu=2.5, scale=0.8)]
)
def test_no_nans_and_bounds(L):
    Y, A, G, W = rng()
    Y = Y.at[0, 0].set(Y[0, 0] + 1e6)  # huge residual
    w_irls = L.weights_irls(Y, A, G, W)
    w_total = W * w_irls
    loss = L.loss(Y, A, G, W)
    assert jnp.all(jnp.isfinite(w_irls))
    assert jnp.all(jnp.isfinite(w_total))
    assert jnp.isfinite(loss)
    # irls factors in [0,1]
    assert jnp.all(w_irls >= -1e-12)
    assert jnp.all(w_irls <= 1.0 + 1e-12)
