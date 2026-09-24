# test_metrics.py
#
# The metrics module replaces score computations that were previously
# reimplemented inline in the paper's analysis scripts, so these tests pin the
# definitions against those originals as well as against the closed forms.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_low_rank_problem, make_outlier_problem

from robusta_hmf import Robusta
from robusta_hmf.likelihoods import GaussianLikelihood, StudentTLikelihood
from robusta_hmf.metrics import (
    ZMoments,
    kl_from_moments,
    mad_z,
    moments_from_z,
    z_moments,
    z_scores,
)

jax.config.update("jax_enable_x64", True)

RANK = 3


# ----------------------------
# Closed-form KL
# ----------------------------
def test_kl_is_zero_for_a_standard_normal():
    assert kl_from_moments(0.0, 1.0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("mean_z, std_z", [(0.5, 1.0), (0.0, 2.0), (0.0, 0.5), (-1.0, 1.3)])
def test_kl_is_positive_away_from_standard_normal(mean_z, std_z):
    assert kl_from_moments(mean_z, std_z) > 0.0


def test_kl_matches_the_analysis_script_formula():
    """The inline expression from examples_paper/toy/analyse_toy.py."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        mu, sigma = rng.normal(), abs(rng.normal()) + 0.1
        expected = -np.log(sigma) + (sigma**2 + mu**2) / 2.0 - 0.5
        assert kl_from_moments(mu, sigma) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("std_z", [0.0, -1.0])
def test_kl_is_infinite_for_a_degenerate_spread(std_z):
    assert np.isinf(kl_from_moments(0.0, std_z))


# ----------------------------
# ZMoments arithmetic
# ----------------------------
def test_moments_recover_the_sample_statistics():
    rng = np.random.default_rng(1)
    z = rng.normal(loc=0.3, scale=1.7, size=(200, 50))
    W = rng.uniform(0.5, 2.0, size=z.shape)
    r = rng.normal(size=z.shape)
    m = moments_from_z(jnp.array(z), jnp.array(W), jnp.array(r))
    assert m.n == z.size
    assert m.mean_z == pytest.approx(z.mean(), rel=1e-10)
    assert m.std_z == pytest.approx(z.std(), rel=1e-10)
    assert m.chi2_red == pytest.approx(np.mean(W * r * r), rel=1e-10)
    assert m.rmse == pytest.approx(np.sqrt(np.mean(W * r * r)), rel=1e-10)


def test_variance_never_goes_negative():
    """Cancellation in sum_z2 - n*mean^2 must clamp, not produce a NaN std."""
    m = ZMoments(n=10, sum_z=10.0, sum_z2=9.999999999, sum_chi2=1.0)
    assert m.var_z == 0.0
    assert m.std_z == 0.0
    assert np.isinf(m.kl)


def test_as_dict_carries_every_score():
    m = ZMoments(n=4, sum_z=1.0, sum_z2=5.0, sum_chi2=8.0)
    d = m.as_dict()
    assert set(d) == {"kl", "mean_z", "std_z", "chi2_red", "rmse", "n"}
    assert d["kl"] == m.kl
    assert d["std_z"] == m.std_z


# ----------------------------
# z-scores and moments on real fits
# ----------------------------
@pytest.fixture(scope="module")
def fitted():
    prob = make_outlier_problem(N=150, M=40, K=RANK)
    model = Robusta(rank=RANK, robust=True, robust_scale=3.0, conv_tol=1e-10)
    state, _ = model.fit(jnp.array(prob["Y"]), jnp.array(prob["W"]), max_iter=100)
    return prob, model, state


def test_z_scores_match_the_analysis_script_definition(fitted):
    """`residuals * sqrt(W) * sqrt(robust_weights)` from analysis_funcs.py."""
    prob, model, state = fitted
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    expected = (
        np.asarray(model.residuals(Y, state=state))
        * np.sqrt(np.asarray(W))
        * np.sqrt(np.asarray(model.robust_weights(Y, W, state=state)))
    )
    got = np.asarray(z_scores(Y, W, state.A, state.G, model._hmf.likelihood))
    assert np.max(np.abs(got - expected)) < 1e-12


def test_z_moments_match_the_analysis_script_metrics(fitted):
    """std_z, chi2_red and rmse as analysis_funcs.py computed them."""
    prob, model, state = fitted
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    resid = np.asarray(model.residuals(Y, state=state))
    rw = np.asarray(model.robust_weights(Y, W, state=state))
    z = resid * np.sqrt(np.asarray(W)) * np.sqrt(rw)

    m = z_moments(Y, W, state.A, state.G, model._hmf.likelihood)
    assert m.std_z == pytest.approx(float(np.std(z)), rel=1e-10)
    assert m.chi2_red == pytest.approx(float(np.mean(np.asarray(W) * resid**2)), rel=1e-10)
    assert m.rmse == pytest.approx(float(np.sqrt(np.mean(np.asarray(W) * resid**2))), rel=1e-10)


def test_mad_z_matches_the_analysis_script_metric(fitted):
    prob, model, state = fitted
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    resid = np.asarray(model.residuals(Y, state=state))
    rw = np.asarray(model.robust_weights(Y, W, state=state))
    expected = float(np.median(np.abs(resid * np.sqrt(np.asarray(W)) * np.sqrt(rw))))
    assert mad_z(Y, W, state.A, state.G, model._hmf.likelihood) == pytest.approx(
        expected, rel=1e-10
    )


def test_gaussian_likelihood_leaves_z_undownweighted():
    """With no robust factor, z is just the ordinary standardised residual."""
    prob = make_low_rank_problem(N=80, M=20, K=RANK)
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    A = jnp.zeros((80, RANK))
    G = jnp.zeros((20, RANK))
    z = np.asarray(z_scores(Y, W, A, G, GaussianLikelihood()))
    assert np.max(np.abs(z - np.asarray(Y) * np.sqrt(np.asarray(W)))) < 1e-12


def test_robust_downweighting_shrinks_std_z_relative_to_gaussian():
    """The whole point of the robust score: outliers stop dominating sigma_z."""
    prob = make_outlier_problem(N=150, M=40, K=RANK, outlier_nsigma=80.0)
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    model = Robusta(rank=RANK, robust=True, robust_scale=3.0, conv_tol=1e-10)
    state, _ = model.fit(Y, W, max_iter=100)

    robust = z_moments(Y, W, state.A, state.G, StudentTLikelihood(nu=1.0, scale=3.0))
    gaussian = z_moments(Y, W, state.A, state.G, GaussianLikelihood())
    assert robust.std_z < gaussian.std_z
    assert robust.kl < gaussian.kl


# ----------------------------
# Robusta.score
# ----------------------------
def test_score_uses_the_models_own_likelihood(fitted):
    prob, model, state = fitted
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    got = model.score(Y, W, state=state)
    expected = z_moments(Y, W, state.A, state.G, model._hmf.likelihood)
    assert got.kl == pytest.approx(expected.kl, rel=1e-12)
    assert got.n == expected.n


def test_score_defaults_to_the_last_fit(fitted):
    prob, model, state = fitted
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    assert model.score(Y, W).kl == pytest.approx(model.score(Y, W, state=state).kl, rel=1e-12)


def test_score_without_a_fit_raises():
    model = Robusta(rank=RANK)
    prob = make_low_rank_problem(N=20, M=10, K=RANK)
    with pytest.raises(ValueError, match="fit"):
        model.score(jnp.array(prob["Y"]), jnp.array(prob["W"]))


def test_lower_kl_prefers_the_better_rank():
    """The score has to actually rank models, which is what it exists for."""
    prob = make_low_rank_problem(N=200, M=40, K=RANK)
    Y, W = jnp.array(prob["Y"]), jnp.array(prob["W"])
    scores = {}
    for rank in (1, RANK):
        model = Robusta(rank=rank, robust=False, conv_tol=1e-10)
        state, _ = model.fit(Y, W, max_iter=200)
        scores[rank] = model.score(Y, W, state=state).kl
    assert scores[RANK] < scores[1]
