# conftest.py

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def make_low_rank_problem(seed=0, N=200, M=50, K=3, sigma_range=(0.05, 0.15)):
    """Y = A_true @ G_true.T + heteroskedastic Gaussian noise, with W = 1/sigma^2."""
    rng = np.random.default_rng(seed)
    A_true = rng.normal(size=(N, K))
    G_true = rng.normal(size=(M, K))
    sigma = rng.uniform(*sigma_range, size=(N, M))
    Y_clean = A_true @ G_true.T
    Y = Y_clean + sigma * rng.normal(size=(N, M))
    W = 1.0 / sigma**2
    return {
        "Y": Y,
        "W": W,
        "A_true": A_true,
        "G_true": G_true,
        "sigma": sigma,
        "Y_clean": Y_clean,
    }


def make_outlier_problem(seed=1, N=200, M=50, K=3, outlier_frac=0.01, outlier_nsigma=50.0):
    """Low-rank problem with a fraction of pixels corrupted by large outliers."""
    prob = make_low_rank_problem(seed=seed, N=N, M=M, K=K)
    rng = np.random.default_rng(seed + 1000)
    mask = rng.random((N, M)) < outlier_frac
    signs = np.where(rng.random((N, M)) < 0.5, -1.0, 1.0)
    prob["Y"] = prob["Y"] + mask * signs * outlier_nsigma * prob["sigma"]
    prob["outlier_mask"] = mask
    return prob


def subspace_alignment(G1, G2):
    """Minimum cosine of the principal angles between column spaces (1 = identical span)."""
    Q1 = np.linalg.qr(np.asarray(G1))[0]
    Q2 = np.linalg.qr(np.asarray(G2))[0]
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(s.min())


@pytest.fixture(scope="session")
def low_rank_problem():
    return make_low_rank_problem()


@pytest.fixture(scope="session")
def outlier_problem():
    return make_outlier_problem()
