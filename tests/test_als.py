# test_als.py

import jax
import jax.numpy as jnp
import pytest
from jax.numpy.linalg import norm
from robusta_hmf.als import (
    WeightedAStep,
    WeightedGStep,
    normal_equations,
    solve_normal_equations,
)
from robusta_hmf.state import RHMFState

jax.config.update("jax_enable_x64", True)


def rng(seed=0, N=6, M=5, K=3):
    key = jax.random.key(seed)
    Y = jax.random.normal(key, (N, M))
    W = jax.random.uniform(key, (N, M), minval=0.5, maxval=2.0)
    A = jax.random.normal(key, (N, K))
    G = jax.random.normal(key, (M, K))
    return Y, W, A, G


def get_init_problem(seed=0, N=6, M=5, K=3):
    Y, W, A, G = rng(seed, N, M, K)
    return Y, W, RHMFState(A, G, it=0)


def resid(Y, A, G):
    return Y - A @ G.T


# ----------------------------
# Verify solutions
# ----------------------------
@pytest.mark.parametrize(
    "stepper",
    [
        WeightedAStep(),
        WeightedGStep(),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (6, 5, 3),  # N > M > K
        (6, 6, 3),  # N = M > K
        (5, 6, 3),  # N < M > K
        (6, 5, 5),  # N > M = K
        (6, 6, 6),  # N = M = K
        (5, 6, 6),  # N < M = K
        (6, 5, 7),  # N > M < K
        (6, 6, 7),  # N = M < K
        (5, 6, 7),  # N < M < K
        (1001, 1001, 5),  # Largeish N, M small K
    ],
)
def test_step_solution(stepper, shape, tol=1e-8):
    # Setup
    N, M, K = shape
    Y, W, state = get_init_problem(0, N, M, K)
    # Take a step
    new_state = stepper(Y, W, state)
    # Check solution optimality wrt. normal eqns
    A, G = new_state.A, new_state.G
    R = resid(Y, A, G)
    # A step conds
    K_A = (W * R) @ G
    A_normal_cond = jnp.allclose(K_A, jnp.zeros((N, K)))
    A_KKT_cond = norm(K_A) / (norm(W * Y) * norm(G, 2)) < tol
    A_orth_cond = norm((W * R) @ G) < tol
    A_conds = A_normal_cond and A_KKT_cond and A_orth_cond
    # G step conds
    K_G = A.T @ (W * R)
    G_normal_cond = jnp.allclose(K_G, jnp.zeros((K, M)))
    G_KKT_cond = norm(K_G) / (norm(A, 2) * norm(W * Y)) < tol
    G_orth_cond = norm(A.T @ (W * R)) < tol
    G_conds = G_normal_cond and G_KKT_cond and G_orth_cond
    # One of these should pass depending if it was A or G step
    assert A_conds or G_conds


# ----------------------------
# Verify ridge solutions
# ----------------------------
@pytest.mark.parametrize("ridge", [1e-3, 0.1, 10.0])
def test_a_step_ridge_solution(ridge, tol=1e-8):
    N, M, K = 6, 5, 3
    Y, W, state = get_init_problem(0, N, M, K)
    new_state = WeightedAStep(ridge=ridge)(Y, W, state)
    A, G = new_state.A, new_state.G
    # Each row satisfies (G^T W_i G + ridge I) a_i = G^T W_i y_i
    for i in range(N):
        lhs = (G.T * W[i]) @ G + ridge * jnp.eye(K)
        rhs = (G.T * W[i]) @ Y[i]
        assert norm(lhs @ A[i] - rhs) / norm(rhs) < tol


@pytest.mark.parametrize("ridge", [1e-3, 0.1, 10.0])
def test_g_step_ridge_solution(ridge, tol=1e-8):
    N, M, K = 6, 5, 3
    Y, W, state = get_init_problem(0, N, M, K)
    new_state = WeightedGStep(ridge=ridge)(Y, W, state)
    A, G = new_state.A, new_state.G
    # Each column satisfies (A^T W_j A + ridge I) g_j = A^T W_j y_j
    for j in range(M):
        lhs = (A.T * W[:, j]) @ A + ridge * jnp.eye(K)
        rhs = (A.T * W[:, j]) @ Y[:, j]
        assert norm(lhs @ G[j] - rhs) / norm(rhs) < tol


# ----------------------------
# Normal equations agree with the whitened formulation
# ----------------------------
@pytest.mark.parametrize("shape", [(6, 5, 3), (12, 7, 4), (200, 50, 3)])
def test_normal_equations_match_whitening(shape, tol=1e-10):
    N, M, K = shape
    Y, W, _, G = rng(0, N, M, K)
    Mat, b = normal_equations(W, W * Y, G)
    for i in range(min(N, 8)):
        s = jnp.sqrt(W[i])
        Gw = G * s[:, None]
        assert norm(Mat[i] - Gw.T @ Gw) / norm(Gw.T @ Gw) < tol
        assert norm(b[i] - Gw.T @ (s * Y[i])) / norm(Gw.T @ (s * Y[i])) < tol


# ----------------------------
# Degenerate rows and pixels: zero weights must not poison the batch
# ----------------------------
def test_a_step_all_zero_weight_row():
    """A spectrum masked in every pixel gives a zero coefficient row, not NaN."""
    N, M, K = 6, 5, 3
    Y, W, state = get_init_problem(0, N, M, K)
    W = W.at[2, :].set(0.0)  # row 2 entirely masked
    A_new = WeightedAStep()(Y, W, state).A
    assert jnp.all(jnp.isfinite(A_new))
    assert jnp.allclose(A_new[2], 0.0)
    # Every other row is unaffected by the degenerate one.
    keep = jnp.array([0, 1, 3, 4, 5])
    A_ref = WeightedAStep()(Y[keep], W[keep], state).A
    assert jnp.allclose(A_new[keep], A_ref)


def test_g_step_all_zero_weight_pixel():
    """A pixel masked in every spectrum gives a zero basis row, not NaN."""
    N, M, K = 6, 5, 3
    Y, W, state = get_init_problem(0, N, M, K)
    W = W.at[:, 1].set(0.0)  # pixel 1 masked everywhere
    G_new = WeightedGStep()(Y, W, state).G
    assert jnp.all(jnp.isfinite(G_new))
    assert jnp.allclose(G_new[1], 0.0)


def test_solve_normal_equations_guard_is_elementwise():
    """Only the empty entries are substituted; the rest solve normally."""
    K = 3
    Mat = jnp.stack([jnp.eye(K) * 2.0, jnp.zeros((K, K)), jnp.eye(K) * 4.0])
    b = jnp.stack([jnp.ones(K), jnp.ones(K), jnp.ones(K)])
    x = solve_normal_equations(Mat, b)
    assert jnp.allclose(x[0], 0.5)
    assert jnp.allclose(x[1], 0.0)
    assert jnp.allclose(x[2], 0.25)


@pytest.mark.parametrize("ridge", [1e-6, 0.1])
def test_ridge_makes_empty_rows_zero_too(ridge):
    """With a ridge the guard never fires, but empty rows still solve to zero."""
    N, M, K = 6, 5, 3
    Y, W, state = get_init_problem(0, N, M, K)
    W = W.at[2, :].set(0.0)
    A_new = WeightedAStep(ridge=ridge)(Y, W, state).A
    assert jnp.all(jnp.isfinite(A_new))
    assert jnp.allclose(A_new[2], 0.0)


@pytest.mark.parametrize("shape", [(6, 5, 7), (5, 6, 7), (4, 3, 6)])
def test_rank_deficient_steps_give_min_norm_solution(shape):
    """With K > rows of the fixed factor, each step matches lstsq on the whitened problem."""
    N, M, K = shape
    Y, W, state = get_init_problem(0, N, M, K)
    A_new = WeightedAStep()(Y, W, state).A
    G_new = WeightedGStep()(Y, W, state).G
    sw = jnp.sqrt(W)
    for i in range(N):
        a_ref = jnp.linalg.lstsq(sw[i, :, None] * state.G, sw[i] * Y[i])[0]
        assert jnp.allclose(A_new[i], a_ref, atol=1e-10)
    for j in range(M):
        g_ref = jnp.linalg.lstsq(sw[:, j, None] * state.A, sw[:, j] * Y[:, j])[0]
        assert jnp.allclose(G_new[j], g_ref, atol=1e-10)
    assert jnp.all(jnp.isfinite(A_new)) and jnp.all(jnp.isfinite(G_new))
