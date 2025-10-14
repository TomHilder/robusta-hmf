# test_als.py

import jax
import jax.numpy as jnp
import pytest
from jax.numpy.linalg import norm
from robusta_hmf.als import WeightedAStep, WeightedGStep
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
