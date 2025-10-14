# test_hmf.py

import jax
import jax.numpy as jnp
import pytest
from robusta_hmf.hmf import ALS_HMF, SGD_HMF
from robusta_hmf.initialisation import Initialiser

jax.config.update("jax_enable_x64", True)


def rng(seed=0, N=6, M=5):
    key = jax.random.key(seed)
    Y = jax.random.normal(key, (N, M))
    W = jax.random.uniform(key, (N, M), minval=0.5, maxval=2.0)
    return Y, W


def get_init_problem(seed=0, N=6, M=5, K=3, opt=None):
    Y, W = rng(seed, N, M)
    init = Initialiser(N, M, K, strategy="svd")
    state, opt = init.execute(Y=Y, opt=opt)
    return Y, W, state, opt


# ----------------------------
# Verify ALS_HMF
# ----------------------------
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
def test_als_hmf_step(shape, tol=1e-8):
    # Setup
    N, M, K = shape
    # Should raise error if K > min(N, M) for svd init
    if K > min(N, M):
        with pytest.raises(ValueError):
            Y, W, state, _ = get_init_problem(0, N, M, K)
        return
    Y, W, state, _ = get_init_problem(0, N, M, K)
    model = ALS_HMF()
    # Take a step
    new_state, loss = model.step(Y, W, state)
    assert new_state.it == state.it + 1
    assert jnp.isscalar(loss)
    assert loss >= 0.0
    assert new_state.A.shape == (N, K)
    assert new_state.G.shape == (M, K)
    # Loss should reduce
    new_state2, loss2 = model.step(Y, W, new_state)
    assert loss2 <= loss + tol
    # Rotation should make no difference to loss
    new_state3, loss3 = model.step(Y, W, new_state, rotate=False)
    assert jnp.isclose(loss3, loss2, atol=tol)


# ----------------------------
# Verify SGD_HMF
# ----------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (6, 5, 3),  # N > M > K
        (6, 6, 3),  # N = M > K
        (5, 6, 3),  # N < M > K
        (6, 5, 5),  # N > M = K
        (6, 6, 6),  # N = M = K
        (1001, 1001, 5),  # Largeish N, M small K
    ],
)
def test_sgd_hmf_step(shape, tol=1e-8):
    # Setup
    N, M, K = shape
    model = SGD_HMF()
    Y, W, state, opt = get_init_problem(0, N, M, K, opt=model.opt)
    # Take a step
    new_state, loss = model.step(Y, W, state)
    assert new_state.it == state.it + 1
    assert jnp.isscalar(loss)
    assert loss >= 0.0
    assert new_state.A.shape == (N, K)
    assert new_state.G.shape == (M, K)
    # Loss should reduce
    new_state2, loss2 = model.step(Y, W, new_state)
    assert loss2 <= loss + tol
    # Rotation should make no difference to loss
    new_state3, loss3 = model.step(Y, W, new_state, rotate=False)
    assert jnp.isclose(loss3, loss2, atol=tol)
