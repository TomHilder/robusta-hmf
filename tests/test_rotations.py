# test_rotations.py

import jax
import jax.numpy as jnp
import pytest
from robusta_hmf.rotations import (
    FastAffine,
    Identity,
    get_rotation_cls,
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


# ----------------------------
# Test shapes
# ----------------------------


@pytest.mark.parametrize(
    "rot",
    [
        Identity(),
        FastAffine(whiten=True),
        FastAffine(whiten=False),
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
def test_rotation_shapes(rot, shape):
    N, M, K = shape
    *_, init_state = get_init_problem(0, N, M, K)
    state = rot(init_state)
    assert state.A.shape == init_state.A.shape
    assert state.G.shape == init_state.G.shape


# ----------------------------
# Verify transformations
# ----------------------------


def test_identity():
    *_, init_state = get_init_problem()
    state = Identity()(init_state)
    assert jnp.allclose(state.A, init_state.A)
    assert jnp.allclose(state.G, init_state.G)


def test_fast_affine_orthogonality():
    # NOTE: G is not orthogonal after rotation, A is
    *_, init_state = get_init_problem()
    state = FastAffine(whiten=False)(init_state)
    AT_A = state.A.T @ state.A
    assert jnp.allclose(AT_A, jnp.diag(jnp.diag(AT_A)))


def test_fast_affine_orthonormality():
    *_, init_state = get_init_problem()
    state = FastAffine(whiten=True)(init_state)
    AT_A = state.A.T @ state.A
    assert jnp.allclose(AT_A, jnp.eye(*AT_A.shape), rtol=1e-6, atol=1e-6)


# ----------------------------
# Test get_rotation_cls
# ----------------------------


def test_get_rotation_cls():
    # Test that the correct class is returned
    cls = get_rotation_cls("fast")
    assert cls == FastAffine
    cls = get_rotation_cls("identity")
    assert cls == Identity
    # Test that an error is raised for unknown methods
    with pytest.raises(ValueError):
        get_rotation_cls("unknown_method")
