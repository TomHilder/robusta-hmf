# test_initialisation.py

import jax
import jax.numpy as jnp
import optax
import pytest
from robusta_hmf.initialisation import Initialiser

jax.config.update("jax_enable_x64", True)


# ----------------------------
# Test random initialisation
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
def test_random_init_shapes(shape):
    N, M, K = shape
    init = Initialiser(N, M, K, strategy="random")
    state, _ = init.execute(seed=0)
    assert state.A.shape == (N, K)
    assert state.G.shape == (M, K)


def test_random_init_determinism():
    N, M, K = 6, 5, 3
    init = Initialiser(N, M, K, strategy="random")
    state1, _ = init.execute(seed=0)
    state2, _ = init.execute(seed=0)
    assert jnp.allclose(state1.A, state2.A)
    assert jnp.allclose(state1.G, state2.G)


def test_random_init_different_seeds():
    N, M, K = 6, 5, 3
    init = Initialiser(N, M, K, strategy="random")
    state1, _ = init.execute(seed=0)
    state2, _ = init.execute(seed=1)
    assert not jnp.allclose(state1.A, state2.A)
    assert not jnp.allclose(state1.G, state2.G)


def test_random_no_seed():
    N, M, K = 6, 5, 3
    init = Initialiser(N, M, K, strategy="random")
    with pytest.raises(ValueError):
        init.execute(seed=None)


# ----------------------------
# Test svd initialisation
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
        (6, 5, 3),  # N > M < K (Invalid)
        (6, 6, 7),  # N = M < K (Invalid)
        (5, 6, 7),  # N < M < K (Invalid)
        (1001, 1001, 5),  # Largeish N, M small K
    ],
)
def test_svd_init_shapes(shape):
    N, M, K = shape
    Y = jax.random.normal(jax.random.key(0), (N, M))
    if K > min(N, M):
        with pytest.raises(ValueError):
            init = Initialiser(N, M, K, strategy="svd")
            init.execute(seed=0, Y=Y)
    else:
        init = Initialiser(N, M, K, strategy="svd")
        state, _ = init.execute(seed=0, Y=Y)
        assert state.A.shape == (N, K)
        assert state.G.shape == (M, K)


def test_svd_no_Y():
    N, M, K = 6, 5, 3
    init = Initialiser(N, M, K, strategy="svd")
    with pytest.raises(ValueError):
        init.execute(seed=0, Y=None)


def test_svd_wrong_Y_shape():
    N, M, K = 6, 5, 3
    Y = jax.random.normal(jax.random.key(0), (N + 1, M))
    init = Initialiser(N, M, K, strategy="svd")
    with pytest.raises(ValueError):
        init.execute(seed=0, Y=Y)


def test_svd_determinism():
    N, M, K = 6, 5, 3
    Y = jax.random.normal(jax.random.key(0), (N, M))
    init = Initialiser(N, M, K, strategy="svd")
    state1, _ = init.execute(seed=0, Y=Y)
    state2, _ = init.execute(seed=0, Y=Y)
    assert jnp.allclose(state1.A, state2.A)
    assert jnp.allclose(state1.G, state2.G)


@pytest.mark.parametrize(
    "shape",
    [
        (12, 10, 3),  # low rank
        (12, 10, 10),  # perfect reconstruction
        (10, 12, 3),  # low rank
        (10, 12, 10),  # perfect reconstruction
        (10, 10, 3),  # low rank
        (10, 10, 10),  # perfect reconstruction
    ],
)
def test_svd_reconstruction(shape):
    N, M, K = shape
    key = jax.random.key(0)
    Y_true = jax.random.normal(key, (N, M))
    init = Initialiser(N, M, K, strategy="svd")
    state, _ = init.execute(seed=0, Y=Y_true)
    Y_recon = state.A @ state.G.T
    if K >= min(N, M):
        assert jnp.allclose(Y_recon, Y_true)
    else:
        # Check that reconstruction is close in Frobenius norm
        frob_norm = jnp.linalg.norm(Y_recon - Y_true, ord="fro")
        assert frob_norm < jnp.linalg.norm(Y_true, ord="fro")


# ----------------------------
# Test custom initialisation
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
def test_custom_init_shapes(shape):
    N, M, K = shape
    A = jax.random.normal(jax.random.key(0), (N, K))
    G = jax.random.normal(jax.random.key(1), (M, K))
    init = Initialiser(N, M, K, strategy="custom")
    state, _ = init.execute(seed=0, A=A, G=G)
    assert state.A.shape == (N, K)
    assert state.G.shape == (M, K)


def test_custom_no_A_G():
    N, M, K = 6, 5, 3
    init = Initialiser(N, M, K, strategy="custom")
    with pytest.raises(ValueError):
        init.execute(seed=0, A=None, G=None)


# ----------------------------
# Test optax state initialisation
# ----------------------------


def test_optax_state_init():
    N, M, K = 6, 5, 3
    init = Initialiser(N, M, K, strategy="random")
    state, _ = init.execute(seed=0)
    assert state.opt_state is None

    opt = optax.adam(1e-3)
    state, _ = init.execute(seed=0, opt=opt)
    assert state.opt_state is not None
    jax.tree_util.tree_flatten(state.opt_state)  # Should not error


# ----------------------------
# Test invalid strategy
# ----------------------------


def test_invalid_strategy():
    N, M, K = 6, 5, 3
    with pytest.raises(ValueError):
        Initialiser(N, M, K, strategy="invalid")
