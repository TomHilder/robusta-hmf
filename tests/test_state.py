# test_state.py

import equinox as eqx
import jax
import jax.numpy as jnp
from optax import adam
from robusta_hmf.state import RHMFState, load_state_from_npz, save_state_to_npz

jax.config.update("jax_enable_x64", True)


def rng(seed=0, N=6, M=5, K=3):
    key = jax.random.key(seed)
    A = jax.random.normal(key, (N, K))
    G = jax.random.normal(key, (M, K))
    return A, G


# ----------------------------
# RHMFState tests
# ----------------------------
def test_rhmfstate_shapes():
    N, M, K = 6, 5, 3
    A, G = rng(N=N, M=M, K=K)
    state = RHMFState(A=A, G=G, it=0)
    assert state.A.shape == (N, K)
    assert state.G.shape == (M, K)
    assert state.it == 0
    assert state.opt_state is None


def test_rhmfstate_iteration_under_jit():
    N, M, K = 6, 5, 3
    A, G = rng(N=N, M=M, K=K)
    state = RHMFState(A=A, G=G, it=0)

    @jax.jit
    def increment_it(state):
        return eqx.tree_at(lambda s: s.it, state, state.it + 1)

    new_state = increment_it(state)
    assert new_state.it == 1
    assert jnp.allclose(new_state.A, state.A)
    assert jnp.allclose(new_state.G, state.G)


def test_rhmfstate_optax():
    N, M, K = 6, 5, 3
    A, G = rng(N=N, M=M, K=K)
    opt = adam(1e-2)
    opt_state = opt.init((A, G))
    state = RHMFState(A=A, G=G, it=0, opt_state=opt_state)
    assert state.opt_state is not None
    assert state.opt_state[0].count == 0


# ----------------------------
# Serialization round-trip
# ----------------------------
def test_npz_roundtrip(tmp_path):
    A, G = rng()
    state = RHMFState(A=A, G=G, it=7)
    path = tmp_path / "state.npz"
    save_state_to_npz(state, path)
    loaded = load_state_from_npz(path)
    assert jnp.array_equal(loaded.A, state.A)
    assert jnp.array_equal(loaded.G, state.G)
    assert loaded.it == 7
    assert loaded.opt_state is None


def test_npz_roundtrip_with_opt_state(tmp_path):
    A, G = rng()
    state = RHMFState(A=A, G=G, it=3)
    path = tmp_path / "state.npz"
    save_state_to_npz(state, path)
    opt = adam(1e-2)
    opt_state = opt.init((A, G))
    loaded = load_state_from_npz(path, opt_state=opt_state)
    assert loaded.opt_state is not None
