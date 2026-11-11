# test_hmf.py
#
# Step-level tests for the unified HMF engine.

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from robusta_hmf.hmf import HMF
from robusta_hmf.initialisation import Initialiser
from robusta_hmf.state import refresh_opt_state

jax.config.update("jax_enable_x64", True)


def rng(seed=0, N=30, M=20):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    Y = jax.random.normal(k1, (N, M))
    W = jax.random.uniform(k2, (N, M), minval=0.5, maxval=2.0)
    return Y, W


def get_init_problem(seed=0, N=30, M=20, K=3, opt=None):
    Y, W = rng(seed, N, M)
    init = Initialiser(N, M, K, strategy="svd")
    state = init.execute(Y=Y, opt=opt)
    return Y, W, state


# ----------------------------
# ALS steps
# ----------------------------
@pytest.mark.parametrize("robust", [False, True])
def test_step_als_reduces_loss(robust):
    hmf = HMF(method="als", robust=robust, robust_scale=3.0)
    Y, W, state = get_init_problem()
    loss_init = hmf.likelihood.loss(Y, W, state.A, state.G)
    state, loss = hmf.step_als(Y, W, state)
    assert jnp.isfinite(loss)
    assert loss < loss_init


@pytest.mark.parametrize("robust", [False, True])
def test_step_als_monotone_over_many_steps(robust):
    hmf = HMF(method="als", robust=robust, robust_scale=3.0)
    Y, W, state = get_init_problem()
    losses = []
    for _ in range(20):
        state, loss = hmf.step_als(Y, W, state, rotate=True)
        losses.append(float(loss))
    losses = np.array(losses)
    assert np.all(np.diff(losses) <= 1e-10 * abs(losses[0]))


def test_step_als_skip_G_fixes_basis():
    hmf = HMF(method="als")
    Y, W, state = get_init_problem()
    new_state, _ = hmf.step_als(Y, W, state, skip_G=True)
    assert jnp.array_equal(new_state.G, state.G)
    assert not jnp.array_equal(new_state.A, state.A)


def test_step_als_increments_iteration():
    hmf = HMF(method="als")
    Y, W, state = get_init_problem()
    new_state, _ = hmf.step_als(Y, W, state)
    assert new_state.it == state.it + 1


def test_rotation_preserves_reconstruction():
    hmf = HMF(method="als", rotation="fast")
    Y, W, state = get_init_problem()
    for _ in range(3):
        state, _ = hmf.step_als(Y, W, state)
    rotated = hmf.rotation(state)
    assert jnp.allclose(rotated.A @ rotated.G.T, state.A @ state.G.T)


# ----------------------------
# SGD steps
# ----------------------------
def test_step_sgd_reduces_loss_over_steps():
    hmf = HMF(method="sgd", robust=False, learning_rate=1e-2)
    Y, W, state = get_init_problem(opt=hmf.opt)
    loss_init = hmf.likelihood.loss(Y, W, state.A, state.G)
    for _ in range(50):
        state, loss = hmf.step_sgd(Y, W, state, rotate=False)
    assert jnp.isfinite(loss)
    assert loss < loss_init
    assert state.it == 50


def test_step_sgd_rotate_preserves_reconstruction():
    hmf = HMF(method="sgd", robust=False, learning_rate=1e-2)
    Y, W, state = get_init_problem(opt=hmf.opt)
    state, _ = hmf.step_sgd(Y, W, state, rotate=False)
    # A rotate step applies the SGD update then rotates; reconstruction after
    # rotation must match the un-rotated update
    state_plain, _ = hmf.step_sgd(Y, W, state, rotate=False)
    state_rot, _ = hmf.step_sgd(Y, W, state, rotate=True)
    assert jnp.allclose(state_rot.A @ state_rot.G.T, state_plain.A @ state_plain.G.T)


def test_custom_optimizer_is_used():
    opt = optax.adam(1e-3)
    hmf = HMF(method="sgd", custom_opt=opt)
    assert hmf.opt is opt


# ----------------------------
# Construction and validation
# ----------------------------
def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        HMF(method="not_a_method")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"robust": True, "robust_scale": 0.0},
        {"robust": True, "robust_scale": -1.0},
        {"robust": True, "robust_nu": 0.0},
        {"robust": True, "robust_nu": -2.0},
        {"method": "als", "als_ridge": -0.5},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        HMF(**kwargs)


def test_get_stepper_dispatch():
    Y, W, state = get_init_problem()
    als = HMF(method="als")
    _, loss_als = als.get_stepper()(Y=Y, W_data=W, state=state)
    assert jnp.isfinite(loss_als)

    sgd = HMF(method="sgd")
    Y, W, state = get_init_problem(opt=sgd.opt)
    state_sgd, loss_sgd = sgd.get_stepper()(Y=Y, W_data=W, state=state, rotate=False)
    assert jnp.isfinite(loss_sgd)
    assert state_sgd.opt_state is not None
