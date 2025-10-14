# test_convergence.py

import jax
import jax.numpy as jnp
import pytest
from robusta_hmf.convergence import ConvergenceTester, max_frac_mat
from robusta_hmf.state import RHMFState

jax.config.update("jax_enable_x64", True)


def setup_matrices(seed, shape, tol):
    key = jax.random.key(0)
    mat1 = jax.random.normal(key, shape)
    mat2 = mat1 + jax.random.uniform(key, shape, minval=0, maxval=tol)
    return mat1, mat2


# ----------------------------
# Test invalid setups
# ----------------------------


def test_invalid_strategy():
    with pytest.raises(ValueError):
        ConvergenceTester(strategy="invalid")


# ----------------------------
# Test defaults
# ----------------------------


def test_defaults():
    conv = ConvergenceTester()
    assert conv.strategy == "max_frac_G"
    assert conv.tol == 1e-6


# ----------------------------
# Test diagnostics work
# ----------------------------


@pytest.mark.parametrize("shape", [(200, 200), (100, 200), (100, 200)])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6, 1e-8])
def test_max_frac_func(shape, tol):
    mat1, mat2 = setup_matrices(0, shape, tol)
    assert max_frac_mat(mat1, mat2, tol)


@pytest.mark.parametrize("shape", [(200, 200), (100, 200), (100, 200)])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6, 1e-8])
def test_conv_tester_A_max_frac(shape, tol):
    conv = ConvergenceTester(strategy="max_frac_A", tol=tol)
    # Version that is converged
    mat1, mat2 = setup_matrices(0, shape, tol)
    old_state = RHMFState(A=mat1, G=jnp.zeros_like(mat1))
    new_state = RHMFState(A=mat2, G=jnp.zeros_like(mat2))
    assert conv.is_converged(old_state, new_state)
    # Version that is not converged
    mat2 = mat1.copy()
    mat2 = mat2.at[0, 0].set(1e4)
    old_state = RHMFState(A=mat1, G=jnp.zeros_like(mat1))
    new_state = RHMFState(A=mat2, G=jnp.zeros_like(mat2))
    assert not conv.is_converged(old_state, new_state)


@pytest.mark.parametrize("shape", [(200, 200), (100, 200), (100, 200)])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6, 1e-8])
def test_conv_tester_G_max_frac(shape, tol):
    conv = ConvergenceTester(strategy="max_frac_G", tol=tol)
    # Version that is converged
    mat1, mat2 = setup_matrices(0, shape, tol)
    old_state = RHMFState(A=jnp.zeros_like(mat1), G=mat1)
    new_state = RHMFState(A=jnp.zeros_like(mat2), G=mat2)
    assert conv.is_converged(old_state, new_state)
    # Version that is not converged
    mat2 = mat1.copy()
    mat2 = mat2.at[0, 0].set(1e4)
    old_state = RHMFState(A=jnp.zeros_like(mat1), G=mat1)
    new_state = RHMFState(A=jnp.zeros_like(mat2), G=mat2)
    assert not conv.is_converged(old_state, new_state)
