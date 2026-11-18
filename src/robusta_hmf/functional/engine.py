# engine.py

"""The optimisation loop of `FunctionalRobusta`, independent of the bases.

A *layout* holds the data and knows how to compute residuals and the two
least-squares steps on them. `FlatLayout` works on any pair of bases through
the `Basis` interface. `DenseLayout` is the fast path for `Discrete` bases
on both axes with rectangular data: it keeps ``Y`` and ``W`` as ``(N, M)``
matrices and runs the dense kernels of `robusta_hmf.als`.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..als import normal_equations, solve_normal_equations
from ..convergence import ConvergenceTester, RunMonitor, max_frac_mat, rel_frac_loss
from ..fit_info import FitInfo
from ..frame import check_factors_finite, make_fit_info
from ..likelihoods import Likelihood
from ..precision import nonfinite_loss_message
from ..rotations import whitening_transform
from .bases import Basis, Discrete
from .measurements import Measurements
from .state import FunctionalState

#: Diagonal shift and eigenvalue floor of the whitening rotation (as `FastAffine`).
ROTATION_EPS = 1e-6


class DenseLayout(eqx.Module):
    """Rectangular data as ``(N, M)`` matrices, for `Discrete` bases on both axes."""

    Y: Array
    W: Array

    @property
    def w_data(self) -> Array:
        return self.W

    def model(self, state: FunctionalState) -> Array:
        return state.row_coeffs @ state.col_coeffs.T

    def residuals(self, state: FunctionalState) -> Array:
        return self.Y - self.model(state)

    def row_step(self, state: FunctionalState, Wt: Array) -> Array:
        G = state.col_coeffs
        M, b = normal_equations(Wt, Wt * self.Y, G)
        return solve_normal_equations(M, b, None, n_obs=G.shape[0])

    def col_step(self, state: FunctionalState, Wt: Array) -> Array:
        A = state.row_coeffs
        M, b = normal_equations(Wt.T, (Wt * self.Y).T, A)
        return solve_normal_equations(M, b, None, n_obs=A.shape[0])


class FlatLayout(eqx.Module):
    """A flat list of measurements, for any pair of bases."""

    data: Measurements
    row_basis: Basis = eqx.field(static=True)
    col_basis: Basis = eqx.field(static=True)

    @property
    def w_data(self) -> Array:
        return self.data.w

    def row_values(self, state: FunctionalState) -> Array:
        return self.row_basis.evaluate(state.row_coeffs, self.data.row)

    def col_values(self, state: FunctionalState) -> Array:
        return self.col_basis.evaluate(state.col_coeffs, self.data.col)

    def model(self, state: FunctionalState) -> Array:
        return jnp.sum(self.row_values(state) * self.col_values(state), axis=1)

    def residuals(self, state: FunctionalState) -> Array:
        return self.data.y - self.model(state)

    def row_step(self, state: FunctionalState, Wt: Array) -> Array:
        return self.row_basis.solve(
            self.data.row, self.col_values(state), self.data.y, Wt, self.col_basis.n_coeffs
        )

    def col_step(self, state: FunctionalState, Wt: Array) -> Array:
        return self.col_basis.solve(
            self.data.col, self.row_values(state), self.data.y, Wt, self.row_basis.n_coeffs
        )


Layout = DenseLayout | FlatLayout


def make_layout(data: Measurements, row_basis: Basis, col_basis: Basis) -> Layout:
    """`DenseLayout` when it applies (see the module docstring), else `FlatLayout`."""
    dense = (
        isinstance(row_basis, Discrete)
        and isinstance(col_basis, Discrete)
        and row_basis.fast
        and col_basis.fast
        and data.shape == (row_basis.n, col_basis.n)
    )
    if dense:
        return DenseLayout(*data.as_matrices())
    return FlatLayout(data, row_basis, col_basis)


def rotate(state: FunctionalState, gram: Array | None) -> FunctionalState:
    """Whiten the column functions under the column basis's Gram matrix.

    Leaves the model unchanged and makes ``γᵀ S γ = I`` (to the eigenvalue
    floor), as `FastAffine` does for ``GᵀG``.
    """
    col = state.col_coeffs
    C = col.T @ col if gram is None else col.T @ (gram @ col)
    R, R_inv = whitening_transform(C, ROTATION_EPS)
    return eqx.tree_at(
        lambda s: (s.row_coeffs, s.col_coeffs),
        state,
        (state.row_coeffs @ R_inv, col @ R),
    )


@eqx.filter_jit
def step(
    layout: Layout,
    likelihood: Likelihood,
    gram: Array | None,
    state: FunctionalState,
    do_rotate: bool,
) -> tuple[FunctionalState, Array]:
    """One iteration: w-step, row step, column step, optional rotation; returns the loss."""
    w_data = layout.w_data
    Wt = w_data * likelihood.weights_from_residuals(layout.residuals(state), w_data)
    state = eqx.tree_at(lambda s: s.row_coeffs, state, layout.row_step(state, Wt))
    state = eqx.tree_at(lambda s: s.col_coeffs, state, layout.col_step(state, Wt))
    if do_rotate:
        state = rotate(state, gram)
    loss = likelihood.loss_from_residuals(layout.residuals(state), w_data)
    state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
    return state, loss


def is_converged(
    tester: ConvergenceTester,
    old: FunctionalState,
    new: FunctionalState,
    old_loss,
    new_loss,
) -> bool:
    """`ConvergenceTester`'s strategies, with ``G`` the column and ``A`` the row coefficients."""
    if tester.strategy == "max_frac_G":
        return bool(max_frac_mat(old.col_coeffs, new.col_coeffs, tol=tester.tol))
    if tester.strategy == "max_frac_A":
        return bool(max_frac_mat(old.row_coeffs, new.row_coeffs, tol=tester.tol))
    if tester.strategy == "rel_frac_loss":
        return bool(rel_frac_loss(old_loss, new_loss, tol=tester.tol))
    return False


def run(
    layout: Layout,
    likelihood: Likelihood,
    gram: Array | None,
    tester: ConvergenceTester,
    monitor: RunMonitor,
    state: FunctionalState,
    max_iter: int,
    conv_check_cadence: int,
    dtype,
) -> tuple[FunctionalState, Array, FitInfo]:
    """Iterate `step` until converged or ``max_iter``.

    The schedule is `Robusta`'s ALS schedule: every iteration but the first is
    rotated, and convergence is tested every ``conv_check_cadence``
    iterations against the state at the previous test.
    """
    loss_history = []
    prev_state, prev_loss = state, jnp.inf
    converged = False
    for i in range(max_iter):
        state, loss = step(layout, likelihood, gram, state, i != 0)
        loss_history.append(loss)
        if i % conv_check_cadence == 0 and i != 0:
            monitor.update(loss_history)
            if is_converged(tester, prev_state, state, prev_loss, loss):
                print(f"Converged at iteration {i}")
                converged = True
                break
            prev_state, prev_loss = state, loss
            print(f"iter {state.it:03d} | loss {loss:.4f}", flush=True)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(nonfinite_loss_message(i, loss, dtype))
            check_factors_finite(state.row_coeffs, state.col_coeffs, i, dtype)

    loss_rises = monitor.finish(loss_history, converged, max_iter, tester.strategy)
    if loss_history and not np.isfinite(float(loss_history[-1])):
        last = len(loss_history) - 1
        raise FloatingPointError(nonfinite_loss_message(last, loss_history[-1], dtype))
    return state, jnp.array(loss_history), make_fit_info(converged, loss_history, loss_rises)
