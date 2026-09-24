# als.py

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .state import RHMFState, update_state


def normal_equations(W: Array, WY: Array, F: Array) -> tuple[Array, Array]:
    """Batched weighted normal equations for a fixed factor ``F``.

    Parameters
    ----------
    W : Array, shape (batch, P)
        Total weights (data weights times any IRLS downweighting).
    WY : Array, shape (batch, P)
        Weights already multiplied into the data, i.e. ``W * Y``.
    F : Array, shape (P, K)
        The factor held fixed: ``G`` for the A step, ``A`` for the G step.

    Returns
    -------
    M : Array, shape (batch, K, K)
        ``M[i] = F.T @ diag(W[i]) @ F``
    b : Array, shape (batch, K)
        ``b[i] = F.T @ (W[i] * Y[i])``

    Notes
    -----
    Written as two matmuls against the ``K*K`` outer products of ``F`` rather
    than whitening ``F`` per row. Whitening is the obvious formulation, but
    under ``vmap`` it gives every one of the ``batch`` rows its own ``(P, K)``
    copy of ``F``, so the peak intermediate is ``batch * P * K``. For the full
    Gaia RVS sample that is 43 GiB at K=5 and 258 GiB at K=30 -- larger than
    any single GPU. Here the largest intermediate is ``batch * K * K`` instead
    (3.3 GiB at K=30), for the same flop count.

    It also shards cleanly: with ``W`` and ``Y`` split along ``batch``, both
    matmuls are row-local, so a data-parallel split needs no cross-device
    communication in the A step and a single all-reduce of the small ``(K, K)``
    blocks in the G step. :mod:`robusta_hmf.sharded` relies on this.
    """
    FF = (F[:, :, None] * F[:, None, :]).reshape(F.shape[0], -1)  # (P, K*K)
    M = (W @ FF).reshape(W.shape[0], F.shape[1], F.shape[1])  # (batch, K, K)
    b = WY @ F  # (batch, K)
    return M, b


def solve_normal_equations(
    M: Array, b: Array, ridge: float | None = None, n_obs: int | None = None
) -> Array:
    """Batched solve of the ALS normal equations, tolerant of empty rows.

    Parameters
    ----------
    M : Array, shape (..., K, K)
        Batched normal-equation matrices.
    b : Array, shape (..., K)
        Batched right-hand sides.
    ridge : float | None, default=None
        Optional ridge added to the diagonal before solving.
    n_obs : int | None, default=None
        Number of rows in the fixed factor ``F`` that built ``M`` (pixels in the
        A step, rows in the G step). ``M`` has rank at most ``n_obs``, so when
        ``n_obs < K`` and there is no ridge every system is singular by
        construction and the minimum-norm solution is returned instead.

    Returns
    -------
    x : Array, shape (..., K)
        Solutions, exactly zero where the system was empty.

    Notes
    -----
    A row (or, in the G step, a pixel) whose weights are all zero gives
    ``M == 0`` exactly. That happens for a spectrum that is NaN throughout, for
    a pixel masked in every spectrum, and for the padding rows that
    :mod:`robusta_hmf.sharded` appends. ``jnp.linalg.solve`` returns NaN there,
    and because the solve is batched a single such row poisons every other
    row's result. Substituting the identity returns a zero factor for that row
    instead, which is exact: such a row enters no downstream sum with non-zero
    weight.

    With a ridge the matrix is never exactly zero and the guard never fires.
    That is correct rather than a gap -- ``ridge * I`` is already non-singular,
    and with ``b == 0`` it gives the same zero solution.

    With ``K > n_obs`` (more components than the fixed factor has rows) the
    weighted least-squares problem is underdetermined and ``solve`` is not
    defined; what it returns varies across jax versions. This path instead
    returns the minimum-norm solution, matching ``lstsq`` on the whitened
    problem. ``b`` lies in the range of ``M``, so the pseudo-inverse solution
    satisfies the normal equations exactly. The choice is made from static
    shapes, so the full-rank case pays nothing for it.
    """
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    if ridge is not None:
        M = M + ridge * eye
    elif n_obs is not None and n_obs < M.shape[-1]:
        return _min_norm_solve(M, b)
    empty = jnp.all(M == 0, axis=(-2, -1))
    M = jnp.where(empty[..., None, None], eye, M)
    b = jnp.where(empty[..., None], jnp.zeros_like(b), b)
    # b as an explicit column so this stays a batched matrix solve rather than
    # depending on solve()'s vector/matrix disambiguation rule.
    return jnp.linalg.solve(M, b[..., None])[..., 0]


def _min_norm_solve(M: Array, b: Array) -> Array:
    """Minimum-norm solution of batched symmetric PSD systems, via ``eigh``.

    Eigenvalues below ``K * eps`` times the largest are treated as zero, the
    same cutoff ``lstsq`` uses. An all-zero ``M`` has no eigenvalue above the
    cutoff and gives a zero solution, so the empty-row guard is not needed here.
    """
    evals, evecs = jnp.linalg.eigh(M)
    cutoff = M.shape[-1] * jnp.finfo(M.dtype).eps * jnp.max(evals, axis=-1, keepdims=True)
    keep = evals > cutoff
    inv = jnp.where(keep, 1 / jnp.where(keep, evals, 1), 0)
    coef = jnp.einsum("...kj,...k->...j", evecs, b)
    return jnp.einsum("...kj,...j->...k", evecs, inv * coef)


class WeightedAStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y: Array, W: Array, state: RHMFState) -> RHMFState:
        G = state.G
        M, b = normal_equations(W, W * Y, G)
        A_new = solve_normal_equations(M, b, self.ridge, n_obs=G.shape[0])  # [N, K]
        return update_state(state, A=A_new)


class WeightedGStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y: Array, W: Array, state: RHMFState) -> RHMFState:
        A = state.A
        M, b = normal_equations(W.T, (W * Y).T, A)
        G_new = solve_normal_equations(M, b, self.ridge, n_obs=A.shape[0])  # [D, K]
        return update_state(state, G=G_new)
