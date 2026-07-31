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


def solve_normal_equations(M: Array, b: Array, ridge: float | None = None) -> Array:
    """Batched solve of the ALS normal equations, tolerant of empty rows.

    Parameters
    ----------
    M : Array, shape (..., K, K)
        Batched normal-equation matrices.
    b : Array, shape (..., K)
        Batched right-hand sides.
    ridge : float | None, default=None
        Optional ridge added to the diagonal before solving.

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
    """
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    if ridge is not None:
        M = M + ridge * eye
    empty = jnp.all(M == 0, axis=(-2, -1))
    M = jnp.where(empty[..., None, None], eye, M)
    b = jnp.where(empty[..., None], jnp.zeros_like(b), b)
    # b as an explicit column so this stays a batched matrix solve rather than
    # depending on solve()'s vector/matrix disambiguation rule.
    return jnp.linalg.solve(M, b[..., None])[..., 0]


class WeightedAStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y: Array, W: Array, state: RHMFState) -> RHMFState:
        G = state.G
        M, b = normal_equations(W, W * Y, G)
        A_new = solve_normal_equations(M, b, self.ridge)  # [N, K]
        return update_state(state, A=A_new)


class WeightedGStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y: Array, W: Array, state: RHMFState) -> RHMFState:
        A = state.A
        M, b = normal_equations(W.T, (W * Y).T, A)
        G_new = solve_normal_equations(M, b, self.ridge)  # [D, K]
        return update_state(state, G=G_new)
