# als.py

import equinox as eqx
import jax.numpy as jnp

from .state import RHMFState, update_state


def _normal_equations(W, WY, F):
    """Batched weighted normal equations for a fixed factor F.

    Returns (M, b) with

        M[i] = F.T @ diag(W[i]) @ F        shape (batch, K, K)
        b[i] = F.T @ (W[i] * Y[i])         shape (batch, K)

    Written as two matmuls against the K*K outer products of F rather than
    whitening F per row. Whitening is the obvious formulation, but under vmap
    it gives every one of the `batch` rows its own (len(F), K) copy of F, so
    the peak intermediate is batch * len(F) * K. For the full Gaia RVS sample
    that is 43 GiB at K=5 and 258 GiB at K=30 -- larger than any single GPU.
    Here the largest intermediate is batch * K * K instead (3.3 GiB at K=30),
    for the same flop count.

    It also shards cleanly: with W and Y split along `batch`, both matmuls are
    row-local, so a data-parallel split needs no cross-device communication in
    the A step and a single all-reduce of the small (K, K) blocks in the G step.
    """
    FF = (F[:, :, None] * F[:, None, :]).reshape(F.shape[0], -1)  # (len(F), K*K)
    M = (W @ FF).reshape(W.shape[0], F.shape[1], F.shape[1])      # (batch, K, K)
    b = WY @ F                                                     # (batch, K)
    return M, b


def _solve(M, b, ridge, dtype):
    if ridge is not None:
        M = M + ridge * jnp.eye(M.shape[-1], dtype=dtype)
    # b as an explicit column so this stays a batched matrix solve rather than
    # depending on solve()'s vector/matrix disambiguation rule.
    return jnp.linalg.solve(M, b[..., None])[..., 0]


class WeightedAStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y, W, state: RHMFState):
        G = state.G
        M, b = _normal_equations(W, W * Y, G)
        A_new = _solve(M, b, self.ridge, G.dtype)  # [N, K]
        return update_state(state, A=A_new)


class WeightedGStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y, W, state: RHMFState):
        A = state.A
        M, b = _normal_equations(W.T, (W * Y).T, A)
        G_new = _solve(M, b, self.ridge, A.dtype)  # [D, K]
        return update_state(state, G=G_new)
