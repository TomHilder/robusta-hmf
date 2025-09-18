# als.py

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import RHMFState


class WeightedAStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y, state: RHMFState, W):
        G = state.G

        def solve_row(y_i, w_i):
            s = jnp.sqrt(w_i)
            Gw = G * s[:, None]
            M = Gw.T @ Gw
            if self.ridge is not None:
                M = M + self.ridge * jnp.eye(G.shape[1], dtype=G.dtype)
            b = Gw.T @ (s * y_i)
            return jnp.linalg.solve(M, b)

        A_new = jax.vmap(solve_row)(Y, W)  # [N, K]
        return eqx.tree_at(lambda s: s.A, state, A_new)


class WeightedGStep(eqx.Module):
    ridge: float | None = eqx.field(static=True, default=None)

    def __call__(self, Y, state: RHMFState, W):
        A = state.A

        def solve_col(y_j, w_j):
            s = jnp.sqrt(w_j)
            Aw = A * s[:, None]
            M = Aw.T @ Aw
            if self.ridge is not None:
                M = M + self.ridge * jnp.eye(A.shape[1], dtype=A.dtype)
            b = Aw.T @ (s * y_j)
            return jnp.linalg.solve(M, b)

        G_new = jax.vmap(solve_col)(Y.T, W.T)  # [D, K]
        return eqx.tree_at(lambda s: s.G, state, G_new)
