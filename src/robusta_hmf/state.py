# state.py

import equinox as eqx
import jax
import optax
from jaxtyping import Array


class RHMFState(eqx.Module):
    A: Array = eqx.field(converter=jax.numpy.asarray)
    G: Array = eqx.field(converter=jax.numpy.asarray)
    it: int = eqx.field(default=0)
    opt_state: optax.OptState | None = eqx.field(default=None)


def update_state(
    state: RHMFState,
    A: Array | None = None,
    G: Array | None = None,
    it: int | None = None,
    opt_state: optax.OptState | None = None,
) -> RHMFState:
    A_ = state.A if A is None else A
    G_ = state.G if G is None else G
    it_ = state.it if it is None else it
    opt_ = state.opt_state if opt_state is None else opt_state
    return eqx.tree_at(
        lambda s: (s.A, s.G, s.it, s.opt_state), state, (A_, G_, it_, opt_)
    )


def refresh_opt_state(state: RHMFState, opt: optax.GradientTransformation) -> RHMFState:
    opt_state = opt.init((state.A, state.G))
    state = eqx.tree_at(lambda s: s.opt_state, state, opt_state)
    return state
