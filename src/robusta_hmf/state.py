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
