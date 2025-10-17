# regularisers.py

import abc

import equinox as eqx
import jax.numpy as jnp

from .state import RHMFState

# NOTE: This stuff is really just a placeholder for now, it's utterly unimplemented and not integrated with any of the rest of the code.
#       But, in principle it's a nice-to-have. So here it is.


class Regulariser(eqx.Module):
    @abc.abstractmethod
    def __call__(self, state: RHMFState) -> float: ...


class L2Regulariser(Regulariser):
    weight: float = 1e-3

    def __call__(self, state: RHMFState) -> float:
        return self.weight * (jnp.sum(state.A**2) + jnp.sum(state.G**2))
