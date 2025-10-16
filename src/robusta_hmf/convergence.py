# convergence.py

from dataclasses import dataclass, field
from typing import Literal, get_args

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .state import RHMFState

jax.config.update("jax_enable_x64", True)

ConvStrategy = Literal["max_frac_G", "max_frac_A", "rel_frac_loss", "none"]

DEFAULT_STRATEGY = "max_frac_G"
DEFAULT_TOL = 1e-6


def max_frac_mat(old_mat: Array, new_mat: Array, tol: float) -> bool:
    d_mat = old_mat - new_mat
    if jnp.max(d_mat * d_mat) / jnp.mean(new_mat * new_mat) < tol:
        return True
    else:
        return False


def rel_frac_loss(old_loss: float, new_loss: float, tol: float) -> bool:
    if jnp.abs(old_loss - new_loss) / jnp.abs(new_loss) < tol:
        return True
    else:
        return False


def never_converged(*args, **kwargs) -> bool:
    return False


@dataclass(frozen=True)
class ConvergenceTester:
    strategy: ConvStrategy = field(default=DEFAULT_STRATEGY)
    tol: float = field(default=DEFAULT_TOL)

    def __post_init__(self):
        if self.strategy not in get_args(ConvStrategy):
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be one of {get_args(ConvStrategy)}."
            )

    def is_converged(
        self,
        old_state: RHMFState,
        new_state: RHMFState,
        old_loss: float,
        new_loss: float,
    ) -> bool:
        if self.strategy == "max_frac_G":
            return max_frac_mat(old_state.G, new_state.G, tol=self.tol)
        elif self.strategy == "max_frac_A":
            return max_frac_mat(old_state.A, new_state.A, tol=self.tol)
        elif self.strategy == "rel_frac_loss":
            return rel_frac_loss(old_loss, new_loss, tol=self.tol)
        elif self.strategy == "none":
            return never_converged()
