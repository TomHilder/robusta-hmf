# rotations.py

import abc
from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from .state import RHMFState, update_state

RotationMethod = Literal["fast", "slow", "fast-weighted", "identity"]


class Rotation(eqx.Module):
    """Base class for rotations to deal with symmetries."""

    @abc.abstractmethod
    def __call__(self, state: RHMFState) -> RHMFState: ...


class Identity(Rotation):
    def __call__(self, state: RHMFState) -> RHMFState:
        return state


class FastAffine(Rotation):
    whiten: bool = eqx.field(static=True, default=True)
    eps: float = eqx.field(static=True, default=1e-6)

    def __call__(self, state: RHMFState) -> RHMFState:
        A = state.A
        K = A.shape[1]
        C = A.T @ A + self.eps * jnp.eye(K, dtype=A.dtype)
        evals, V = jnp.linalg.eigh(C)

        if self.whiten:
            invsqrt = 1.0 / jnp.sqrt(jnp.maximum(evals, self.eps))
            sqrtv = jnp.sqrt(jnp.maximum(evals, self.eps))
            R = V @ (invsqrt[:, None] * V.T)  # V Λ^{-1/2} V^T
            Rinverse = V @ (sqrtv[:, None] * V.T)
        else:
            R, Rinverse = V, V.T

        A_new = A @ R
        G_new = state.G @ Rinverse
        return update_state(state, A=A_new, G=G_new)


class SlowAffine(Rotation):
    eps: float = eqx.field(static=True, default=1e-6)

    def __call__(self, state: RHMFState) -> RHMFState:
        A = state.A
        G = state.G
        K = A.shape[1]
        C = A.T @ G + self.eps * jnp.eye(K, dtype=A.dtype)
        U, S, V = jnp.linalg.svd(C, full_matrices=False)
        A_new = (U[:, :K] * S[:K]).T
        G_new = V[:K, :]
        return update_state(state, A=A_new, G=G_new)


class FastWeightedAffine(Rotation):
    # TODO
    def __call__(self, state: RHMFState) -> RHMFState:
        raise NotImplementedError


def get_rotation_cls(method: RotationMethod) -> Rotation:
    # Returns the class not an instance
    if method == "fast":
        return FastAffine
    elif method == "slow":
        return SlowAffine
    elif method == "fast-weighted":
        return FastWeightedAffine
    elif method == "identity":
        return Identity
    else:
        raise ValueError(f"Unknown rotation method: {method}")
