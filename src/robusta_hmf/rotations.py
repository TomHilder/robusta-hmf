# rotations.py

import abc
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from .state import RHMFState, update_state

RotationMethod: TypeAlias = Literal["fast", "slow", "fast-weighted", "identity"]


class Rotation(eqx.Module):
    """Base class for rotations to deal with symmetries."""

    @abc.abstractmethod
    def __call__(self, state: RHMFState) -> RHMFState: ...


class Identity(Rotation):
    def __call__(self, state: RHMFState) -> RHMFState:
        return state


class FastAffine(Rotation):
    target: Literal["A", "G", "none"] = eqx.field(static=True, default="G")
    whiten: bool = eqx.field(static=True, default=False)
    eps: float = eqx.field(static=True, default=1e-6)

    def __call__(self, state: RHMFState) -> RHMFState:
        A, G = state.A, state.G
        K = A.shape[1]
        I = jnp.eye(K, dtype=A.dtype)

        # Pick matrix for eigendecomposition
        if self.target == "A":
            X = A
        elif self.target == "G":
            X = G
        else:
            X = A

        # Compute symmetric covariance
        C = 0.5 * (X.T @ X + (X.T @ X).T) + self.eps * I
        evals, V = jnp.linalg.eigh(C)
        lam = jnp.maximum(evals, self.eps)

        if self.whiten:
            invsqrt = 1.0 / jnp.sqrt(lam)
            sqrtv = jnp.sqrt(lam)
            R = V @ (invsqrt[:, None] * V.T)  # V Λ^{-1/2} V^T
            Rinverse = V @ (sqrtv[:, None] * V.T)  # inverse of above
        else:
            R = V
            Rinverse = V

        # Apply transform depending on target
        if self.target == "A":
            A_new = A @ R
            G_new = G @ Rinverse
        elif self.target == "G":
            A_new = A @ Rinverse
            G_new = G @ R
        else:  # "none"
            A_new = A @ R
            G_new = G @ Rinverse

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
