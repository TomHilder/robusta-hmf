# initialisation.py

from dataclasses import dataclass
from typing import Literal, get_args

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .state import RHMFState

InitStrategy = Literal["random", "svd", "custom"]


def random_init(seed: int, N: int, M: int, K: int) -> tuple[Array]:
    if seed is None:
        raise ValueError("Must provide seed for random initialisation.")
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (N, K))
    G = jax.random.normal(k2, (M, K))
    return A, G


def svd_init(Y: Array, N: int, M: int, K: int) -> tuple[Array]:
    if Y is None:
        raise ValueError("Must provide Y for svd initialisation.")
    if Y.shape != (N, M):
        raise ValueError(
            f"Shape of provided Y: {Y.shape} does not match (N, M) = ({N}, {M})"
        )
    if K > min(N, M):
        raise ValueError(f"K={K} exceeds min(N, M)={min(N, M)} in SVD initialisation.")
    U, s, Vh = jnp.linalg.svd(Y, full_matrices=False)
    s = jnp.sqrt(s[:K])
    return U[:, :K] * s, Vh[:K, :].T * s


def custom_init(A: Array, G: Array, N: int, M: int, K: int) -> tuple[Array]:
    if A is None or G is None:
        raise ValueError("Must provide A and G for custom initialisation.")
    if A.shape[0] != N or A.shape[1] != K:
        raise ValueError(
            f"Shape of provided A: {A.shape} does not match (N, K) = ({N}, {K})"
        )
    if G.shape[0] != M or G.shape[1] != K:
        raise ValueError(
            f"Shape of provided G: {G.shape} does not match (M, K) = ({M}, {K})"
        )
    return A, G


@dataclass(frozen=True)
class Initialiser:
    N: int
    M: int
    K: int
    strategy: InitStrategy

    def __post_init__(self):
        if self.strategy not in get_args(InitStrategy):
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be one of {get_args(InitStrategy)}."
            )

    def execute(
        self,
        seed: int | None = None,
        A: Array | None = None,
        G: Array | None = None,
        Y: Array | None = None,
        opt: optax.GradientTransformation
        | tuple[optax.GradientTransformation, optax.GradientTransformation]
        | None = None,
    ) -> RHMFState:
        # Initialise A and G according to strategy
        if self.strategy == "random":
            A, G = random_init(seed, self.N, self.M, self.K)
        elif self.strategy == "svd":
            A, G = svd_init(Y, self.N, self.M, self.K)
        elif self.strategy == "custom":
            A, G = custom_init(A, G, self.N, self.M, self.K)

        # Initialise the optax state
        if opt is None:
            opt_state = None
        else:
            opt_state = opt.init((A, G))

        return RHMFState(A=A, G=G, it=0, opt_state=opt_state)
