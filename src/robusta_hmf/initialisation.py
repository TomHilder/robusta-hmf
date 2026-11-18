# initialisation.py

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, get_args

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .state import RHMFState

InitStrategy: TypeAlias = Literal["random", "svd", "custom"]


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
    """Seeds ``(A, G)`` before the optimisation loop starts.

    Pass one to [`Robusta`][robusta_hmf.Robusta] as ``override_initialiser``.
    ``N``, ``M`` and ``K`` may be left unset, in which case
    [`Robusta.fit()`][robusta_hmf.Robusta.fit] fills them in from the data,
    so an initialiser can be built without knowing the shapes up front.

    Parameters
    ----------
    N, M, K : int | None, default=None
        Rows and columns of ``Y``, and the rank.
    strategy : {"svd", "random", "custom"}, default="svd"
        ``"svd"``: ``A = U S^½`` and ``G = V S^½`` from the leading ``K``
        singular values and vectors of ``Y`` (the zero-filled data, without
        the weights). ``"random"``: independent standard normals.
        ``"custom"``: the ``A`` and ``G`` passed to [`execute()`][robusta_hmf.Initialiser.execute].
    """

    N: int | None = None
    M: int | None = None
    K: int | None = None
    strategy: InitStrategy = "svd"

    def __post_init__(self):
        if self.strategy not in get_args(InitStrategy):
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be one of {get_args(InitStrategy)}."
            )

    def with_dims(self, N: int, M: int, K: int) -> "Initialiser":
        """Fill in whichever of ``N``, ``M``, ``K`` are unset. Explicit values win."""
        missing = {
            name: value
            for name, value in (("N", N), ("M", M), ("K", K))
            if getattr(self, name) is None
        }
        return replace(self, **missing) if missing else self

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
        """Build the initial state.

        Parameters
        ----------
        seed : int | None, default=None
            Random seed, required for ``strategy="random"``.
        A, G : Array | None, default=None
            Initial factors, shapes ``(N, K)`` and ``(M, K)``, required for
            ``strategy="custom"``.
        Y : Array | None, default=None
            The ``(N, M)`` data, required for ``strategy="svd"``.
        opt : optax.GradientTransformation | None, default=None
            If given, its state is initialised on ``(A, G)`` (for SGD).

        Returns
        -------
        state : RHMFState
            The initial factors, with ``it = 0``.
        """
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
