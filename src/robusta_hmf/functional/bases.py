# bases.py

"""Bases for the row and column functions of `FunctionalRobusta`.

A basis is the only place that knows what its functions are. The engine
talks to it through `Basis.evaluate`, `Basis.adjoint`, `Basis.gram` and
`Basis.solve`; a basis family with structure to exploit overrides `solve`
with a fast path and keeps the generic solver as its reference.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..als import solve_normal_equations
from .solvers import generic_solve


class Basis(eqx.Module):
    """A finite set of functions on one axis (rows or columns).

    Coefficients are stored as ``(n_coeffs, K)`` arrays: column ``k`` holds
    the expansion of the ``k``-th function on this axis.
    """

    @property
    @abc.abstractmethod
    def n_coeffs(self) -> int:
        """Number of basis functions."""

    @abc.abstractmethod
    def evaluate(self, coeffs: Array, points: Array) -> Array:
        """Values of the expanded functions at ``points``.

        Maps coefficients ``(n_coeffs, K)`` to values ``(n_points, K)``. Linear
        in ``coeffs``.
        """

    @abc.abstractmethod
    def adjoint(self, values: Array, points: Array) -> Array:
        """Transpose of `evaluate`: ``(n_points, K)`` to ``(n_coeffs, K)``."""

    @abc.abstractmethod
    def gram(self) -> Array | None:
        """Inner products of the basis functions over the data domain.

        ``(n_coeffs, n_coeffs)``, or None when it is the identity.
        """

    def design(self, points: Array) -> Array:
        """The dense design matrix ``(n_points, n_coeffs)``, from `evaluate`."""
        eye = jnp.eye(self.n_coeffs, dtype=_float_dtype(points))
        return self.evaluate(eye, points)

    def check_points(self, points) -> None:
        """Raise ``ValueError`` if ``points`` are not valid coordinates for this basis."""

    def solve(self, points: Array, other: Array, y: Array, w: Array, other_n: int) -> Array:
        """Weighted least-squares coefficients on this axis, the other axis fixed.

        Minimises ``Σⱼ wⱼ (yⱼ − Σₖ fₖ(pⱼ) oⱼₖ)²`` over the coefficients of the
        functions ``fₖ`` on this axis, where ``other`` holds the values
        ``oⱼₖ`` of the other axis's functions at each measurement.

        Parameters
        ----------
        points : Array, shape (n_points, ...)
            Coordinates of the measurements on this axis.
        other : Array, shape (n_points, K)
            The other axis's functions evaluated at each measurement.
        y, w : Array, shape (n_points,)
            Values and total weights.
        other_n : int
            Number of basis functions on the other axis, for bases whose
            systems are singular by construction when it is below ``K``.

        Returns
        -------
        coeffs : Array, shape (n_coeffs, K)
        """
        return generic_solve(self.design(points), other, y, w)


def _float_dtype(points):
    dtype = jnp.asarray(points).dtype
    return dtype if jnp.issubdtype(dtype, jnp.floating) else jnp.zeros(()).dtype


class Discrete(Basis):
    """One free coefficient per category: ``n`` functions, each 1 on one category.

    The coordinates on this axis are integer category ids in ``[0, n)``. With
    `Discrete` bases on both axes the model is the matrix factorisation
    ``Y ≈ A Gᵀ`` of [`Robusta`][robusta_hmf.Robusta].

    Parameters
    ----------
    n : int
        Number of categories.
    fast : bool, default=True
        Solve each category's ``K × K`` system separately. With False the
        generic dense solver is used, which builds a design matrix with one
        column per category: for testing on small problems only.
    """

    n: int = eqx.field(static=True)
    fast: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        if int(self.n) < 1:
            raise ValueError(f"Discrete basis needs n >= 1, got {self.n}.")

    @property
    def n_coeffs(self) -> int:
        return self.n

    def evaluate(self, coeffs: Array, points: Array) -> Array:
        return coeffs[points]

    def adjoint(self, values: Array, points: Array) -> Array:
        return jax.ops.segment_sum(values, points, num_segments=self.n)

    def gram(self) -> None:
        return None

    def check_points(self, points) -> None:
        points = jnp.asarray(points)
        if points.ndim != 1 or not jnp.issubdtype(points.dtype, jnp.integer):
            raise ValueError(
                "A Discrete axis needs 1-D integer category ids, got "
                f"dtype {points.dtype} and shape {points.shape}."
            )
        if points.size and (int(points.min()) < 0 or int(points.max()) >= self.n):
            raise ValueError(
                f"Category ids must lie in [0, {self.n}), got range "
                f"[{int(points.min())}, {int(points.max())}]."
            )

    def solve(self, points: Array, other: Array, y: Array, w: Array, other_n: int) -> Array:
        if not self.fast:
            return super().solve(points, other, y, w, other_n)
        K = other.shape[1]
        outer = (other[:, :, None] * other[:, None, :]).reshape(other.shape[0], K * K)
        M = self.adjoint(w[:, None] * outer, points).reshape(self.n, K, K)
        b = self.adjoint((w * y)[:, None] * other, points)
        return solve_normal_equations(M, b, None, n_obs=other_n)
