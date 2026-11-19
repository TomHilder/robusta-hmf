# bases.py

"""Bases for the row and column functions of `FunctionalRobusta`.

A basis is the only place that knows what its functions are. Its interface
follows FINUFFT's plan structure:

- the constructor fixes everything that does not depend on the data (as
  ``makeplan``);
- `Basis.bind` takes the measurement coordinates once per fit and returns a
  `BoundBasis` holding whatever can be precomputed for them (as ``setpts``);
- every iteration only applies the bound operator: `BoundBasis.evaluate`,
  `BoundBasis.adjoint` and `BoundBasis.solve` (as ``execute``).

The engine sees nothing else. A basis family with structure to exploit gives
its bound operator a fast `BoundBasis.solve`; the generic dense solver stays as
its reference.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..als import _min_norm_solve, solve_normal_equations
from .solvers import generic_solve


#: Measurements per chunk when accumulating per-category ``K × K`` sums, which bounds the
#: ``chunk · K²`` intermediate (2¹⁸ · 12² doubles is 300 MB).
CHUNK = 2**18


def segment_gram(other: Array, y: Array, w: Array, segments: Array, n_segments: int):
    """Per-segment ``Σ wⱼ oⱼ oⱼᵀ`` and ``Σ wⱼ yⱼ oⱼ``, accumulated in chunks of measurements.

    Returns ``M`` of shape ``(n_segments, K, K)`` and ``b`` of shape ``(n_segments, K)``.
    With at most `CHUNK` measurements this is a single segment sum; beyond that the
    measurements are processed `CHUNK` at a time so the ``(chunk, K, K)`` intermediate
    stays bounded.
    """
    n, K = other.shape

    def partial(o, yy, ww, seg):
        outer = (ww[:, None, None] * o[:, :, None] * o[:, None, :]).reshape(o.shape[0], K * K)
        M = jax.ops.segment_sum(outer, seg, num_segments=n_segments).reshape(n_segments, K, K)
        b = jax.ops.segment_sum((ww * yy)[:, None] * o, seg, num_segments=n_segments)
        return M, b

    if n <= CHUNK:
        return partial(other, y, w, segments)
    pad = (-n) % CHUNK
    n_chunks = (n + pad) // CHUNK

    def chunked(x):
        x = jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:], x.dtype)])
        return x.reshape((n_chunks, CHUNK) + x.shape[1:])

    xs = (chunked(other), chunked(y), chunked(w), chunked(segments))  # padded w = 0

    def body(carry, chunk):
        M, b = partial(*chunk)
        return (carry[0] + M, carry[1] + b), None

    init = (jnp.zeros((n_segments, K, K), other.dtype), jnp.zeros((n_segments, K), other.dtype))
    (M, b), _ = jax.lax.scan(body, init, xs)
    return M, b


class BoundBasis(eqx.Module):
    """A basis bound to a set of measurement coordinates.

    Coefficients are ``(n_coeffs, K)`` arrays: column ``k`` holds the
    expansion of the ``k``-th function. Values are ``(n_points, K)``.
    """

    @property
    @abc.abstractmethod
    def n_coeffs(self) -> int:
        """Number of basis functions."""

    @abc.abstractmethod
    def evaluate(self, coeffs: Array) -> Array:
        """Values of the expanded functions at the bound points. Linear in ``coeffs``."""

    @abc.abstractmethod
    def adjoint(self, values: Array) -> Array:
        """Transpose of `evaluate`: ``(n_points, K)`` to ``(n_coeffs, K)``."""

    def design(self) -> Array:
        """The dense design matrix ``(n_points, n_coeffs)``, from `evaluate`."""
        return self.evaluate(jnp.eye(self.n_coeffs, dtype=jnp.zeros(()).dtype))

    def design_rows(self, start: int, stop: int) -> Array:
        """Rows ``start:stop`` of the design matrix, without building all of it if avoidable."""
        return self.design()[start:stop]

    def solve(self, other: Array, y: Array, w: Array, other_n: int) -> Array:
        """Weighted least-squares coefficients on this axis, the other axis fixed.

        Minimises ``Σⱼ wⱼ (yⱼ − Σₖ fₖ(pⱼ) oⱼₖ)²`` over the coefficients of the
        functions ``fₖ`` on this axis, where ``other`` holds the values
        ``oⱼₖ`` of the other axis's functions at each measurement.

        Parameters
        ----------
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
        return generic_solve(self.design(), other, y, w)


class DenseBound(BoundBasis):
    """A bound basis held as its dense design matrix ``Φ``."""

    Phi: Array

    @property
    def n_coeffs(self) -> int:
        return int(self.Phi.shape[1])

    def evaluate(self, coeffs: Array) -> Array:
        return self.Phi @ coeffs

    def adjoint(self, values: Array) -> Array:
        return self.Phi.T @ values

    def design(self) -> Array:
        return self.Phi


class IndexedBound(BoundBasis):
    """A bound basis whose measurements share a limited set of coordinates.

    ``Phi`` holds the basis at the ``T`` distinct coordinates and ``index`` maps each
    measurement to one of them. Evaluation is a ``(T, K)`` product then a gather, and the
    least-squares step needs only per-coordinate ``K × K`` sums, so the cost of the normal
    equations no longer scales as ``n_points · (n_coeffs · K)²``.
    """

    Phi: Array
    index: Array

    @property
    def n_coeffs(self) -> int:
        return int(self.Phi.shape[1])

    def evaluate(self, coeffs: Array) -> Array:
        return (self.Phi @ coeffs)[self.index]

    def adjoint(self, values: Array) -> Array:
        summed = jax.ops.segment_sum(values, self.index, num_segments=self.Phi.shape[0])
        return self.Phi.T @ summed

    def design(self) -> Array:
        return self.Phi[self.index]

    def design_rows(self, start: int, stop: int) -> Array:
        return self.Phi[self.index[start:stop]]

    def solve(self, other: Array, y: Array, w: Array, other_n: int) -> Array:
        """The generic minimum-norm solve, with the normal equations built per coordinate.

        ``M[(b,k),(b′,k′)] = Σₜ Φₜ_b Φₜ_b′ Cₜ[k,k′]`` with ``Cₜ = Σ_{j at t} wⱼ oⱼ oⱼᵀ``: the
        same system `generic_solve` builds, assembled from ``T`` coordinates instead of
        every measurement.
        """
        T, B = self.Phi.shape
        K = other.shape[1]
        C, c = segment_gram(other, y, w, self.index, T)
        M = jnp.einsum("tb,tc,tkl->bkcl", self.Phi, self.Phi, C).reshape(B * K, B * K)
        rhs = jnp.einsum("tb,tk->bk", self.Phi, c).reshape(B * K)
        return _min_norm_solve(M, rhs).reshape(B, K)


class Basis(eqx.Module):
    """A finite set of functions on one axis (rows or columns)."""

    @property
    @abc.abstractmethod
    def n_coeffs(self) -> int:
        """Number of basis functions."""

    @abc.abstractmethod
    def bind(self, points: Array) -> BoundBasis:
        """Precompute for the coordinates ``points``; called once per fit."""

    @abc.abstractmethod
    def gram(self) -> Array | None:
        """Inner products of the basis functions over the data domain.

        ``(n_coeffs, n_coeffs)``, or None when it is the identity.
        """

    def domain_values(self, coeffs: Array) -> Array:
        """The expanded functions sampled over the data domain, for comparing two states.

        Convergence tests look at these rather than at raw coefficients, which
        may hold directions the data do not determine.
        """
        return coeffs

    def check_points(self, points) -> None:
        """Raise ``ValueError`` if ``points`` are not valid coordinates for this basis."""


class DiscreteBound(BoundBasis):
    """`Discrete` bound to a set of category ids."""

    points: Array
    n: int = eqx.field(static=True)
    fast: bool = eqx.field(static=True, default=True)

    @property
    def n_coeffs(self) -> int:
        return self.n

    def evaluate(self, coeffs: Array) -> Array:
        return coeffs[self.points]

    def adjoint(self, values: Array) -> Array:
        return jax.ops.segment_sum(values, self.points, num_segments=self.n)

    def solve(self, other: Array, y: Array, w: Array, other_n: int) -> Array:
        if not self.fast:
            return super().solve(other, y, w, other_n)
        M, b = segment_gram(other, y, w, self.points, self.n)
        return solve_normal_equations(M, b, None, n_obs=other_n)


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

    def bind(self, points: Array) -> DiscreteBound:
        return DiscreteBound(jnp.asarray(points), self.n, self.fast)

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


class Fourier(Basis):
    """Real Fourier series on a padded period around a 1-D data domain.

    With ``domain = (lo, hi)``, span ``s = hi − lo`` and period
    ``L = s · (1 + pad)``, the ``2·n_freq + 1`` basis functions are, in order,

        1,  cos(m θ) for m = 1..n_freq,  sin(m θ) for m = 1..n_freq,
        θ(x) = 2π (x − lo) / L.

    The data occupy the fraction ``1 / (1 + pad)`` of the period, so the
    functions do not wrap one end of the domain onto the other. The functions
    are defined only on the domain; coordinates outside it are rejected.

    Relation to FINUFFT's complex modes: with real coefficients ``a₀``,
    ``a_m`` (cosines) and ``b_m`` (sines), ``g(x) = Σ_m c_m e^{i m θ}`` with
    ``c₀ = a₀``, ``c_m = (a_m − i b_m) / 2`` and ``c₋ₘ = conj(c_m)``; FINUFFT's
    coordinate is ``θ − π ∈ [−π, π)``, which multiplies ``c_m`` by ``(−1)^m``.

    Parameters
    ----------
    n_freq : int
        Highest harmonic of the period.
    domain : (float, float)
        The range ``(lo, hi)`` the coordinates cover.
    pad : float, default=0.25
        Gap between the end of the data and its periodic image, as a fraction
        of the span.
    """

    n_freq: int = eqx.field(static=True)
    domain: tuple[float, float] = eqx.field(static=True)
    pad: float = eqx.field(static=True, default=0.25)

    def __check_init__(self):
        lo, hi = self.domain
        if not hi > lo:
            raise ValueError(f"Fourier domain must have hi > lo, got {self.domain}.")
        if int(self.n_freq) < 0:
            raise ValueError(f"n_freq must be >= 0, got {self.n_freq}.")
        if not self.pad >= 0:
            raise ValueError(f"pad must be >= 0, got {self.pad}.")

    @property
    def span(self) -> float:
        return float(self.domain[1] - self.domain[0])

    @property
    def period(self) -> float:
        return self.span * (1.0 + self.pad)

    @property
    def n_coeffs(self) -> int:
        return 2 * self.n_freq + 1

    def features(self, points: Array) -> Array:
        """The design matrix ``(n_points, n_coeffs)`` at ``points``."""
        x = jnp.asarray(points)
        x = x.astype(x.dtype if jnp.issubdtype(x.dtype, jnp.floating) else jnp.zeros(()).dtype)
        theta = (2 * jnp.pi / self.period) * (x - self.domain[0])
        m = jnp.arange(1, self.n_freq + 1, dtype=x.dtype)
        mt = theta[:, None] * m[None, :]
        return jnp.concatenate([jnp.ones_like(theta)[:, None], jnp.cos(mt), jnp.sin(mt)], axis=1)

    def bind(self, points: Array) -> DenseBound | IndexedBound:
        """Dense design, or `IndexedBound` when the coordinates repeat (≤ ¼ distinct)."""
        values, index = np.unique(np.asarray(points), return_inverse=True)
        if 4 * values.size <= index.size:
            return IndexedBound(self.features(jnp.asarray(values)), jnp.asarray(index.ravel()))
        return DenseBound(self.features(points))

    def gram(self) -> Array:
        """``∫[domain] H_m H_m′ dx`` in closed form (product-to-sum identities)."""
        F, s = self.n_freq, self.span
        omega = 2 * np.pi / self.period

        def ic(k):  # ∫₀ˢ cos(k ω u) du
            k = np.abs(k)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(k == 0, s, np.sin(k * omega * s) / (k * omega))

        def is_(k):  # ∫₀ˢ sin(k ω u) du, odd in k
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(k == 0, 0.0, (1 - np.cos(k * omega * s)) / (k * omega))

        c = np.arange(F + 1)  # cosine harmonics, 0 is the constant
        n = np.arange(1, F + 1)  # sine harmonics
        cc = 0.5 * (ic(c[:, None] - c[None, :]) + ic(c[:, None] + c[None, :]))
        ss = 0.5 * (ic(n[:, None] - n[None, :]) - ic(n[:, None] + n[None, :]))
        cs = 0.5 * (is_(c[:, None] + n[None, :]) + is_(n[None, :] - c[:, None]))
        S = np.block([[cc, cs], [cs.T, ss]])
        return jnp.asarray(S)

    def domain_values(self, coeffs: Array) -> Array:
        grid = jnp.linspace(self.domain[0], self.domain[1], 2 * self.n_coeffs, dtype=coeffs.dtype)
        return self.features(grid) @ coeffs

    def check_points(self, points) -> None:
        x = np.asarray(points)
        if x.ndim != 1:
            raise ValueError(f"A Fourier axis needs 1-D coordinates, got shape {x.shape}.")
        lo, hi = self.domain
        tol = 1e-9 * self.span
        if x.size and (x.min() < lo - tol or x.max() > hi + tol):
            raise ValueError(
                f"Coordinates must lie in the Fourier domain [{lo}, {hi}], got range "
                f"[{x.min()}, {x.max()}]."
            )
