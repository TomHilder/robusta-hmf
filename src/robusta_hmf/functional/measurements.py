# measurements.py

"""Data for `FunctionalRobusta`: a flat list of measurements."""

import warnings

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..warnings import DataWarning, user_stacklevel


class Measurements(eqx.Module):
    """A flat list of measurements ``(row, col, y, w)``.

    Measurement ``j`` is the value ``y[j]`` with inverse-variance weight
    ``w[j]``, taken at row coordinate ``row[j]`` and column coordinate
    ``col[j]``. On an axis with a `Discrete` basis the coordinate is an
    integer category id (an object, a pixel); on a continuous axis it is a
    position (a wavelength, a sky position). Every row and column may have its
    own number of measurements.

    Attributes
    ----------
    row : Array, shape (n,) or (n, d)
        Row coordinate of each measurement.
    col : Array, shape (n,) or (n, d)
        Column coordinate of each measurement.
    y : Array, shape (n,)
        Measured values.
    w : Array, shape (n,)
        Inverse-variance weights; 0 marks a measurement as missing.
    shape : tuple of int, or None
        ``(N, M)`` when the measurements are the entries of a rectangular
        matrix in row-major order (see `from_matrix`), else None.
    """

    row: Array
    col: Array
    y: Array
    w: Array
    shape: tuple[int, int] | None = eqx.field(static=True, default=None)

    @classmethod
    def from_matrix(cls, Y, W) -> "Measurements":
        """All entries of a rectangular ``(N, M)`` data matrix, row-major.

        Entry ``(i, j)`` becomes a measurement with ``row = i`` and
        ``col = j``. Entries with zero weight are kept, so the measurements
        map back onto ``Y`` exactly.
        """
        Y = jnp.asarray(Y)
        W = jnp.asarray(W)
        if Y.ndim != 2 or W.shape != Y.shape:
            raise ValueError(
                f"Y and W must be 2-D with the same shape, got {Y.shape} and {W.shape}."
            )
        N, M = Y.shape
        rows, cols = jnp.meshgrid(jnp.arange(N), jnp.arange(M), indexing="ij")
        return cls(rows.ravel(), cols.ravel(), Y.ravel(), W.ravel(), shape=(N, M))

    def __check_init__(self):
        n = np.shape(self.y)[0] if np.ndim(self.y) == 1 else None
        if n is None:
            raise ValueError(f"y must be 1-D, got shape {np.shape(self.y)}.")
        for name in ("row", "col", "w"):
            got = np.shape(getattr(self, name))
            if not got or got[0] != n:
                raise ValueError(f"{name} has {got} entries, but y has {n}.")
        if self.shape is not None and self.shape[0] * self.shape[1] != n:
            raise ValueError(f"shape={self.shape} does not match {n} measurements.")

    @property
    def n(self) -> int:
        """Number of measurements."""
        return int(self.y.shape[0])

    def as_matrices(self) -> tuple[Array, Array]:
        """``(Y, W)`` as ``(N, M)`` matrices, for measurements made by `from_matrix`."""
        if self.shape is None:
            raise ValueError("These measurements are not a rectangular matrix.")
        return self.y.reshape(self.shape), self.w.reshape(self.shape)

    def cast(self, dtype) -> "Measurements":
        """These measurements with ``y``, ``w`` and float coordinates in ``dtype``."""

        def coord(x):
            x = jnp.asarray(x)
            return x if jnp.issubdtype(x.dtype, jnp.integer) else x.astype(dtype)

        with np.errstate(over="ignore"):
            return Measurements(
                coord(self.row),
                coord(self.col),
                jnp.asarray(self.y, dtype),
                jnp.asarray(self.w, dtype),
                shape=self.shape,
            )


def sanitise_measurements(data: Measurements, nan_error: bool) -> Measurements:
    """Validate ``data`` and return it with non-finite measurements marked missing.

    Measurements where ``y`` or ``w`` is NaN or ±inf become ``y = w = 0``, with
    one `DataWarning`, or raise ``ValueError`` when ``nan_error`` is set.
    Negative weights and non-finite coordinates always raise ``ValueError``.
    """
    for name in ("row", "col"):
        x = getattr(data, name)
        if not jnp.issubdtype(x.dtype, jnp.integer) and not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError(f"Measurements.{name} contains non-finite coordinates.")
    bad = ~(jnp.isfinite(data.y) & jnp.isfinite(data.w))
    n_bad = int(bad.sum())
    negative = ~bad & (data.w < 0)
    if int(negative.sum()):
        first = int(jnp.argmax(negative))
        raise ValueError(
            f"Found {int(negative.sum()):,} negative weights (first at measurement {first}). "
            "w holds inverse variances, which cannot be negative; use 0 for missing data."
        )
    if not n_bad:
        return data
    first = int(jnp.argmax(bad))
    if nan_error:
        raise ValueError(
            f"Found {n_bad:,} measurements with non-finite y or w (first at measurement "
            f"{first}). Set y = w = 0 there to mark them missing, or pass nan_error=False to "
            "have that done automatically."
        )
    warnings.warn(
        f"Found {n_bad:,} measurements with non-finite y or w (first at measurement {first}); "
        "treated them as missing (y = w = 0). Pass nan_error=True to raise instead.",
        DataWarning,
        stacklevel=user_stacklevel(),
    )
    zero = jnp.zeros((), data.y.dtype)
    return Measurements(
        data.row,
        data.col,
        jnp.where(bad, zero, data.y),
        jnp.where(bad, zero, data.w),
        shape=data.shape,
    )
