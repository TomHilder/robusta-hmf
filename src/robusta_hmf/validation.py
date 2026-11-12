# validation.py

"""Checks on ``Y`` and ``W`` before they reach an optimisation loop.

Non-finite entries are treated as missing (``Y = W = 0`` there) with one
:class:`~robusta_hmf.warnings.DataWarning`, or rejected when ``nan_error`` is
set. Negative weights are always rejected. The rank must not exceed the
number of rows or columns.
"""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .warnings import DataWarning


@dataclass
class BadEntries:
    """Running tally of non-finite and negative-weight entries over row blocks."""

    n_nonfinite: int = 0
    first_nonfinite: tuple[int, int] | None = None
    n_negative: int = 0
    first_negative: tuple[int, int] | None = None

    def add_block(self, Y, W, row_offset: int = 0) -> None:
        """Count the bad entries of one row block starting at ``row_offset``."""
        xp = jnp if isinstance(Y, jax.Array) or isinstance(W, jax.Array) else np
        nonfinite = ~(xp.isfinite(Y) & xp.isfinite(W))
        negative = ~nonfinite & (W < 0)
        self.n_nonfinite, self.first_nonfinite = _tally(
            nonfinite, self.n_nonfinite, self.first_nonfinite, row_offset
        )
        self.n_negative, self.first_negative = _tally(
            negative, self.n_negative, self.first_negative, row_offset
        )

    def resolve(self, nan_error: bool) -> None:
        """Raise or warn for what was counted; see :func:`sanitise`."""
        if self.n_nonfinite and nan_error:
            raise ValueError(
                f"Found {self.n_nonfinite:,} non-finite entries in Y or W "
                f"({_where(self.first_nonfinite)}). Set Y = W = 0 there to mark them "
                "missing, or pass nan_error=False to have that done automatically."
            )
        if self.n_negative:
            raise ValueError(
                f"Found {self.n_negative:,} negative entries in W "
                f"({_where(self.first_negative)}). W holds inverse variances, which "
                "cannot be negative; use 0 for missing data."
            )
        if self.n_nonfinite:
            warnings.warn(
                f"Found {self.n_nonfinite:,} non-finite entries in Y or W "
                f"({_where(self.first_nonfinite)}); treated them as missing "
                "(Y = W = 0). Pass nan_error=True to raise instead.",
                DataWarning,
            )


def _tally(mask, count: int, first, row_offset: int):
    n = int(mask.sum())
    if n and first is None:
        r, c = np.unravel_index(int(mask.argmax()), mask.shape)
        first = (int(r) + row_offset, int(c))
    return count + n, first


def _where(pos: tuple[int, int] | None) -> str:
    return "position unknown" if pos is None else f"first at row {pos[0]}, column {pos[1]}"


def zero_nonfinite(Y, W):
    """``(Y, W)`` with both set to 0 wherever either is non-finite."""
    xp = jnp if isinstance(Y, jax.Array) or isinstance(W, jax.Array) else np
    bad = ~(xp.isfinite(Y) & xp.isfinite(W))
    return xp.where(bad, 0, Y).astype(Y.dtype), xp.where(bad, 0, W).astype(W.dtype)


def check_same_shape(Y, W) -> None:
    """Raise unless ``Y`` and ``W`` are 2-D arrays of the same shape."""
    if np.ndim(Y) != 2:
        raise ValueError(f"Y must be 2-D, got shape {np.shape(Y)}")
    if np.shape(W) != np.shape(Y):
        raise ValueError(f"Y and W must have the same shape, got {np.shape(Y)} and {np.shape(W)}")


def sanitise(Y, W, nan_error: bool):
    """Validate a dense ``(Y, W)`` pair and return it with non-finite entries zeroed.

    With ``nan_error=False``, every entry where ``Y`` or ``W`` is NaN or ±inf
    becomes ``Y = W = 0`` (missing), with one ``DataWarning``. With
    ``nan_error=True`` such entries raise ``ValueError``. Negative weights
    always raise ``ValueError``.
    """
    check_same_shape(Y, W)
    bad = BadEntries()
    bad.add_block(Y, W)
    bad.resolve(nan_error)
    if bad.n_nonfinite:
        Y, W = zero_nonfinite(Y, W)
    return Y, W


def check_rank(rank: int, n_rows: int, n_cols: int, *, allow_few_rows: bool = False) -> None:
    """Raise if ``rank`` exceeds the number of columns, or of rows unless allowed.

    ``allow_few_rows`` is for inference, where each row of ``A`` is solved
    against the columns alone, so a single row at any rank ≤ ``n_cols`` is fine.
    """
    too_few_rows = n_rows < rank and not allow_few_rows
    if too_few_rows or n_cols < rank:
        which = "rows" if too_few_rows else "columns"
        raise ValueError(
            f"rank={rank} is larger than the number of {which} of Y (shape "
            f"{(n_rows, n_cols)}), so the factors cannot be determined. This is "
            "usually a shape mistake, e.g. Y passed transposed; otherwise lower the rank."
        )
