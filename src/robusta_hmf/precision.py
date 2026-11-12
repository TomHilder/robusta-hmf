# precision.py

"""Floating-point precision of a fit.

robusta-hmf computes in float64 unless told otherwise. JAX only provides
float64 when ``jax_enable_x64`` is on, and silently computes in float32 when it
is off, so a float64 fit with x64 disabled raises :class:`PrecisionError`
rather than quietly running in float32. float32 is supported, but only as an
explicit choice (``Robusta(..., dtype="float32")``): it is much faster on GPUs,
and it is also where the ALS solves become fragile -- rows or columns of ``W``
with fewer non-zero weights than the rank can make a float32 fit diverge.
"""

import jax
import numpy as np

SUPPORTED_DTYPES = ("float32", "float64")
DEFAULT_DTYPE = np.dtype("float64")


class PrecisionError(RuntimeError):
    """A fit asked for float64 but JAX's float64 support is disabled."""


def resolve_dtype(dtype) -> np.dtype:
    """Normalise ``dtype`` to ``np.dtype('float32')`` or ``np.dtype('float64')``.

    Accepts anything ``np.dtype`` does: a string, a numpy or JAX dtype, or a
    scalar type such as ``np.float32``.
    """
    try:
        resolved = np.dtype(dtype)
    except TypeError as err:
        raise ValueError(
            f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}."
        ) from err
    if resolved.name not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}.")
    return resolved


def check_precision(dtype: np.dtype, rank: int | None = None) -> None:
    """Raise :class:`PrecisionError` if ``dtype`` is float64 but x64 is off."""
    if np.dtype(dtype) != np.float64 or jax.config.jax_enable_x64:
        return
    rank_arg = f"rank={rank}, " if rank is not None else ""
    raise PrecisionError(
        "robusta-hmf computes in float64 by default, but JAX's float64 support "
        "is off (jax_enable_x64=False), so this fit would silently run in "
        "float32. float32 fits can diverge or return NaN, for example when "
        "rows or columns of W have fewer non-zero weights than the rank.\n"
        "Either enable float64 before creating any JAX arrays:\n"
        '    jax.config.update("jax_enable_x64", True)\n'
        "or opt in to float32 explicitly:\n"
        f'    Robusta({rank_arg}dtype="float32")'
    )


def nonfinite_loss_message(iteration: int, loss, dtype) -> str:
    """Error message for a loss that has gone NaN or infinite."""
    dtype_name = np.dtype(dtype).name if dtype is not None else "the input dtype"
    hint = (
        "This fit computes in float32: try float64 first "
        '(jax.config.update("jax_enable_x64", True) and the default dtype).'
        if dtype_name == "float32"
        else f"This fit computes in {dtype_name}."
    )
    return (
        f"Loss is not finite at iteration {iteration}: {loss}. {hint} Other common "
        "causes: rows or columns of W with fewer non-zero weights than the rank "
        "(an under-determined ALS solve), or values in Y or W large enough to overflow "
        f"{dtype_name}."
    )
