# state.py

import equinox as eqx
import jax
from jaxtyping import Array


class FunctionalState(eqx.Module):
    """Coefficients of the row and column functions, plus the iteration count.

    The model at a measurement with coordinates ``(r, c)`` is
    ``Σₖ a_k(r) g_k(c)``, with ``a_k`` expanded in the row basis and ``g_k``
    in the column basis. With `Discrete` bases on both axes ``row_coeffs`` and
    ``col_coeffs`` are [`Robusta`][robusta_hmf.Robusta]'s ``A`` and ``G``.

    Attributes
    ----------
    row_coeffs : Array, shape (n_row_coeffs, K)
        Row-basis coefficients of the ``K`` row functions.
    col_coeffs : Array, shape (n_col_coeffs, K)
        Column-basis coefficients of the ``K`` column functions.
    it : int
        Number of iterations run to reach this state.
    """

    row_coeffs: Array = eqx.field(converter=jax.numpy.asarray)
    col_coeffs: Array = eqx.field(converter=jax.numpy.asarray)
    it: int = eqx.field(converter=jax.numpy.asarray, default=0)
