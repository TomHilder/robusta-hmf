# solvers.py

"""Step solvers shared by every basis."""

from jaxtyping import Array

from ..als import _min_norm_solve


def generic_solve(design: Array, other: Array, y: Array, w: Array) -> Array:
    """Minimum-norm weighted least squares for one axis's coefficients, densely.

    The model at measurement ``j`` is ``Σₗₖ design[j, l] · c[l, k] · other[j, k]``,
    linear in ``c``. This builds the full ``(n_coeffs·K)²`` normal equations
    and solves them with an eigenvalue cut-off (`als._min_norm_solve`), so a
    singular system -- a basis function no measurement touches, or a
    combination of functions that vanishes on every measurement -- returns the
    minimum-norm minimiser instead of failing.

    Parameters
    ----------
    design : Array, shape (n_points, n_coeffs)
        This axis's basis functions evaluated at each measurement.
    other : Array, shape (n_points, K)
        The other axis's functions evaluated at each measurement.
    y, w : Array, shape (n_points,)
        Values and total weights.

    Returns
    -------
    coeffs : Array, shape (n_coeffs, K)
    """
    n_points, n_coeffs = design.shape
    K = other.shape[1]
    X = (design[:, :, None] * other[:, None, :]).reshape(n_points, n_coeffs * K)
    M = X.T @ (w[:, None] * X)
    b = X.T @ (w * y)
    return _min_norm_solve(M, b).reshape(n_coeffs, K)
