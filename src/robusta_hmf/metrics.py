# metrics.py

"""Held-out scoring for a fitted factorisation.

The headline number is :meth:`ZMoments.kl`, the paper's model-selection score
(Eq. 20). All of the moment-based metrics are derived from three sums over the
residuals, so they are accumulated once into a :class:`ZMoments` and read off
as properties rather than recomputed per metric.

Two accumulation routes produce the same :class:`ZMoments`:

- :func:`z_moments`, which forms the full ``(N, M)`` residual array; and
- :meth:`robusta_hmf.sharded.ShardedFrame.z_moments`, which accumulates the
  same three sums chunk by chunk and never materialises an ``(N, M)`` array.

:func:`mad_z` is deliberately not part of :class:`ZMoments`: a median needs the
whole ``z`` array, which is exactly what the chunked route exists to avoid.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .likelihoods import Likelihood


def kl_from_moments(mean_z: float, std_z: float) -> float:
    """KL divergence from N(mean_z, std_z^2) to N(0, 1). Target: 0.

    The paper's model-selection score (Eq. 20), under a Gaussian approximation
    to the empirical distribution of the standardised residuals:

    .. math::
        \\mathrm{KL} = -\\log\\sigma_z + (\\sigma_z^2 + \\mu_z^2) / 2 - 1/2

    which is zero only for :math:`\\mu_z = 0, \\sigma_z = 1` and positive
    otherwise. Lower is better.
    """
    if not std_z > 0:
        return float(np.inf)
    return float(-np.log(std_z) + (std_z**2 + mean_z**2) / 2.0 - 0.5)


def z_scores(Y: Array, W: Array, A: Array, G: Array, likelihood: Likelihood) -> Array:
    """Robustly-weighted standardised residuals.

    Parameters
    ----------
    Y : Array, shape (N, M)
        Data matrix.
    W : Array, shape (N, M)
        Data weights (inverse variance).
    A : Array, shape (N, K)
        Coefficients.
    G : Array, shape (M, K)
        Basis vectors.
    likelihood : Likelihood
        Supplies the IRLS downweighting.

    Returns
    -------
    z : Array, shape (N, M)
        ``z = (Y - A @ G.T) * sqrt(W * w_irls)``, which is unit normal under a
        calibrated model whose outliers are correctly downweighted.
    """
    r = Y - A @ G.T
    return r * jnp.sqrt(W * likelihood.weights_irls(Y, W, A, G))


@dataclass(frozen=True)
class ZMoments:
    """Sufficient statistics for the moment-based held-out scores.

    Attributes
    ----------
    n : int
        Number of residuals summed over, ``N * M``. Every pixel counts,
        including those masked to zero weight, matching the convention used
        throughout the paper's analysis scripts.
    sum_z, sum_z2 : float
        ``sum(z)`` and ``sum(z**2)``.
    sum_chi2 : float
        ``sum(W * r**2)``, the unrobustified chi-squared.
    """

    n: int
    sum_z: float
    sum_z2: float
    sum_chi2: float

    @property
    def mean_z(self) -> float:
        """Mean of the robustly-weighted residuals. Target: 0."""
        return self.sum_z / self.n

    @property
    def var_z(self) -> float:
        """Variance of the robustly-weighted residuals. Target: 1."""
        return max(self.sum_z2 / self.n - self.mean_z**2, 0.0)

    @property
    def std_z(self) -> float:
        """Standard deviation of the robustly-weighted residuals. Target: 1."""
        return float(np.sqrt(self.var_z))

    @property
    def chi2_red(self) -> float:
        """Reduced chi-squared, ``mean(W * r**2)``. Target: 1."""
        return self.sum_chi2 / self.n

    @property
    def rmse(self) -> float:
        """Weighted RMSE, ``sqrt(chi2_red)``. Lower is better."""
        return float(np.sqrt(self.chi2_red))

    @property
    def kl(self) -> float:
        """KL divergence of the z distribution from N(0, 1). Target: 0.

        The paper's model-selection score (Eq. 20). See
        :func:`kl_from_moments`. Lower is better.
        """
        return kl_from_moments(self.mean_z, self.std_z)

    def as_dict(self) -> dict:
        """The scores as a plain dict, for tabulating over a (K, Q) grid."""
        return {
            "kl": self.kl,
            "mean_z": self.mean_z,
            "std_z": self.std_z,
            "chi2_red": self.chi2_red,
            "rmse": self.rmse,
            "n": self.n,
        }


def moments_from_z(z: Array, W: Array, r: Array) -> ZMoments:
    """Accumulate a :class:`ZMoments` from an already-computed residual block.

    Used by both the dense and the chunked routes so there is one definition
    of what each sum means.
    """
    return ZMoments(
        n=int(z.size),
        sum_z=float(jnp.sum(z)),
        sum_z2=float(jnp.sum(z * z)),
        sum_chi2=float(jnp.sum(W * r * r)),
    )


def z_moments(Y: Array, W: Array, A: Array, G: Array, likelihood: Likelihood) -> ZMoments:
    """Held-out moments for ``(A, G)`` on ``(Y, W)``, forming the full residual.

    Parameters
    ----------
    Y, W : Array, shape (N, M)
        Data and weights.
    A : Array, shape (N, K)
        Coefficients, e.g. from :meth:`robusta_hmf.Robusta.infer`.
    G : Array, shape (M, K)
        Basis vectors.
    likelihood : Likelihood
        Supplies the IRLS downweighting.

    Returns
    -------
    moments : ZMoments

    See Also
    --------
    robusta_hmf.sharded.ShardedFrame.z_moments : chunked, out-of-core route.
    """
    r = Y - A @ G.T
    z = r * jnp.sqrt(W * likelihood.weights_irls(Y, W, A, G))
    return moments_from_z(z, W, r)


def mad_z(Y: Array, W: Array, A: Array, G: Array, likelihood: Likelihood) -> float:
    """Median absolute z-score. Target: 0.6745 (the normal MAD).

    Not available from :class:`ZMoments`: a median needs the whole ``(N, M)``
    ``z`` array, so this has no out-of-core equivalent.
    """
    return float(jnp.median(jnp.abs(z_scores(Y, W, A, G, likelihood))))
