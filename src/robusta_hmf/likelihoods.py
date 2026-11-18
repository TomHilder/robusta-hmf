# likelihoods.py

import abc

import equinox as eqx
import jax.numpy as jnp


class Likelihood(eqx.Module):
    """IRLS interface.

    The residual-level methods do the work and accept residuals of any shape
    (a dense ``(N, M)`` matrix or a flat list of measurements):

    - ``weights_from_residuals(R, W_data)``: robust factor, same shape as ``R``
    - ``loss_from_residuals(R, W_data)``: robust loss (scalar)

    The matrix-level methods apply them to ``R = Y - A @ G.T``:

    - ``weights_irls(Y, W_data, A, G)``: robust factor (same shape as Y)
    - ``weights_total(Y, W_data, A, G)``: ``W_data * weights_irls``
    - ``loss(Y, W_data, A, G)``: robust loss (scalar)
    """

    @abc.abstractmethod
    def weights_from_residuals(self, R, W_data): ...

    @abc.abstractmethod
    def loss_from_residuals(self, R, W_data): ...

    def weights_irls(self, Y, W_data, A, G):
        return self.weights_from_residuals(Y - A @ G.T, W_data)

    def weights_total(self, Y, W_data, A, G):
        return W_data * self.weights_irls(Y, W_data, A, G)

    def loss(self, Y, W_data, A, G):
        return self.loss_from_residuals(Y - A @ G.T, W_data)


class GaussianLikelihood(Likelihood):
    def weights_from_residuals(self, R, W_data):
        return jnp.ones_like(R)

    def loss_from_residuals(self, R, W_data):
        return jnp.sum(W_data * R**2)


class CauchyLikelihood(Likelihood):
    scale: float = 1.0

    def weights_from_residuals(self, R, W_data):
        s2 = jnp.asarray(self.scale, dtype=R.dtype) ** 2
        return s2 / (s2 + W_data * R**2)

    def loss_from_residuals(self, R, W_data):
        s2 = jnp.asarray(self.scale, dtype=R.dtype) ** 2
        return s2 * jnp.sum(jnp.log1p((W_data * R**2) / s2))


class StudentTLikelihood(Likelihood):
    nu: float = 3.0
    scale: float = 1.0

    def weights_from_residuals(self, R, W_data):
        s2 = jnp.asarray(self.scale, dtype=R.dtype) ** 2
        nu = jnp.asarray(self.nu, dtype=R.dtype)
        return (nu * s2) / (nu * s2 + W_data * R**2)

    def loss_from_residuals(self, R, W_data):
        s2 = jnp.asarray(self.scale, dtype=R.dtype) ** 2
        nu = jnp.asarray(self.nu, dtype=R.dtype)
        return (nu * s2) * jnp.sum(jnp.log1p((W_data * R**2) / (nu * s2)))
