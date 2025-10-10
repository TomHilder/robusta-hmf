# likelihoods.py

import abc

import equinox as eqx
import jax.numpy as jnp


class Likelihood(eqx.Module):
    """IRLS interface:
    - weights_irls(Y,A,G,W_data): latent robust factor (same shape as Y)
    - weights_total(Y,A,G,W_data): W_data * weights_irls
    - loss(Y,A,G,W_data): robust loss (scalar)
    """

    @abc.abstractmethod
    def weights_irls(self, Y, W_data, A, G): ...

    def weights_total(self, Y, W_data, A, G):
        return W_data * self.weights_irls(Y, W_data, A, G)

    @abc.abstractmethod
    def loss(self, Y, W_data, A, G): ...


class GaussianLikelihood(Likelihood):
    def weights_irls(self, Y, W_data, A, G):
        return jnp.ones_like(Y)

    def loss(self, Y, W_data, A, G):
        r2 = (Y - A @ G.T) ** 2
        return jnp.sum(W_data * r2)


class CauchyLikelihood(Likelihood):
    scale: float = 1.0

    def weights_irls(self, Y, W_data, A, G):
        r2 = (Y - A @ G.T) ** 2
        s2 = jnp.asarray(self.scale, dtype=Y.dtype) ** 2
        return s2 / (s2 + W_data * r2)

    def loss(self, Y, W_data, A, G):
        r2 = (Y - A @ G.T) ** 2
        s2 = jnp.asarray(self.scale, dtype=Y.dtype) ** 2
        return s2 * jnp.sum(jnp.log1p((W_data * r2) / s2))


class StudentTLikelihood(Likelihood):
    nu: float = 3.0
    scale: float = 1.0

    def weights_irls(self, Y, W_data, A, G):
        r2 = (Y - A @ G.T) ** 2
        s2 = jnp.asarray(self.scale, dtype=Y.dtype) ** 2
        nu = jnp.asarray(self.nu, dtype=Y.dtype)
        return (nu * s2) / (nu * s2 + W_data * r2)

    def loss(self, Y, W_data, A, G):
        r2 = (Y - A @ G.T) ** 2
        s2 = jnp.asarray(self.scale, dtype=Y.dtype) ** 2
        nu = jnp.asarray(self.nu, dtype=Y.dtype)
        return (nu * s2) * jnp.sum(jnp.log1p((W_data * r2) / (nu * s2)))
