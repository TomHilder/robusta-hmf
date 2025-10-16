# hmf.py

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .als import WeightedAStep, WeightedGStep
from .likelihoods import GaussianLikelihood, Likelihood
from .rotations import FastAffine, Identity, Rotation, RotationMethod, get_rotation_cls
from .state import RHMFState, refresh_opt_state, update_state


class ALS_HMF(eqx.Module):
    likelihood: Likelihood
    a_step: WeightedAStep
    g_step: WeightedGStep
    rotation: Rotation

    def __init__(
        self,
        als_ridge: float | None = None,
        rotation: RotationMethod = "fast",
        **rotation_kwargs,
    ):
        self.likelihood = GaussianLikelihood()
        self.a_step = WeightedAStep(ridge=als_ridge)
        self.g_step = WeightedGStep(ridge=als_ridge)
        self.rotation = get_rotation_cls(method=rotation)(**rotation_kwargs)

    @eqx.filter_jit
    def step(
        self,
        Y: Array,
        W_data: Array,
        state: RHMFState,
        rotate: bool = True,
    ) -> tuple[RHMFState, float]:
        # W step (in this case trivial)
        W = self.likelihood.weights_total(Y, W_data, state.A, state.G)
        # ALS steps
        state = self.a_step(Y, W, state)
        state = self.g_step(Y, W, state)
        # Optional rotation step
        if rotate:
            state = self.rotation(state)
        # Compute loss, update states
        loss = self.likelihood.loss(Y, W_data, state.A, state.G)
        state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
        return state, loss


class SGD_HMF(eqx.Module):
    likelihood: Likelihood = eqx.field(static=True)
    opt: optax.GradientTransformation = eqx.field(static=True)
    rotation: Rotation = eqx.field(static=True)

    def __init__(
        self,
        learning_rate: float = 1e-3,
        rotation: RotationMethod = "fast",
        **rotation_kwargs,
    ):
        self.likelihood = GaussianLikelihood()
        self.opt = optax.adafactor(
            factored=True,
            decay_rate=0.9,
            learning_rate=learning_rate,
        )
        # self.opt = optax.lamb(
        # learning_rate=learning_rate,
        # )
        self.rotation = get_rotation_cls(method=rotation)(**rotation_kwargs)

    @eqx.filter_jit
    def step(
        self,
        Y: Array,
        W_data: Array,
        state: RHMFState,
        rotate: bool = True,
    ) -> tuple[RHMFState, float]:
        # Define loss function
        def loss_fn(params, Y):
            A, G = params
            return self.likelihood.loss(Y, W_data, A, G)

        # Perform SGD step equivalent to W, A, G steps
        params = (state.A, state.G)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, Y)
        updates, opt_state = self.opt.update(grads, state.opt_state, params)
        A_new, G_new = optax.apply_updates(params, updates)
        # Apply updates and optionally rotate which also re-initialises optimiser state
        if rotate:
            state = update_state(state, A=A_new, G=G_new)
            state = self.rotation(state)  # rotates A/G
            state = refresh_opt_state(state, self.opt)  # refresh
        else:
            state = update_state(
                state,
                A=A_new,
                G=G_new,
                opt_state=opt_state,
            )
        # # Recompute loss
        # loss = self.likelihood.loss(Y, W_data, state.A, state.G)

        state = update_state(state, it=state.it + 1)
        return state, loss
