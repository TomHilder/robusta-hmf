# rhmf.py

import equinox as eqx
import optax
from jaxtyping import Array

from .als import WeightedAStep, WeightedGStep
from .likelihoods import Likelihood, StudentTLikelihood
from .opt_methods import OptMethod
from .rotations import Rotation, RotationMethod, get_rotation_cls
from .state import RHMFState, refresh_opt_state, update_state

# TODO: restructure to have a base HMF class and then subclasses for SGD and ALS?
# Then also subclass the same thing for RHMF classes too?


# NOTE: Tom suspects that since this is SO similar to ALS_HMF, we could probably reduce the repeated code
# significantly. The only real differences are the likelihood and the W step. For now, it's fine.
class ALS_RHMF(eqx.Module):
    likelihood: Likelihood
    a_step: WeightedAStep
    g_step: WeightedGStep
    rotation: Rotation
    opt_method: OptMethod = eqx.field(static=True, default="als")

    def __init__(
        self,
        robust_scale: float = 1.0,
        robust_nu: float = 1.0,
        als_ridge: float | None = None,
        rotation: RotationMethod = "fast",
        **rotation_kwargs,
    ):
        self.likelihood = StudentTLikelihood(nu=robust_nu, scale=robust_scale)
        self.a_step = WeightedAStep(ridge=als_ridge)
        self.g_step = WeightedGStep(ridge=als_ridge)
        self.rotation = get_rotation_cls(method=rotation)(**rotation_kwargs)
        self.opt_method = "als"

    @eqx.filter_jit
    def step(
        self,
        Y: Array,
        W_data: Array,
        state: RHMFState,
        rotate: bool = True,
    ) -> tuple[RHMFState, float]:
        # W step
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


# NOTE: Ditto above note but for similarity to SGD_HMF.
class SGD_RHMF(eqx.Module):
    likelihood: Likelihood = eqx.field(static=True)
    opt: optax.GradientTransformation = eqx.field(static=True)
    rotation: Rotation = eqx.field(static=True)
    opt_method: OptMethod = eqx.field(static=True, default="sgd")

    def __init__(
        self,
        robust_scale: float = 1.0,
        robust_nu: float = 1.0,
        learning_rate: float = 1e-3,
        rotation: RotationMethod = "fast",
        custom_opt: optax.GradientTransformation | None = None,
        **rotation_kwargs,
    ):
        """
        Note that the learning_rate is only used if custom_opt is None.
        """
        self.likelihood = StudentTLikelihood(nu=robust_nu, scale=robust_scale)
        if custom_opt is not None:
            self.opt = custom_opt
        else:
            self.opt = optax.adafactor(
                factored=True,
                decay_rate=0.9,
                learning_rate=learning_rate,
            )
        self.rotation = get_rotation_cls(method=rotation)(**rotation_kwargs)
        self.opt_method = "sgd"

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
            # Recalculate loss after rotation
            loss = self.likelihood.loss(Y, W_data, state.A, state.G)
        else:
            state = update_state(
                state,
                A=A_new,
                G=G_new,
                opt_state=opt_state,
            )

        state = update_state(state, it=state.it + 1)
        return state, loss
