# rhmf.py

import equinox as eqx
import jax
import optax

from .als import WeightedAStep, WeightedGStep
from .likelihoods import Likelihood
from .regularisers import Regulariser
from .rotations import Rotation
from .state import RHMFState


class ALS_RHMF(eqx.Module):
    likelihood: Likelihood
    a_step: WeightedAStep
    g_step: WeightedGStep
    rotation: Rotation
    regulariser: Regulariser | None = eqx.field(default=None)

    def __post_init__(self):
        if self.regulariser is not None:
            raise NotImplementedError("Regularisers not implemented yet.")

    def init_state(self, N, M, K, key):
        pass

    def random_init(self, N, M, K, key):
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (N, K))
        G = jax.random.normal(k2, (M, K))
        return RHMFState(A=A, G=G, it=0)

    def custom_init(self, A, G):
        return RHMFState(A=A, G=G, it=0)

    @eqx.filter_jit
    def step(self, Y, W_data, state: RHMFState):
        W = self.likelihood.weights_total(Y, W_data, state.A, state.G)
        state = self.a_step(Y, W, state)
        state = self.g_step(Y, W, state)
        state = self.rotation(state)
        loss = self.likelihood.loss(Y, W_data, state.A, state.G)
        state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
        return state, loss


class SGD_RHMF(eqx.Module):
    likelihood: Likelihood
    opt: optax.GradientTransformation
    regulariser: Regulariser | None = eqx.field(default=None)

    def __post_init__(self):
        if self.regulariser is not None:
            raise NotImplementedError("Regularisers not implemented in solve.")

    def init_state(self, N, M, K, key):
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (N, K))
        G = jax.random.normal(k2, (M, K))
        opt_state = self.opt.init((A, G))
        return RHMFState(A=A, G=G, it=0, opt_state=opt_state)

    def custom_init(self, A, G):
        opt_state = self.opt.init((A, G))
        return RHMFState(A=A, G=G, it=0, opt_state=opt_state)

    @eqx.filter_jit
    def step(self, Y, W_data, state: RHMFState):
        def loss_fn(params, Y):
            A, G = params
            return self.likelihood.loss(Y, W_data, A, G)

        params = (state.A, state.G)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, Y)
        updates, opt_state = self.opt.update(grads, state.opt_state, params)
        A_new, G_new = optax.apply_updates(params, updates)

        state = eqx.tree_at(lambda s: s.A, state, A_new)
        state = eqx.tree_at(lambda s: s.G, state, G_new)
        state = eqx.tree_at(lambda s: s.opt_state, state, opt_state)
        state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
        return state, loss
