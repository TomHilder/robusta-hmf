# hmf.py

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .als import WeightedAStep, WeightedGStep
from .likelihoods import GaussianLikelihood, Likelihood
from .rotations import FastAffine, Rotation, RotationMethod, get_rotation_cls
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
        self.opt = optax.adam(learning_rate)
        # self.opt = optax.adamw(learning_rate)
        # self.opt = optax.adafactor(
        # factored=True, decay_rate=0.9, learning_rate=learning_rate
        # )
        # self.opt = optax.sgd(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        # self.opt = optax.chain(
        #     optax.clip_by_global_norm(1.0),
        #     optax.adafactor(factored=True, decay_rate=0.9, learning_rate=learning_rate),
        #     # optax.adam(learning_rate),
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
        # Recompute loss
        loss = self.likelihood.loss(Y, W_data, state.A, state.G)

        state = update_state(state, it=state.it + 1)
        return state, loss


# class SGD_BlockHMF(eqx.Module):
#     likelihood: Likelihood = eqx.field(static=True)
#     opt: Tuple[optax.GradientTransformation, optax.GradientTransformation] = eqx.field(
#         static=True
#     )
#     rot_A: Rotation = eqx.field(static=True)  # whiten A before updating G
#     rot_G: Rotation = eqx.field(static=True)  # whiten G before updating A
#     kA: int = eqx.field(static=True)
#     kG: int = eqx.field(static=True)
#     reset_on_rotate: bool = eqx.field(static=True)

#     def __init__(
#         self,
#         learning_rate_A: float = 1e-3,
#         learning_rate_G: float = 1e-3,
#         kA: int = 3,
#         kG: int = 3,
#         whiten: bool = True,
#         rotation_eps: float = 1e-6,
#         reset_on_rotate: bool = False,  # True for Adam/SGD+mom; False for Adafactor
#         use_adafactor: bool = True,
#     ):
#         self.likelihood = GaussianLikelihood()

#         if use_adafactor:
#             opt_A = optax.chain(
#                 optax.clip_by_global_norm(1.0),
#                 optax.adafactor(
#                     factored=True, decay_rate=0.9, learning_rate=learning_rate_A
#                 ),
#             )
#             opt_G = optax.chain(
#                 optax.clip_by_global_norm(1.0),
#                 optax.adafactor(
#                     factored=True, decay_rate=0.9, learning_rate=learning_rate_G
#                 ),
#             )
#         else:
#             opt_A = optax.chain(
#                 optax.clip_by_global_norm(2.0),
#                 optax.sgd(learning_rate=learning_rate_A, momentum=0.9, nesterov=True),
#             )
#             opt_G = optax.chain(
#                 optax.clip_by_global_norm(2.0),
#                 optax.sgd(learning_rate=learning_rate_G, momentum=0.9, nesterov=True),
#             )

#         self.opt = (opt_A, opt_G)
#         self.rot_G = FastAffine(target="G", whiten=whiten, eps=rotation_eps)
#         self.rot_A = FastAffine(target="A", whiten=whiten, eps=rotation_eps)
#         self.kA = kA
#         self.kG = kG
#         self.reset_on_rotate = reset_on_rotate

#     # --- helpers ---
#     @staticmethod
#     def _mask_grads(grads, update_A: bool):
#         gA, gG = grads
#         zA = jax.tree_util.tree_map(jnp.zeros_like, gA)
#         zG = jax.tree_util.tree_map(jnp.zeros_like, gG)
#         return (gA, zG) if update_A else (zA, gG)

#     def init_state(self, A0: Array, G0: Array) -> RHMFState:
#         optA = self.opt[0].init((A0, G0))
#         optG = self.opt[1].init((A0, G0))
#         return RHMFState(A=A0, G=G0, it=0, opt_state=(optA, optG))

#     def _maybe_reset(self, state: RHMFState) -> RHMFState:
#         if not self.reset_on_rotate:
#             return state
#         return update_state(
#             state,
#             opt_state=(
#                 self.opt[0].init((state.A, state.G)),
#                 self.opt[1].init((state.A, state.G)),
#             ),
#         )

#     @eqx.filter_jit
#     def _step_A(self, Y: Array, W_data: Array, state: RHMFState) -> RHMFState:
#         def loss_fn(params):
#             A, G = params
#             return self.likelihood.loss(Y, W_data, A, G)

#         params = (state.A, state.G)
#         _, grads = eqx.filter_value_and_grad(loss_fn)(params)
#         grads = self._mask_grads(grads, update_A=True)
#         updates, optA = self.opt[0].update(grads, state.opt_state[0], params)
#         A_new, G_same = optax.apply_updates(params, updates)
#         return update_state(
#             state, A=A_new, G=G_same, opt_state=(optA, state.opt_state[1])
#         )

#     @eqx.filter_jit
#     def _step_G(self, Y: Array, W_data: Array, state: RHMFState) -> RHMFState:
#         def loss_fn(params):
#             A, G = params
#             return self.likelihood.loss(Y, W_data, A, G)

#         params = (state.A, state.G)
#         _, grads = eqx.filter_value_and_grad(loss_fn)(params)
#         grads = self._mask_grads(grads, update_A=False)
#         updates, optG = self.opt[1].update(grads, state.opt_state[1], params)
#         A_same, G_new = optax.apply_updates(params, updates)
#         return update_state(
#             state, A=A_same, G=G_new, opt_state=(state.opt_state[0], optG)
#         )

#     @eqx.filter_jit
#     def step(
#         self, Y: Array, W_data: Array, state: RHMFState
#     ) -> tuple[RHMFState, float]:
#         state = self.rot_G(state)  # whiten G, update A kA times
#         state = self._maybe_reset(state)
#         for _ in range(self.kA):
#             state = self._step_A(Y, W_data, state)

#         state = self.rot_A(state)  # whiten A, update G kG times
#         state = self._maybe_reset(state)
#         for _ in range(self.kG):
#             state = self._step_G(Y, W_data, state)

#         state = update_state(state, it=state.it + 1)
#         loss = self.likelihood.loss(Y, W_data, state.A, state.G)
#         return state, loss


class SGD_BlockHMF(eqx.Module):
    likelihood: Likelihood = eqx.field(static=True)
    opt: Tuple[optax.GradientTransformation, optax.GradientTransformation] = eqx.field(
        static=True
    )
    rot_A: Rotation = eqx.field(static=True)
    rot_G: Rotation = eqx.field(static=True)
    kA: int = eqx.field(static=True)
    kG: int = eqx.field(static=True)
    reset_on_rotate: bool = eqx.field(static=True)

    def __init__(
        self,
        learning_rate_A: float = 1e-3,
        learning_rate_G: float = 1e-3,
        kA: int = 3,
        kG: int = 3,
        whiten: bool = True,
        rotation_eps: float = 1e-6,
        reset_on_rotate: bool = False,
        use_adafactor: bool = True,
    ):
        self.likelihood = GaussianLikelihood()

        # -------------------- optimisers --------------------
        if use_adafactor:
            opt_A = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adafactor(
                    factored=True, decay_rate=0.9, learning_rate=learning_rate_A
                ),
            )
            opt_G = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adafactor(
                    factored=True, decay_rate=0.9, learning_rate=learning_rate_G
                ),
            )
        else:
            opt_A = optax.chain(
                optax.clip_by_global_norm(2.0),
                optax.sgd(learning_rate=learning_rate_A, momentum=0.9, nesterov=True),
            )
            opt_G = optax.chain(
                optax.clip_by_global_norm(2.0),
                optax.sgd(learning_rate=learning_rate_G, momentum=0.9, nesterov=True),
            )

        self.opt = (opt_A, opt_G)
        # -------------------- rotations ---------------------
        # NOTE: whiten argument now does nothing since it's hardcoded
        self.rot_G = FastAffine(target="G", whiten=True, eps=rotation_eps)
        self.rot_A = FastAffine(target="A", whiten=False, eps=rotation_eps)
        self.kA = kA
        self.kG = kG
        self.reset_on_rotate = reset_on_rotate

    # =====================================================================
    # internal helpers
    # =====================================================================
    @staticmethod
    def _mask_grads(grads, update_A: bool):
        gA, gG = grads
        zA = jax.tree_util.tree_map(jnp.zeros_like, gA)
        zG = jax.tree_util.tree_map(jnp.zeros_like, gG)
        return (gA, zG) if update_A else (zA, gG)

    def _maybe_reset(self, state: RHMFState) -> RHMFState:
        if not self.reset_on_rotate:
            return state
        optA = self.opt[0].init((state.A, state.G))
        optG = self.opt[1].init((state.A, state.G))
        return update_state(state, opt_state=(optA, optG))

    # -------------------- QR reorientation --------------------
    @staticmethod
    def _qr_reorient_A(state: RHMFState) -> RHMFState:
        Q, R = jnp.linalg.qr(state.A, mode="reduced")
        G_new = state.G @ R.T
        return update_state(state, A=Q, G=G_new)

    @staticmethod
    def _qr_reorient_G(state: RHMFState) -> RHMFState:
        Q, R = jnp.linalg.qr(state.G, mode="reduced")
        A_new = state.A @ R.T
        return update_state(state, A=A_new, G=Q)

    # =====================================================================
    # block updates
    # =====================================================================
    @eqx.filter_jit
    def _step_A(self, Y: Array, W_data: Array, state: RHMFState) -> RHMFState:
        def loss_fn(params):
            A, G = params
            return self.likelihood.loss(Y, W_data, A, G)

        params = (state.A, state.G)
        _, grads = eqx.filter_value_and_grad(loss_fn)(params)
        grads = self._mask_grads(grads, update_A=True)
        updates, optA = self.opt[0].update(grads, state.opt_state[0], params)
        A_new, G_same = optax.apply_updates(params, updates)
        state = update_state(
            state, A=A_new, G=G_same, opt_state=(optA, state.opt_state[1])
        )
        # micro orthonormalisation: keeps basis vectors unique
        state = self._qr_reorient_A(state)
        return state

    @eqx.filter_jit
    def _step_G(self, Y: Array, W_data: Array, state: RHMFState) -> RHMFState:
        def loss_fn(params):
            A, G = params
            return self.likelihood.loss(Y, W_data, A, G)

        params = (state.A, state.G)
        _, grads = eqx.filter_value_and_grad(loss_fn)(params)
        grads = self._mask_grads(grads, update_A=False)
        updates, optG = self.opt[1].update(grads, state.opt_state[1], params)
        A_same, G_new = optax.apply_updates(params, updates)
        state = update_state(
            state, A=A_same, G=G_new, opt_state=(state.opt_state[0], optG)
        )
        state = self._qr_reorient_G(state)
        return state

    # =====================================================================
    # public API
    # =====================================================================
    def init_state(self, A0: Array, G0: Array) -> RHMFState:
        optA = self.opt[0].init((A0, G0))
        optG = self.opt[1].init((A0, G0))
        return RHMFState(A=A0, G=G0, it=0, opt_state=(optA, optG))

    @eqx.filter_jit
    def step(
        self, Y: Array, W_data: Array, state: RHMFState
    ) -> tuple[RHMFState, float]:
        # --- A block ---
        state = self.rot_G(state)  # whiten G before updating A
        state = self._maybe_reset(state)
        for _ in range(self.kA):
            state = self._step_A(Y, W_data, state)

        # --- G block ---
        state = self.rot_A(state)  # whiten A before updating G
        state = self._maybe_reset(state)
        for _ in range(self.kG):
            state = self._step_G(Y, W_data, state)

        state = update_state(state, it=state.it + 1)
        loss = self.likelihood.loss(Y, W_data, state.A, state.G)
        return state, loss
