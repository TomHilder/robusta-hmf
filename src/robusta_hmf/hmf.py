# hmf.py

from typing import Literal, TypeAlias

import equinox as eqx
import optax
from jaxtyping import Array

from .als import WeightedAStep, WeightedGStep
from .likelihoods import GaussianLikelihood, Likelihood, StudentTLikelihood
from .rotations import Rotation, RotationMethod, get_rotation_cls
from .state import RHMFState, refresh_opt_state, update_state

OptMethod: TypeAlias = Literal["sgd", "als"]


class HMF(eqx.Module):
    # Common fields
    likelihood: Likelihood = eqx.field(static=True)
    rotation: Rotation = eqx.field(static=True)
    opt_method: OptMethod = eqx.field(static=True)

    # ALS-specific fields (None if using SGD)
    a_step: WeightedAStep | None
    g_step: WeightedGStep | None

    # SGD-specific fields (None if using ALS)
    opt: optax.GradientTransformation | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        method: OptMethod = "als",
        robust: bool = True,
        robust_scale: float = 1.0,
        robust_nu: float = 1.0,
        als_ridge: float | None = None,
        learning_rate: float = 1e-3,
        custom_opt: optax.GradientTransformation | None = None,
        rotation: RotationMethod = "fast",
        **rotation_kwargs,
    ):
        """
        Unified Heteroscedastic Matrix Factorization class. Not really meant for users but live your best life I guess.

        Args:
            method: Optimization method, either "als" or "sgd"
            robust: If True, use Student-t likelihood; if False, use Gaussian
            robust_scale: Scale parameter for Student-t likelihood (only used if robust=True)
            robust_nu: Degrees of freedom for Student-t likelihood (only used if robust=True); set to 1 for Cauchy distribution aka Hogg's version
            als_ridge: Ridge parameter for ALS updates (only used if method="als")
            learning_rate: Learning rate for SGD (only used if method="sgd" and custom_opt is None)
            rotation: Rotation method to use
            custom_opt: Custom optimizer for SGD (only used if method="sgd")
            **rotation_kwargs: Additional arguments for rotation method
        """
        # Configure likelihood based on robust flag
        if robust:
            self.likelihood = StudentTLikelihood(nu=robust_nu, scale=robust_scale)
        else:
            self.likelihood = GaussianLikelihood()

        # Set optimization method
        self.opt_method = method

        # Configure rotation
        self.rotation = get_rotation_cls(method=rotation)(**rotation_kwargs)

        # Configure method-specific components
        if method == "als":
            self.a_step = WeightedAStep(ridge=als_ridge)
            self.g_step = WeightedGStep(ridge=als_ridge)
            self.opt = None
        elif method == "sgd":
            self.a_step = None
            self.g_step = None
            if custom_opt is not None:
                self.opt = custom_opt
            else:
                self.opt = optax.adafactor(
                    factored=True,
                    decay_rate=0.9,
                    learning_rate=learning_rate,
                )
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'als' or 'sgd'.")

    def __check_init__(self):
        """Validate parameters."""
        if isinstance(self.likelihood, StudentTLikelihood):
            if self.likelihood.scale <= 0:
                raise ValueError(
                    f"robust_scale must be positive, got {self.likelihood.scale}"
                )
            if self.likelihood.nu <= 0:
                raise ValueError(
                    f"robust_nu must be positive, got {self.likelihood.nu}"
                )

        if self.opt_method == "als":
            if self.a_step.ridge is not None and self.a_step.ridge < 0:
                raise ValueError(
                    f"als_ridge must be non-negative or None, got {self.a_step.ridge}"
                )

    @eqx.filter_jit
    def step_als(
        self,
        Y: Array,
        W_data: Array,
        state: RHMFState,
        rotate: bool = True,
    ) -> tuple[RHMFState, float]:
        """Perform one ALS optimization step. Not intended to be called by user."""
        # W step
        W = self.likelihood.weights_total(Y, W_data, state.A, state.G)
        # ALS steps
        state = self.a_step(Y, W, state)
        state = self.g_step(Y, W, state)
        # Optional rotation step (Actually should never be skipped for ALS, but this is handled in OptFrame, and kept here for interface consistency)
        if rotate:
            state = self.rotation(state)
        # Compute loss, update states
        loss = self.likelihood.loss(Y, W_data, state.A, state.G)
        state = eqx.tree_at(lambda s: s.it, state, state.it + 1)
        return state, loss

    @eqx.filter_jit
    def step_sgd(
        self,
        Y: Array,
        W_data: Array,
        state: RHMFState,
        rotate: bool = True,
    ) -> tuple[RHMFState, float]:
        """Perform one SGD optimization step. Not intended to be called by user."""

        # Define loss function
        def loss_fn(params, Y):
            A, G = params
            return self.likelihood.loss(Y, W_data, A, G)

        # Perform SGD step
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

    def get_stepper(self):
        """Return the appropriate step function based on opt_method. Not intended to be called by user."""
        if self.opt_method == "als":
            return self.step_als
        elif self.opt_method == "sgd":
            return self.step_sgd
        else:
            raise ValueError(f"Unknown opt_method: {self.opt_method}")
