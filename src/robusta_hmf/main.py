# robusta.py

import warnings
from copy import deepcopy
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from .als import WeightedAStep
from .convergence import ConvergenceTester
from .frame import OptFrame
from .hmf import HMF, OptMethod
from .initialisation import Initialiser
from .rotations import RotationMethod
from .state import RHMFState


@dataclass
class Robusta:
    """
    Unified interface for Robust Heteroscedastic Matrix Factorization.

    This class provides a scikit-learn-like API for training and inference
    with heteroscedastic matrix factorization, supporting both standard Gaussian
    and robust Student-t likelihoods, and both ALS and SGD optimization methods.
    """

    rank: int
    method: OptMethod

    _hmf: HMF
    _initialiser: Initialiser
    _conv_tester: ConvergenceTester
    _frame: OptFrame

    # Internal state from last fit
    _state: RHMFState | None = None
    _loss_history: Array | None = None

    def __init__(
        self,
        rank: int,
        method: OptMethod = "als",
        robust: bool = True,
        robust_nu: float = 1.0,
        robust_scale: float = 1.0,
        # Init params
        init_strategy: str = "svd",
        override_initialiser: Initialiser | None = None,
        # Convergence params
        conv_strategy: str = "max_frac_G",
        conv_tol: float = 1e-3,
        override_conv_tester: ConvergenceTester | None = None,
        # HMF params
        als_ridge: float | None = None,
        learning_rate: float = 1e-3,
        rotation: RotationMethod = "fast",
        **rotation_kwargs,
    ):
        """
        Initialize Robusta model.

        Parameters
        ----------
        rank : int
            Number of latent factors/basis vectors
        method : OptMethod, default="als"
            Optimization method, either "als" or "sgd"
        robust : bool, default=False
            If True, use Student-t likelihood; if False, use Gaussian
        robust_nu : float, default=1.0
            Degrees of freedom for Student-t likelihood (only used if robust=True)
        robust_scale : float, default=1.0
            Scale parameter for Student-t likelihood (only used if robust=True)
        init_strategy : str, default="svd"
            Initialization strategy for factors
        override_initialiser : Initialiser | None, default=None
            Custom initialiser object (overrides init_strategy if provided)
        conv_strategy : str, default="rel_frac_loss"
            Convergence detection strategy
        conv_tol : float, default=1e-3
            Convergence tolerance
        conv_tester : ConvergenceTester | None, default=None
            Custom convergence tester (overrides conv_strategy/conv_tol if provided)
        als_ridge : float | None, default=None
            Ridge parameter for ALS updates (only used if method="als")
        learning_rate : float, default=1e-3
            Learning rate for SGD (only used if method="sgd")
        rotation : RotationMethod, default="fast"
            Rotation method to use
        **rotation_kwargs
            Additional arguments for rotation method
        """
        self.rank = rank
        self.method = method

        # Build HMF
        self._hmf = HMF(
            method=method,
            robust=robust,
            robust_nu=robust_nu,
            robust_scale=robust_scale,
            als_ridge=als_ridge,
            learning_rate=learning_rate,
            rotation=rotation,
            **rotation_kwargs,
        )

        # Build or use provided initialiser
        if override_initialiser is None:
            # Note: N, M will be set when fit() is called
            # For now we create a partial initialiser
            self._init_strategy = init_strategy
            self._initialiser = None
        else:
            self._initialiser = override_initialiser
            self._init_strategy = override_initialiser.strategy

        # Build or use provided convergence tester
        if override_conv_tester is None:
            self._conv_tester = ConvergenceTester(strategy=conv_strategy, tol=conv_tol)
        else:
            self._conv_tester = override_conv_tester

        # Build optimization frame
        self._frame = OptFrame(method=self._hmf, conv_tester=self._conv_tester)

        # Initialize internal state
        self._state = None
        self._loss_history = None

    def set_state(self, state: RHMFState) -> "Robusta":
        """
        Returns a copy of the object with the given state.

        Parameters
        ----------
        state : RHMFState
            State to set

        Returns
        -------
        new_obj : Robusta
            New Robusta object with updated state
        """
        new_obj = deepcopy(self)
        new_obj._state = state
        return new_obj

    def fit(
        self,
        Y: Array,
        W: Array,
        max_iter: int = 1000,
        rotation_cadence: int = 1,
        conv_check_cadence: int = 10,
        seed: int = 0,
        init_state: RHMFState | None = None,
    ) -> tuple[RHMFState, Array]:
        """
        Fit the model to data.

        Parameters
        ----------
        Y : Array, shape (N, M)
            Data matrix to factorize
        W : Array, shape (N, M)
            Weight matrix (inverse variance)
        max_iter : int, default=1000
            Maximum number of optimization iterations
        rotation_cadence : int, default=10
            How often to apply rotation (set to 1 for ALS internally by OptFrame)
        conv_check_cadence : int, default=20
            How often to check convergence
        seed : int, default=0
            Random seed for initialization
        init_state : RHMFState | None, default=None
            Initial state to continue training from. If None, initialize from scratch.

        Returns
        -------
        state : RHMFState
            Final optimization state
        loss_history : Array
            Loss values over iterations
        """
        N, M = Y.shape

        # Give warning if rotation cadence is not 1 for ALS
        if self.method == "als" and rotation_cadence != 1:
            warnings.warn(
                "Using ALS with rotation_cadence != 1. This may lead to unexpected behavior."
            )

        # Initialize state if not provided
        if init_state is None:
            # Create initialiser if not provided
            if self._initialiser is None:
                self._initialiser = Initialiser(
                    N=N,
                    M=M,
                    K=self.rank,
                    strategy=self._init_strategy,
                )
            print("Initializing state... ", flush=True, end="")
            init_state = self._initialiser.execute(
                seed=seed,
                Y=Y,
                opt=self._hmf.opt if self.method == "sgd" else None,
            )
            print("done.", flush=True)

        # Run optimization
        final_state, loss_history = self._frame.run(
            Y=Y,
            W=W,
            init_state=init_state,
            rotation_cadence=rotation_cadence,
            conv_check_cadence=conv_check_cadence,
            max_iter=max_iter,
        )

        # Store internally
        self._state = final_state
        self._loss_history = loss_history

        return final_state, loss_history

    def synthesize(
        self,
        state: RHMFState | None = None,
        indices: Array | None = None,
    ) -> Array:
        """
        Synthesize data from the model: A @ G.T

        Parameters
        ----------
        state : RHMFState | None, default=None
            State to use. If None, use self._state from last fit.
        indices : Array | None, default=None
            If provided, only synthesize these rows (indices into A)

        Returns
        -------
        synthesis : Array, shape (N, M) or (len(indices), M)
            Synthesized data matrix
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")

        A = state.A if indices is None else state.A[indices]
        return A @ state.G.T

    def infer(
        self,
        Y_infer: Array,
        W_infer: Array,
        state: RHMFState | None = None,
        max_iter: int = 100,
        conv_strategy: str = "max_frac_A",
        conv_tol: float = 1e-3,
        conv_check_cadence: int = 10,
    ) -> tuple[RHMFState, Array]:
        """
        Predict coefficients and reconstruction for new observation(s). Always uses least-squares not gradient descent.
        """
        # NOTE: This implementation is a bit of a mess

        N, _ = Y_infer.shape
        A_dummy = jnp.zeros((N, self.rank))
        try:
            G_fixed = state.G if state is not None else self.G
        except AttributeError:
            raise ValueError(
                "No trained state available. Either provide state or call fit() first."
            )

        # Get an initial guess for the coefficients using the data weights
        a_step = WeightedAStep(ridge=self._hmf.a_step.ridge if self.method == "als" else None)
        A_init = a_step(Y_infer, W_infer, RHMFState(A=A_dummy, G=G_fixed, it=0)).A

        # We can't use conv_strategy of max_frac_G here because G is fixed
        if conv_strategy == "max_frac_G":
            warnings.warn("max_frac_G is not supported for infer(). Using max_frac_A instead.")
            conv_strategy = "max_frac_A"

        # Run one-sided optimization
        conv_tester = ConvergenceTester(strategy=conv_strategy, tol=conv_tol)
        frame = OptFrame(method=self._hmf, conv_tester=conv_tester)
        final_state, loss_history = frame.run(
            Y=Y_infer,
            W=W_infer,
            init_state=RHMFState(
                A=A_init,
                G=G_fixed,
                it=0,
            ),
            rotation_cadence=1,  # No rotation since G is fixed so this is a dummy value
            conv_check_cadence=conv_check_cadence,
            max_iter=max_iter,
            skip_G=True,
        )
        return final_state, loss_history  # Not stored because this is not the main fit

    def basis_vectors(self, state: RHMFState | None = None) -> Array:
        """
        Get the basis vectors (G matrix).

        Parameters
        ----------
        state : RHMFState | None, default=None
            State to use. If None, use self._state from last fit.

        Returns
        -------
        G : Array, shape (M, K)
            Basis vectors
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        return state.G

    def coefficients(self, state: RHMFState | None = None) -> Array:
        """
        Get the coefficients (A matrix).

        Parameters
        ----------
        state : RHMFState | None, default=None
            State to use. If None, use self._state from last fit.

        Returns
        -------
        A : Array, shape (N, K)
            Coefficients
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        return state.A

    def residuals(self, Y: Array, state: RHMFState | None = None) -> Array:
        """
        Compute residuals: Y - A @ G.T

        Parameters
        ----------
        Y : Array, shape (N, M)
            Data matrix
        state : RHMFState | None, default=None
            State to use. If None, use self._state from last fit.

        Returns
        -------
        residuals : Array, shape (N, M)
            Residuals
        """
        return Y - self.synthesize(state=state)

    def mse(self, Y: Array, state: RHMFState | None = None) -> float:
        """
        Compute mean squared error of the reconstruction.

        Parameters
        ----------
        Y : Array, shape (N, M)
            Data matrix
        state : RHMFState | None, default=None
            State to use. If None, use self._state from last fit.

        Returns
        -------
        mse : float
            Mean squared error
        """
        raise NotImplementedError(
            "MSE calculation not implemented yet since I don't know exactly what metric is appropriate and this is name MSE as a placeholder only."
        )

    def robust_weights(
        self,
        Y: Array,
        W: Array,
        state: RHMFState | None = None,
    ) -> Array:
        """
        Compute IRLS robust weights (between 0 and 1).

        Parameters
        ----------
        Y : Array, shape (N, M)
            Data matrix
        W : Array, shape (N, M)
            Weight matrix (inverse variance)
        state : RHMFState | None, default=None
            State to use. If None, use self._state from last fit.

        Returns
        -------
        weights : Array, shape (N, M)
            Robust weights
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")

        return self._hmf.likelihood.weights_irls(Y, W, state.A, state.G)

    # Convenience properties
    @property
    def A(self) -> Array | None:
        """Coefficients from last fit."""
        return self._state.A if self._state is not None else None

    @property
    def G(self) -> Array | None:
        """Basis vectors from last fit."""
        return self._state.G if self._state is not None else None

    @property
    def state(self) -> RHMFState | None:
        """Full state from last fit."""
        return self._state

    @property
    def loss_history(self) -> Array | None:
        """Loss history from last fit."""
        return self._loss_history
