# robusta.py

from dataclasses import dataclass

from jaxtyping import Array

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
        conv_strategy: str = "rel_frac_loss",
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

    def fit(
        self,
        Y: Array,
        W: Array,
        max_iter: int = 1000,
        rotation_cadence: int = 10,
        conv_check_cadence: int = 20,
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
        y_new: Array,
        w_new: Array,
        override_method: OptMethod | None = None,
        state: RHMFState | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> tuple[Array, Array, Array]:
        """
        Predict coefficients and reconstruction for new observation(s).
        """
        # TODO: Implement this method properly. Need to implement the one-sided fitting logic.
        raise NotImplementedError("infer() method not yet implemented.")

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

    # NOTE: LLM slop version for the infer() method above commented out below. Proper implementation is TODO.
    # def predict(
    #     self,
    #     y_new: Array,
    #     w_new: Array,
    #     override_method: OptMethod | None = None,
    #     state: RHMFState | None = None,
    #     max_iter: int = 100,
    #     tol: float = 1e-5,
    # ) -> tuple[Array, Array, Array]:
    #     """
    #     Predict coefficients and reconstruction for new observation(s).

    #     This method fixes the basis vectors G and optimizes only the coefficients a
    #     for the new data, using iterative reweighting for robust estimation.

    #     Parameters
    #     ----------
    #     y_new : Array, shape (M,) or (N_new, M)
    #         New observation(s) to predict for
    #     w_new : Array, shape (M,) or (N_new, M)
    #         Weights for new observation(s)
    #     override_method : OptMethod | None, default=None
    #         If None, use the method from training (self.method).
    #         If provided, override the inference method (e.g., 'als' for faster inference).
    #     state : RHMFState | None, default=None
    #         State to use for prediction. If None, use self._state from last fit.
    #     max_iter : int, default=100
    #         Maximum iterations for inference optimization
    #     tol : float, default=1e-5
    #         Convergence tolerance for inference (checks fractional change in a)

    #     Returns
    #     -------
    #     a_new : Array, shape (K,) or (N_new, K)
    #         Inferred coefficients
    #     y_pred : Array, shape (M,) or (N_new, M)
    #         Predicted reconstruction (a_new @ G.T)
    #     w_robust : Array, shape (M,) or (N_new, M)
    #         Robust weights (IRLS weights) for the new data
    #     """
    #     # Use provided state or internal state
    #     state = state if state is not None else self._state
    #     if state is None:
    #         raise ValueError("No trained state available. Call fit() first.")

    #     # Determine method to use
    #     method = override_method if override_method is not None else self.method

    #     # Get the basis vectors
    #     G = state.G  # Shape (M, K)

    #     # Handle single observation vs batch
    #     single_obs = y_new.ndim == 1
    #     if single_obs:
    #         y_new = y_new[jnp.newaxis, :]
    #         w_new = w_new[jnp.newaxis, :]

    #     N_new, M = y_new.shape

    #     # Initialize coefficients to zero
    #     a_new = jnp.zeros((N_new, self.rank))

    #     # Initialize robust weights to data weights
    #     w_robust = w_new.copy()

    #     # Iteratively solve for a_new with robust reweighting
    #     converged = False
    #     n_iter = 0

    #     while not converged and n_iter < max_iter:
    #         # Store old a for convergence check
    #         a_old = a_new.copy()

    #         # Weighted least squares for each observation (A-step on new data)
    #         if method == "als":
    #             # Use weighted least squares: a = (G^T W G)^-1 G^T W y
    #             for i in range(N_new):
    #                 # Get diagonal weight matrix for this observation
    #                 w_i = w_robust[i]
    #                 # Solve weighted least squares
    #                 # a_i = (G^T diag(w_i) G)^-1 G^T diag(w_i) y_i
    #                 WG = G.T * w_i  # (K, M) with column i scaled by w_i
    #                 GtWG = WG @ G  # (K, K)
    #                 GtWy = WG @ y_new[i]  # (K,)

    #                 # Add ridge if needed
    #                 if (
    #                     self._hmf.a_step is not None
    #                     and self._hmf.a_step.ridge is not None
    #                 ):
    #                     GtWG = GtWG + self._hmf.a_step.ridge * jnp.eye(self.rank)

    #                 a_new = a_new.at[i].set(jnp.linalg.solve(GtWG, GtWy))
    #         else:
    #             # For SGD, we'd do gradient descent on a
    #             # For simplicity, fall back to ALS for now
    #             # TODO: Implement proper SGD inference if needed
    #             raise NotImplementedError(
    #                 "SGD inference not yet implemented. Use override_method='als'."
    #             )

    #         # Compute residuals
    #         resid = y_new - a_new @ G.T

    #         # Update robust weights
    #         w_robust = self._hmf.likelihood.weights_total(
    #             Y=y_new,
    #             W_data=w_new,
    #             A=a_new.T,
    #             G=G,
    #         )

    #         # Check convergence: fractional change in a
    #         if jnp.max(jnp.abs(a_new - a_old)) / jnp.mean(jnp.abs(a_new)) < tol:
    #             converged = True

    #         n_iter += 1

    #     # Compute final prediction
    #     y_pred = a_new @ G.T

    #     # Return to original shape if single observation
    #     if single_obs:
    #         a_new = a_new[0]
    #         y_pred = y_pred[0]
    #         w_robust = w_robust[0]

    #     return a_new, y_pred, w_robust
