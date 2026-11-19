# model.py

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..convergence import ConvergenceTester, ConvStrategy, RunMonitor
from ..fit_info import FitInfo
from ..initialisation import random_init, svd_init
from ..likelihoods import GaussianLikelihood, Likelihood, StudentTLikelihood
from ..precision import DEFAULT_DTYPE, check_precision, resolve_dtype
from .bases import Basis, Discrete
from .engine import make_layout, run
from .measurements import Measurements, sanitise_measurements
from .state import FunctionalState


class FunctionalRobusta:
    """Robust heteroskedastic factorisation with a basis of functions on each axis.

    Models each measurement ``(r, c, y, w)`` as ``y ≈ Σₖ a_k(r) g_k(c)``, where
    the ``K`` row functions ``a_k`` are expanded in ``row_basis`` and the ``K``
    column functions ``g_k`` in ``col_basis``. The data are a flat list of
    measurements ([`Measurements`][robusta_hmf.functional.Measurements]), so
    rows need not share coordinates or have the same number of measurements.
    The fit is [`Robusta`][robusta_hmf.Robusta]'s: iteratively reweighted
    alternating least squares on a Student-t likelihood. With `Discrete` bases
    on both axes it is the same model as ``Robusta``.

    Parameters
    ----------
    rank : int
        Number of row and column functions, ``K``.
    row_basis, col_basis : Basis
        Bases for the row and column functions.
    robust : bool, default=True
        Student-t likelihood if True, Gaussian if False.
    robust_nu : float, default=1.0
        Student-t degrees of freedom.
    robust_scale : float, default=1.0
        Student-t scale, ``Q`` in the paper.
    init_strategy : {"svd", "random"}, default="svd"
        ``"svd"``: needs a `Discrete` row basis. With a `Discrete` column basis
        too, the leading singular vectors of the zero-filled data matrix, as
        ``Robusta``; otherwise those of each row's data projected onto the
        column basis, with the row coefficients fitted to them.
        ``"random"``: standard normal coefficients.
    conv_strategy : {"max_frac_G", "max_frac_A", "rel_frac_loss", "none"}, default="max_frac_G"
        As for ``Robusta``, with ``G`` the column and ``A`` the row coefficients.
    conv_tol : float, default=1e-3
        Convergence tolerance.
    dtype : {"float64", "float32"} or dtype, default="float64"
        Precision of the fit, with ``Robusta``'s rules.
    nan_error : bool, default=False
        Raise for non-finite ``y`` or ``w`` instead of treating them as missing.
    """

    def __init__(
        self,
        rank: int,
        row_basis: Basis,
        col_basis: Basis,
        robust: bool = True,
        robust_nu: float = 1.0,
        robust_scale: float = 1.0,
        init_strategy: Literal["svd", "random"] = "svd",
        conv_strategy: ConvStrategy = "max_frac_G",
        conv_tol: float = 1e-3,
        dtype: str | np.dtype | None = None,
        nan_error: bool = False,
    ):
        if robust_scale <= 0:
            raise ValueError(f"robust_scale must be positive, got {robust_scale}")
        if robust_nu <= 0:
            raise ValueError(f"robust_nu must be positive, got {robust_nu}")
        if init_strategy not in ("svd", "random"):
            raise ValueError(f"init_strategy must be 'svd' or 'random', got {init_strategy!r}")
        self.rank = rank
        self.row_basis = row_basis
        self.col_basis = col_basis
        self.likelihood: Likelihood = (
            StudentTLikelihood(nu=float(robust_nu), scale=float(robust_scale))
            if robust
            else GaussianLikelihood()
        )
        self.init_strategy = init_strategy
        self.conv_tester = ConvergenceTester(strategy=conv_strategy, tol=conv_tol)
        self.dtype = DEFAULT_DTYPE if dtype is None else resolve_dtype(dtype)
        self.nan_error = nan_error
        self._state: FunctionalState | None = None
        self._loss_history: Array | None = None
        self._fit_info: FitInfo | None = None

    def fit(
        self,
        data: Measurements,
        max_iter: int = 1000,
        conv_check_cadence: int = 10,
        seed: int = 0,
        init_state: FunctionalState | None = None,
    ) -> tuple[FunctionalState, Array]:
        """Fit the model to measurements.

        Parameters
        ----------
        data : Measurements
            The measurements.
        max_iter : int, default=1000
            Maximum number of iterations.
        conv_check_cadence : int, default=10
            Test convergence every this many iterations.
        seed : int, default=0
            Random seed for ``init_strategy="random"``.
        init_state : FunctionalState | None, default=None
            Start from this state instead of initialising.

        Returns
        -------
        state : FunctionalState
            The fitted coefficients.
        loss_history : Array
            Loss at every iteration.
        """
        check_precision(self.dtype, self.rank)
        data = self._prepare(data)
        self._check_rank()
        layout = make_layout(data, self.row_basis, self.col_basis)

        if init_state is None:
            print("Initializing state... ", flush=True, end="")
            state = self._initial_state(data, seed)
            print("done.", flush=True)
        else:
            self._check_state(init_state)
            state = init_state
        state = FunctionalState(
            jnp.asarray(state.row_coeffs, self.dtype),
            jnp.asarray(state.col_coeffs, self.dtype),
            it=state.it,
        )

        monitor = RunMonitor(
            rank=self.rank,
            scale=getattr(self.likelihood, "scale", None),
            check_rises=True,
            infer=False,
            n_rows=self.row_basis.n_coeffs,
        )
        monitor.data_scale = float(self.likelihood.loss_from_residuals(data.y, data.w))
        state, loss_history, fit_info = run(
            layout,
            self.likelihood,
            self.row_basis,
            self.col_basis,
            self.conv_tester,
            monitor,
            state,
            max_iter,
            conv_check_cadence,
            self.dtype,
        )
        self._state, self._loss_history, self._fit_info = state, loss_history, fit_info
        return state, loss_history

    def predict(self, row: Array, col: Array, state: FunctionalState | None = None) -> Array:
        """The model ``Σₖ a_k(r) g_k(c)`` at paired coordinates ``(row[j], col[j])``."""
        state = self._resolve_state(state)
        self.row_basis.check_points(row)
        self.col_basis.check_points(col)
        a = self.row_basis.bind(jnp.asarray(row)).evaluate(state.row_coeffs)
        g = self.col_basis.bind(jnp.asarray(col)).evaluate(state.col_coeffs)
        return jnp.sum(a * g, axis=1)

    def robust_weights(self, data: Measurements, state: FunctionalState | None = None) -> Array:
        """Robust weight of each measurement, ``Q² / (Q² + w·r²)`` for the Student-t.

        Missing measurements (``w = 0``) get weight 1.
        """
        state = self._resolve_state(state)
        data = self._prepare(data)
        r = data.y - self.predict(data.row, data.col, state)
        return self.likelihood.weights_from_residuals(r, data.w)

    # -- helpers ------------------------------------------------------------- #

    def _prepare(self, data: Measurements) -> Measurements:
        if not isinstance(data, Measurements):
            raise TypeError(
                f"data must be Measurements, got {type(data).__name__}; use "
                "Measurements(row=..., col=..., y=..., w=...) or Measurements.from_matrix(Y, W)."
            )
        self.row_basis.check_points(data.row)
        self.col_basis.check_points(data.col)
        return sanitise_measurements(data.cast(self.dtype), self.nan_error)

    def _check_rank(self) -> None:
        n_row, n_col = self.row_basis.n_coeffs, self.col_basis.n_coeffs
        if self.rank < 1 or self.rank > min(n_row, n_col):
            raise ValueError(
                f"rank={self.rank} must lie in [1, min(n_row_coeffs, n_col_coeffs)] = "
                f"[1, {min(n_row, n_col)}] (row basis {n_row}, column basis {n_col})."
            )

    def _check_state(self, state: FunctionalState) -> None:
        want = {
            "row_coeffs": (self.row_basis.n_coeffs, self.rank),
            "col_coeffs": (self.col_basis.n_coeffs, self.rank),
        }
        for name, shape in want.items():
            got = tuple(getattr(state, name).shape)
            if got != shape:
                raise ValueError(f"init_state.{name} has shape {got}, expected {shape}.")

    def _initial_state(self, data: Measurements, seed: int) -> FunctionalState:
        n_row, n_col = self.row_basis.n_coeffs, self.col_basis.n_coeffs
        if self.init_strategy == "random":
            A, G = random_init(seed, n_row, n_col, self.rank)
            return FunctionalState(A, G, it=0)
        if not isinstance(self.row_basis, Discrete):
            raise ValueError(
                "init_strategy='svd' needs a Discrete row basis; use "
                "init_strategy='random' or pass init_state."
            )
        if isinstance(self.col_basis, Discrete):
            Y = jnp.zeros((n_row, n_col), self.dtype).at[data.row, data.col].set(data.y)
            A, G = svd_init(Y, n_row, n_col, self.rank)
            return FunctionalState(A, G, it=0)
        return self._projection_svd_state(data)

    def _projection_svd_state(self, data: Measurements) -> FunctionalState:
        """SVD start for a Discrete row basis and any column basis.

        Each row's measurements are projected onto the column basis
        (``Φᵢᵀ yᵢ``, which for a Discrete column basis is the zero-filled data
        row). The leading right singular vectors of these projections seed the
        column coefficients, and the row coefficients are then their weighted
        least-squares fit to the data.
        """
        row_op = self.row_basis.bind(data.row)
        col_op = self.col_basis.bind(data.col)
        y = jnp.where(data.w > 0, data.y, 0.0)
        D = row_op.adjoint(y[:, None] * col_op.design())
        _, s, Vh = jnp.linalg.svd(D, full_matrices=False)
        G = Vh[: self.rank].T * jnp.sqrt(s[: self.rank])
        A = row_op.solve(col_op.evaluate(G), data.y, data.w, col_op.n_coeffs)
        return FunctionalState(A, G, it=0)

    def _resolve_state(self, state: FunctionalState | None) -> FunctionalState:
        state = self._state if state is None else state
        if state is None:
            raise ValueError("No state: call fit() first or pass state.")
        return state

    @property
    def state(self) -> FunctionalState | None:
        """State from the last `fit`."""
        return self._state

    @property
    def loss_history(self) -> Array | None:
        """Loss history from the last `fit`."""
        return self._loss_history

    @property
    def fit_info(self) -> FitInfo | None:
        """How the last `fit` ran."""
        return self._fit_info
