# robusta.py

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .convergence import ConvergenceTester
from .coverage import (
    cascade,
    exclusion_message,
    indices_by_axis,
    negligible_ridge,
    negligible_ridge_message,
    ridge_message,
    under_covered,
)
from .fit_info import FitInfo
from .frame import OptFrame
from .hmf import HMF, OptMethod
from .initialisation import Initialiser
from .metrics import ZMoments
from .precision import DEFAULT_DTYPE, check_precision, resolve_dtype
from .rotations import RotationMethod
from .state import (
    RHMFState,
    check_state_finite,
    check_state_shape,
    fill_excluded,
    has_exclusions,
    with_exclusions,
    working_form,
)
from .validation import check_rank
from .warnings import UnderdeterminedWarning


@dataclass
class Robusta:
    """
    Robust heteroskedastic matrix factorisation (RHMF) of a data matrix.

    Factorises an ``(N, M)`` data matrix ``Y`` with inverse-variance weights
    ``W`` as ``Y ≈ A @ G.T``, where ``A`` ``(N, K)`` holds per-row coefficients
    and ``G`` ``(M, K)`` holds ``K`` shared basis vectors (``K`` is ``rank``).
    Missing entries are entries with ``W = 0``.

    With ``robust=True`` (the default) the fit maximises a Student-t
    likelihood by iteratively reweighted least squares: each iteration
    computes per-entry robust weights ``Q² / (Q² + W r²)``, where ``r`` is the
    residual and ``Q² = robust_nu · robust_scale²``, then solves weighted least
    squares for ``A`` and then ``G``, then rotates so the columns of ``G`` are
    orthonormal. Entries the model cannot explain get robust weights near 0,
    which flags them as outliers. With ``robust=False`` it is plain
    heteroskedastic matrix factorisation (HMF).

    The interface follows scikit-learn: :meth:`fit` learns ``A`` and ``G``,
    :meth:`infer` finds coefficients for new rows with ``G`` held fixed,
    :meth:`score` scores held-out data for choosing ``rank`` and
    ``robust_scale``, and :meth:`robust_weights` returns the per-entry outlier
    weights. The model keeps the state of its last :meth:`fit`, which the
    other methods use when no ``state`` is passed.

    Examples
    --------
    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> from robusta_hmf import Robusta
    >>> model = Robusta(rank=5, robust_scale=3.0)
    >>> state, loss_history = model.fit(Y, W)
    >>> weights = model.robust_weights(Y, W)        # (N, M), 1 = inlier
    >>> per_row = jnp.quantile(weights, 0.01, axis=1)  # one outlier score per row
    """

    rank: int
    method: OptMethod
    dtype: np.dtype

    _hmf: HMF
    _initialiser: Initialiser
    _conv_tester: ConvergenceTester
    _frame: OptFrame

    # Internal state from last fit
    _state: RHMFState | None = None
    _loss_history: Array | None = None
    _fit_info: FitInfo | None = None
    nan_error: bool = False  # class default, so models pickled before it existed still load

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
        # Execution params
        override_frame: OptFrame | None = None,
        # HMF params
        als_ridge: float | None = None,
        learning_rate: float = 1e-3,
        rotation: RotationMethod = "fast",
        # Precision
        dtype: str | np.dtype | None = None,
        # Data checks
        nan_error: bool = False,
        **rotation_kwargs,
    ):
        """
        Configure a model. Nothing is computed until :meth:`fit`.

        Parameters
        ----------
        rank : int
            Number of basis vectors, ``K``. It must be chosen before fitting
            (unlike PCA, the components of a rank-``K`` fit are not the first
            ``K`` of a larger fit); :meth:`score` on held-out data helps choose it.
        method : {"als", "sgd"}, default="als"
            ``"als"``: iteratively reweighted alternating least squares, the
            algorithm of the paper. ``"sgd"``: gradient descent on the loss
            with optax (Adafactor unless overridden).
        robust : bool, default=True
            If True, use the Student-t likelihood, which downweights outliers;
            if False, use a Gaussian likelihood (plain HMF, all robust weights 1).
        robust_nu : float, default=1.0
            Student-t degrees of freedom. The fit depends on ``robust_nu`` and
            ``robust_scale`` only through ``Q² = robust_nu · robust_scale²``,
            so the default of 1 (a Cauchy likelihood) loses nothing.
        robust_scale : float, default=1.0
            The soft outlier threshold ``Q`` (with ``robust_nu=1``), in units
            of the measurement uncertainty: residuals much smaller than
            ``Q / sqrt(W)`` are treated as Gaussian noise, much larger ones are
            downweighted. Values of roughly 2 to 5 are typical starting points.
        init_strategy : {"svd", "random"}, default="svd"
            How ``A`` and ``G`` are seeded. ``"svd"`` uses the leading ``rank``
            singular vectors of ``Y`` (scaled by the square roots of the
            singular values); ``"random"`` draws standard normals from ``seed``.
            To start from factors you already have, pass ``init_state`` to
            :meth:`fit` instead.
        override_initialiser : Initialiser | None, default=None
            Custom initialiser; replaces ``init_strategy`` if given.
        conv_strategy : str, default="max_frac_G"
            Convergence test, checked every ``conv_check_cadence`` iterations:
            ``"max_frac_G"`` (largest squared change in ``G`` relative to the
            mean square of ``G``), ``"max_frac_A"`` (the same for ``A``),
            ``"rel_frac_loss"`` (relative change in the loss) or ``"none"``
            (run to ``max_iter``).
        conv_tol : float, default=1e-3
            Tolerance for ``conv_strategy``.
        override_conv_tester : ConvergenceTester | None, default=None
            Custom convergence tester; replaces ``conv_strategy`` and
            ``conv_tol`` if given.
        override_frame : OptFrame | None, default=None
            Custom optimisation frame, which owns how the loop is scheduled.
            Defaults to a plain ``OptFrame`` that evaluates each step over the
            whole matrix. Pass
            :class:`robusta_hmf.sharded.ShardedFrame` to chunk and shard the
            row axis, for data larger than one device's memory.
        als_ridge : float | None, default=None
            Ridge added to the diagonal of every least-squares solve for ``A``
            and ``G`` (``method="als"`` only). Without it, rows and columns
            with fewer non-zero weights than ``rank`` are excluded from the
            fit; with it they are kept and shrunk towards zero, and an
            :class:`~robusta_hmf.warnings.UnderdeterminedWarning` names them.
        learning_rate : float, default=1e-3
            Learning rate of the default Adafactor optimiser
            (``method="sgd"`` only).
        rotation : {"fast", "slow", "identity"}, default="fast"
            How ``A`` and ``G`` are rotated after each update to remove the
            factorisation's ``K × K`` degeneracy, leaving ``A @ G.T``
            unchanged. ``"fast"`` makes the columns of ``G`` orthonormal by
            symmetric whitening (``K × K`` eigendecomposition). ``"slow"``
            uses an SVD so that ``G`` is orthonormal and the columns of ``A``
            are orthogonal and ordered by decreasing norm, like PCA.
            ``"identity"`` does not rotate.
        dtype : {"float64", "float32"} or dtype, default="float64"
            Precision of the fit. ``Y``, ``W``, ``A`` and ``G`` are cast to it.
            float64 needs JAX's float64 support, enabled with
            ``jax.config.update("jax_enable_x64", True)`` before any JAX array
            is created; without it, fitting raises
            :class:`~robusta_hmf.precision.PrecisionError` rather than silently
            running in float32. float32 is faster, especially on GPUs, but can
            diverge when rows or columns of ``W`` have fewer non-zero weights
            than ``rank``, so it has to be requested explicitly. When None, a
            ``compute_dtype`` set on ``override_frame`` is used, else float64.
        nan_error : bool, default=False
            What :meth:`fit`, :meth:`infer` and :meth:`score` do with entries
            where ``Y`` or ``W`` is NaN or infinite. With False they are
            treated as missing (``Y = W = 0`` there) and one
            :class:`~robusta_hmf.warnings.DataWarning` is issued; with True
            they raise ``ValueError``. Negative weights always raise.
        **rotation_kwargs
            Passed to the rotation. For ``"fast"``: ``target`` (``"G"``, the
            default, or ``"A"``: which factor is made orthonormal), ``whiten``
            (default True) and ``eps`` (default 1e-6, floor on the eigenvalues).
        """
        self.rank = rank
        self.method = method
        self.nan_error = nan_error
        self.dtype = self._resolve_model_dtype(dtype, override_frame)

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

        # Build optimization frame. A provided frame carries only its
        # scheduling parameters; bind() wires it to this model so the two never
        # hold duplicate copies of the configuration.
        if override_frame is None:
            self._frame = OptFrame(
                method=self._hmf, conv_tester=self._conv_tester, compute_dtype=self.dtype
            )
        else:
            self._frame = override_frame.bind(self._hmf, self._conv_tester, self.dtype)

        # Initialize internal state
        self._state = None
        self._loss_history = None
        self._fit_info = None

    @staticmethod
    def _resolve_model_dtype(dtype, override_frame: OptFrame | None) -> np.dtype:
        """The model's dtype, reconciled with any ``compute_dtype`` on the frame."""
        frame_dtype = None if override_frame is None else override_frame.requested_dtype()
        frame_dtype = None if frame_dtype is None else resolve_dtype(frame_dtype)
        if dtype is None:
            return frame_dtype if frame_dtype is not None else DEFAULT_DTYPE
        dtype = resolve_dtype(dtype)
        if frame_dtype is not None and frame_dtype != dtype:
            raise ValueError(
                f"Robusta(dtype={dtype.name!r}) conflicts with the frame's "
                f"compute_dtype={frame_dtype.name!r}; set only one of them."
            )
        return dtype

    def set_state(self, state: RHMFState) -> "Robusta":
        """
        Return a copy of this model holding ``state`` as its fitted state.

        Use it to restore a saved fit without refitting, e.g. with a state
        from :func:`~robusta_hmf.load_state_from_npz`, so that methods called
        without ``state`` use it. The model itself is left unchanged.

        Parameters
        ----------
        state : RHMFState
            The state the copy should hold.

        Returns
        -------
        model : Robusta
            A deep copy of this model with ``state`` set. Its
            ``loss_history`` and ``fit_info`` are those of this model.
        """
        new_obj = deepcopy(self)
        new_obj._state = state
        return new_obj

    def fit(
        self,
        Y: Array,
        W: Array | None = None,
        max_iter: int = 1000,
        rotation_cadence: int = 1,
        conv_check_cadence: int = 10,
        seed: int = 0,
        init_state: RHMFState | None = None,
    ) -> tuple[RHMFState, Array]:
        """
        Fit ``A`` and ``G`` to data, and keep the result on the model.

        Iterates until the convergence test passes or ``max_iter`` is reached,
        printing the loss at every convergence check. Rows and columns of
        ``W`` with fewer non-zero weights than ``rank`` are excluded first
        (unless ``als_ridge`` is set), since the data cannot determine their
        factors; see :class:`~robusta_hmf.FitInfo` for how the run is recorded.

        Parameters
        ----------
        Y : Array, shape (N, M)
            Data matrix. May instead be a handle returned by the frame's
            ``prepare()`` (e.g. a :class:`~robusta_hmf.ShardedData`), in which
            case ``W`` is carried inside it and the ``W`` argument is ignored.
            Reuse that handle to fit several models to one sample without
            re-transferring it.
        W : Array, shape (N, M)
            Inverse-variance weights, ``1 / σ²``. Required unless ``Y`` is a
            prepared handle. Zero marks a missing entry. Entries where ``Y``
            or ``W`` is NaN or infinite are treated as missing (or raise, with
            ``nan_error=True``); negative weights raise ``ValueError``.
        max_iter : int, default=1000
            Maximum number of iterations.
        rotation_cadence : int, default=1
            Rotate every this many iterations. ALS always rotates every
            iteration (another value only triggers a warning); this matters for
            ``method="sgd"``.
        conv_check_cadence : int, default=10
            Test convergence (and print the loss) every this many iterations.
        seed : int, default=0
            Random seed, used by ``init_strategy="random"``.
        init_state : RHMFState | None, default=None
            Start from this state instead of initialising, e.g. to continue a
            fit or to start from factors computed elsewhere. Its ``A`` and
            ``G`` must have shapes ``(N, rank)`` and ``(M, rank)``. Exclusions
            are recomputed from ``W``, so rows and columns the state excluded
            come back if ``W`` now covers them.

        Returns
        -------
        state : RHMFState
            The fitted factors. Excluded rows of ``A`` and ``G`` are NaN and
            marked by ``state.row_excluded`` / ``state.col_excluded``.
        loss_history : Array, shape (n_iter,)
            The loss after each iteration.

        Raises
        ------
        PrecisionError
            If the model's dtype is float64 and JAX's float64 support is off.
        FloatingPointError
            If the loss or the factors become NaN or infinite.

        Warns
        -----
        UnderdeterminedWarning
            If rows or columns are excluded (or, with ``als_ridge``, are
            determined only by the ridge).
        ConvergenceWarning
            If the fit reaches ``max_iter`` without converging, or the loss
            rises by more than round-off allows.
        DataWarning
            If NaN or infinite entries were treated as missing.

        Notes
        -----
        After the fit, :attr:`fit_info` records whether it converged, the
        number of iterations, any loss rises and any exclusions.
        """
        check_precision(self.dtype, self.rank)
        check_rank(self.rank, *_data_shape(Y))

        # Let the frame place the data first: the initialiser and the loop both
        # consume whatever it hands back.
        Y, W = self._frame.prepare(Y, W, nan_error=self.nan_error)
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
                self._initialiser = self._frame.make_initialiser(
                    N=N,
                    M=M,
                    K=self.rank,
                    strategy=self._init_strategy,
                )
            else:
                # A user-supplied initialiser may have been built without
                # knowing the data shape, or without knowing which frame it
                # would run under; fill in both.
                self._initialiser = self._frame.adopt_initialiser(
                    self._initialiser.with_dims(N, M, self.rank)
                )
            print("Initializing state... ", flush=True, end="")
            init_state = self._initialiser.execute(
                seed=seed,
                Y=Y,
                opt=self._hmf.opt if self.method == "sgd" else None,
            )
            print("done.", flush=True)
        else:
            check_state_shape(init_state, N, M, self.rank, "init_state")
            check_state_finite(init_state)
            init_state = _restart_point(init_state)

        # Rows and columns the data cannot determine are excluded up front, from
        # these data: masks on an init_state only mark which rows start at zero.
        if self._ridge is None:
            row_excluded, col_excluded, exclusions = self._exclusions(Y, W)
            under = (np.zeros(0, int), np.zeros(0, int))
        else:
            row_excluded = col_excluded = None
            exclusions = ()
            under = self._report_ridge_determined(Y, W)
        init_state = with_exclusions(init_state, row_excluded, col_excluded)
        ridge_warned = self._check_ridge_resolution(Y, W, init_state, under)

        # Run optimization
        final_state, loss_history, fit_info = self._frame.run(
            Y=Y,
            W=W,
            init_state=init_state,
            rotation_cadence=rotation_cadence,
            conv_check_cadence=conv_check_cadence,
            max_iter=max_iter,
        )

        if not ridge_warned:
            self._check_ridge_resolution(Y, W, final_state, under)

        # Store internally
        self._state = final_state
        self._loss_history = loss_history
        self._fit_info = replace(fit_info, exclusions=exclusions + fit_info.exclusions)

        return final_state, loss_history

    @property
    def _ridge(self) -> float | None:
        return self._hmf.a_step.ridge if self.method == "als" else None

    def _exclusions(self, Y, W, col_excluded=None, *, infer: bool = False):
        """Rows (and, when fitting, columns) whose coverage is below the rank.

        Excludes them and cascades to a fixed point, warning once. In ``infer``
        the columns are fixed by ``col_excluded`` (from training) and only rows
        are excluded. Returns ``(row_excluded, col_excluded, events)``, with
        ``None`` for an axis where nothing is excluded.
        """
        K = self.rank
        N, M = _data_shape(Y)
        coverage = partial(self._frame.coverage, Y, W)
        row_active, col_active, events = cascade(
            coverage, K, np.ones(N, bool), _active(col_excluded, M), update_cols=not infer
        )
        if not infer and (row_active.sum() < K or col_active.sum() < K):
            raise ValueError(
                f"After excluding rows and columns with fewer than {K} non-zero weights, "
                f"{row_active.sum()} rows and {col_active.sum()} columns remain, fewer than "
                f"rank={K}: there is nothing left to fit. Check W, or lower the rank."
            )
        if events:
            rows, cols = indices_by_axis(events)
            warnings.warn(
                exclusion_message(K, rows, cols, infer=infer), UnderdeterminedWarning, stacklevel=3
            )
        return (
            None if row_active.all() else ~row_active,
            None if col_active.all() else ~col_active,
            tuple(events),
        )

    def _report_ridge_determined(self, Y, W, col_excluded=None, *, infer: bool = False):
        """With ``als_ridge`` set: warn about the rows and columns the ridge determines.

        Nothing is excluded. Returns ``(rows, cols)``, the under-covered indices
        (columns only when fitting).
        """
        K = self.rank
        N, M = _data_shape(Y)
        coverage = partial(self._frame.coverage, Y, W)
        rows, cols = under_covered(coverage, K, np.ones(N, bool), _active(col_excluded, M))
        if infer:
            cols = cols[:0]
        if rows.size or cols.size:
            warnings.warn(
                ridge_message(K, rows, cols, self._ridge), UnderdeterminedWarning, stacklevel=3
            )
        return rows, cols

    def _check_ridge_resolution(self, Y, W, state: RHMFState, under) -> bool:
        """Warn if ``als_ridge`` is too small to affect any under-covered system.

        A ridge changes a system only if it is resolvable next to the
        system's diagonal at the working precision: here, if
        ``ridge > 10·ε·d`` for d its largest diagonal entry. Only systems
        with fewer than K non-zero weights need the ridge, so only those are
        checked. Returns whether it warned.
        """
        rows, cols = under
        if self._ridge is None or not (rows.size or cols.size):
            return False
        dA, dG = self._frame.max_diagonals(Y, W, state)
        eps = float(np.finfo(self.dtype).eps)
        bad_rows = rows[negligible_ridge(self._ridge, eps, dA[rows])]
        bad_cols = cols[negligible_ridge(self._ridge, eps, dG[cols])]
        if not (bad_rows.size or bad_cols.size):
            return False
        d_max = float(max(dA[bad_rows].max(initial=0), dG[bad_cols].max(initial=0)))
        warnings.warn(
            negligible_ridge_message(self._ridge, self.dtype, bad_rows, bad_cols, d_max),
            UnderdeterminedWarning,
            stacklevel=3,
        )
        return True

    def synthesize(
        self,
        state: RHMFState | None = None,
        indices: Array | None = None,
    ) -> Array:
        """
        The model's reconstruction of the data, ``A @ G.T``.

        Because outliers are downweighted in the fit, this is the low-rank
        description of each row without its outliers; ``Y`` minus this is
        where they show up (see :meth:`residuals`).

        Parameters
        ----------
        state : RHMFState | None, default=None
            State to use, e.g. the output of :meth:`infer`. If None, use the
            last fit.
        indices : Array | None, default=None
            If given, reconstruct only these rows (indices into ``A``).

        Returns
        -------
        synthesis : Array, shape (N, M) or (len(indices), M)
            The reconstruction. NaN in rows and columns the state excludes,
            where the model is undefined.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")

        A = state.A if indices is None else state.A[indices]
        return A @ state.G.T

    def infer(
        self,
        Y_infer: Array,
        W_infer: Array | None = None,
        state: RHMFState | None = None,
        max_iter: int = 100,
        conv_strategy: str = "max_frac_A",
        conv_tol: float = 1e-3,
        conv_check_cadence: int = 10,
    ) -> tuple[RHMFState, Array]:
        """
        Infer coefficients for new rows with the basis ``G`` held fixed.

        Alternates the robust-weight update and the least-squares solve for
        ``A`` (never ``G``), starting from a weighted least-squares fit with
        the data weights alone. Always uses least squares, even for a model
        fitted with ``method="sgd"``. Runs through the same frame as
        :meth:`fit`, so a sharded model infers on sharded data with the same
        chunking. The model's stored state is not changed, since one model is
        often applied to many datasets.

        Use it to apply a trained basis to held-out data before
        :meth:`score`, or to new data; then call :meth:`robust_weights` with
        the returned state to find outliers in the new rows.

        Parameters
        ----------
        Y_infer : Array, shape (N', M)
            New data with the same ``M`` columns as the training data, or a
            handle from the frame's ``prepare()``.
        W_infer : Array, shape (N', M)
            Inverse-variance weights for ``Y_infer``. Required unless
            ``Y_infer`` is a prepared handle.
        state : RHMFState | None, default=None
            State supplying the fixed basis. If None, use the last fit.
            Columns it excludes stay excluded; new rows with fewer than
            ``rank`` non-zero weights in the other columns are excluded.
        max_iter : int, default=100
            Maximum iterations.
        conv_strategy : str, default="max_frac_A"
            Convergence strategy. ``"max_frac_G"`` is meaningless here (G is
            fixed) and is switched to ``"max_frac_A"`` with a warning.
        conv_tol : float, default=1e-3
            Convergence tolerance.
        conv_check_cadence : int, default=10
            How often to check convergence.

        Returns
        -------
        state : RHMFState
            ``A`` of shape ``(N', rank)`` for the new rows (NaN for excluded
            rows), with ``G`` unchanged.
        loss_history : Array, shape (n_iter,)
            The loss after each iteration.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError(
                "No trained state available. Either provide state or call fit() first."
            )
        check_precision(self.dtype, self.rank)
        check_rank(self.rank, *_data_shape(Y_infer), allow_few_rows=True)
        check_state_shape(state, None, _data_shape(Y_infer)[1], self.rank)
        check_state_finite(state)
        # Columns excluded in training have NaN rows of G; they stay excluded.
        col_excluded = state.col_excluded
        G_fixed = jnp.asarray(state.G, self.dtype)
        if col_excluded is not None:
            G_fixed = jnp.where(col_excluded[:, None], jnp.zeros((), self.dtype), G_fixed)

        # Place the held-out data the same way the frame places training data.
        Y_infer, W_infer = self._frame.prepare(Y_infer, W_infer, nan_error=self.nan_error)

        # New rows with fewer than K non-zero weights in the active columns are excluded.
        n_rows = _data_shape(Y_infer)[0]
        if self._ridge is None:
            row_excluded, _, _ = self._exclusions(Y_infer, W_infer, col_excluded, infer=True)
        else:
            row_excluded = None
            self._report_ridge_determined(Y_infer, W_infer, col_excluded, infer=True)
        if row_excluded is not None and row_excluded.all():
            # Nothing to infer: every row is excluded.
            empty = RHMFState(
                A=jnp.zeros((n_rows, self.rank), self.dtype),
                G=G_fixed,
                row_excluded=row_excluded,
                col_excluded=col_excluded,
            )
            return fill_excluded(empty, jnp.nan), jnp.zeros(0, self.dtype)

        # Get an initial guess for the coefficients using the data weights
        ridge = self._ridge
        A_init = self._frame.initial_coefficients(
            Y_infer, W_infer, G_fixed, ridge, row_excluded=row_excluded
        )

        # We can't use conv_strategy of max_frac_G here because G is fixed
        if conv_strategy == "max_frac_G":
            warnings.warn("max_frac_G is not supported for infer(). Using max_frac_A instead.")
            conv_strategy = "max_frac_A"

        # Run one-sided optimization through this model's frame
        conv_tester = ConvergenceTester(strategy=conv_strategy, tol=conv_tol)
        frame = self._frame.bind(self._hmf, conv_tester, self.dtype)
        final_state, loss_history, _ = frame.run(
            Y=Y_infer,
            W=W_infer,
            init_state=RHMFState(
                A=A_init,
                G=G_fixed,
                it=0,
                row_excluded=row_excluded,
                col_excluded=col_excluded,
            ),
            rotation_cadence=1,  # No rotation since G is fixed so this is a dummy value
            conv_check_cadence=conv_check_cadence,
            max_iter=max_iter,
            skip_G=True,
        )
        return final_state, loss_history  # Not stored because this is not the main fit

    def score(
        self,
        Y: Array,
        W: Array | None = None,
        state: RHMFState | None = None,
    ) -> ZMoments:
        """
        Held-out scores for a state, as sufficient statistics.

        Computes the standardised residuals
        ``z = (Y - A @ G.T) · sqrt(W · w_robust)``, which follow ``N(0, 1)``
        when the model is calibrated and the outliers are correctly
        downweighted, and summarises them. ``.kl``, the KL divergence of a
        Gaussian with the mean and standard deviation of ``z`` from
        ``N(0, 1)``, is the model-selection score of the paper: lower is
        better. Read it alongside a goodness-of-fit measure such as
        ``.chi2_red``, since a rank that is too small with an aggressive
        ``robust_scale`` can also score well by downweighting most of the data.

        Typical use is cross-validation over ``rank`` and ``robust_scale``:
        :meth:`fit` on a training split, :meth:`infer` on the held-out split,
        then score the inferred state.

        Parameters
        ----------
        Y : Array, shape (N, M)
            Held-out data, or a handle from the frame's ``prepare()``.
        W : Array, shape (N, M), optional
            Held-out weights.
        state : RHMFState | None, default=None
            State to score, normally the output of :meth:`infer`. If None, use
            the last fit.

        Returns
        -------
        moments : ZMoments
            Read ``.kl`` for the paper's model-selection score (Eq. 20), or
            ``.std_z``, ``.chi2_red``, ``.rmse``. Accumulated chunk by chunk
            when the frame is sharded, so this never forms an ``(N, M)`` array.

        See Also
        --------
        robusta_hmf.metrics.mad_z : needs the full residual array, so it is
            not available here.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        check_precision(self.dtype, self.rank)
        check_state_shape(state, *_data_shape(Y), self.rank)
        check_state_finite(state)
        Y, W = self._frame.prepare(Y, W, nan_error=self.nan_error)
        moments = self._frame.z_moments(Y, W, self._frame.cast_state(state))
        if not has_exclusions(state):
            return moments
        N, M = _data_shape(Y)
        rows, cols = _active(state.row_excluded, N), _active(state.col_excluded, M)
        everything = np.ones(N, bool), np.ones(M, bool)
        total = int(self._frame.coverage(Y, W, *everything)[0].sum())
        kept = int(self._frame.coverage(Y, W, rows, cols)[0][rows].sum())
        return replace(
            moments,
            n_excluded_rows=int((~rows).sum()),
            n_excluded_cols=int((~cols).sum()),
            n_excluded_entries=total - kept,
        )

    def basis_vectors(self, state: RHMFState | None = None) -> Array:
        """
        The basis vectors, ``G``.

        Parameters
        ----------
        state : RHMFState | None, default=None
            State to use. If None, use the last fit.

        Returns
        -------
        G : Array, shape (M, K)
            One basis vector per column. With the default rotation the
            columns are orthonormal; they are not ordered by explained
            variance. Rows for excluded columns of ``Y`` are NaN.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        return state.G

    def coefficients(self, state: RHMFState | None = None) -> Array:
        """
        The coefficients, ``A``.

        Parameters
        ----------
        state : RHMFState | None, default=None
            State to use, e.g. the output of :meth:`infer`. If None, use the
            last fit.

        Returns
        -------
        A : Array, shape (N, K)
            The coefficients of each row of ``Y`` on the basis vectors. Rows
            for excluded rows of ``Y`` are NaN.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        return state.A

    def residuals(self, Y: Array, state: RHMFState | None = None) -> Array:
        """
        Residuals of the data from the model, ``Y - A @ G.T``.

        Large residuals relative to the uncertainties mark outliers; multiply
        by ``sqrt(W)`` to put them in units of the uncertainty.

        Parameters
        ----------
        Y : Array, shape (N, M)
            The data the state describes (the training data for the last
            fit, or the data passed to :meth:`infer` for its state).
        state : RHMFState | None, default=None
            State to use. If None, use the last fit.

        Returns
        -------
        residuals : Array, shape (N, M)
            The residuals. NaN in rows and columns the state excludes.
        """
        return Y - self.synthesize(state=state)

    def mse(self, Y: Array, state: RHMFState | None = None) -> float:
        """
        Not implemented: always raises ``NotImplementedError``.

        Use :meth:`score` for goodness-of-fit statistics (``.chi2_red``,
        ``.rmse``) instead.

        Parameters
        ----------
        Y : Array, shape (N, M)
            Data matrix.
        state : RHMFState | None, default=None
            State to use. If None, use the last fit.

        Raises
        ------
        NotImplementedError
            Always.
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
        Per-entry robust weights, the outlier measure of RHMF.

        ``w = Q² / (Q² + W r²)`` with ``r = Y - A @ G.T`` and
        ``Q² = robust_nu · robust_scale²``: 1 for an entry the model fits
        exactly, falling towards 0 as the residual grows past about ``Q``
        standard deviations. Summarise them per row for an outlier score per
        object, e.g. the median or a low quantile (the paper uses the 1%
        quantile and flags rows below 0.5), or per column for one per feature.
        With ``robust=False`` every weight is 1.

        Parameters
        ----------
        Y : Array, shape (N, M)
            The data the state describes.
        W : Array, shape (N, M)
            Inverse-variance weights for ``Y``. Entries with ``W = 0`` get
            weight 1.
        state : RHMFState | None, default=None
            State to use, e.g. the output of :meth:`infer` for new data. If
            None, use the last fit.

        Returns
        -------
        weights : Array, shape (N, M)
            Robust weights in ``(0, 1]``. Entries in rows or columns the state excludes are
            treated like missing data and get weight 1, so per-row summaries
            stay finite; ``state.row_excluded`` / ``col_excluded`` mark them.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        W, state = working_form(W, state)
        return self._hmf.likelihood.weights_irls(Y, W, state.A, state.G)

    # Convenience properties
    @property
    def A(self) -> Array | None:
        """``A`` ``(N, K)`` from the last :meth:`fit`, or None before fitting."""
        return self._state.A if self._state is not None else None

    @property
    def G(self) -> Array | None:
        """``G`` ``(M, K)`` from the last :meth:`fit`, or None before fitting."""
        return self._state.G if self._state is not None else None

    @property
    def state(self) -> RHMFState | None:
        """The :class:`~robusta_hmf.state.RHMFState` of the last :meth:`fit`.

        None before fitting.
        """
        return self._state

    @property
    def loss_history(self) -> Array | None:
        """The loss after each iteration of the last :meth:`fit`, or None before fitting."""
        return self._loss_history

    @property
    def fit_info(self) -> FitInfo | None:
        """The :class:`~robusta_hmf.FitInfo` of the last :meth:`fit`, or None before fitting.

        Not updated by :meth:`infer`.
        """
        return self._fit_info


def _data_shape(Y) -> tuple[int, int]:
    """``(rows, columns)`` of an array or of a frame's prepared data handle."""
    shape = Y.shape if hasattr(Y, "shape") else np.shape(Y)
    if len(shape) != 2:
        raise ValueError(f"Y must be 2-D, got shape {tuple(shape)}")
    return int(shape[0]), int(shape[1])


def _active(excluded, n: int) -> np.ndarray:
    """Host bool mask of active entries from an optional exclusion mask."""
    return np.ones(n, bool) if excluded is None else ~np.asarray(excluded, bool)


def _restart_point(state: RHMFState) -> RHMFState:
    """``state`` as a starting point: excluded (NaN) rows set to 0, masks dropped.

    Exclusions are recomputed from the data being fitted, so rows the state
    excluded can come back if the new weights cover them.
    """
    if not has_exclusions(state):
        return state
    for name, mask in (("A", state.row_excluded), ("G", state.col_excluded)):
        if mask is not None and bool(np.asarray(mask).all()):
            raise ValueError(
                f"init_state has every row of {name} excluded, so there is nothing to start "
                "the fit from."
            )
    return with_exclusions(fill_excluded(state, 0.0), None, None)
