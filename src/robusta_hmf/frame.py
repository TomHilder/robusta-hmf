# frame.py

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .als import WeightedAStep
from .convergence import ConvergenceTester, RunMonitor
from .fit_info import FitInfo
from .hmf import HMF
from .initialisation import Initialiser, InitStrategy
from .metrics import ZMoments, z_moments
from .precision import nonfinite_factors_message, nonfinite_loss_message
from .state import RHMFState, fill_excluded, has_exclusions, mask_weights, update_state
from .validation import sanitise


@dataclass
class OptFrame:
    """Owns the optimisation loop: data placement, initialisation, iteration.

    :class:`~robusta_hmf.Robusta` configures the model (an :class:`HMF`) and
    delegates execution to a frame. A subclass can change *how* the loop is
    scheduled -- see :class:`robusta_hmf.sharded.ShardedFrame`, which chunks
    and shards the row axis -- without re-declaring the model's parameter list.

    ``compute_dtype`` is the dtype of the arithmetic and of ``A``/``G``.
    ``Robusta`` sets it through :meth:`bind`; ``None`` leaves arrays in
    whatever dtype they arrive in.
    """

    method: HMF
    conv_tester: ConvergenceTester
    compute_dtype: Any = None

    def bind(
        self, method: HMF, conv_tester: ConvergenceTester, compute_dtype: Any = None
    ) -> "OptFrame":
        """Return a copy of this frame wired to a model and a convergence test.

        Lets a user construct a frame with only its scheduling parameters --
        ``ShardedFrame(row_block=4096)`` -- and have ``Robusta`` fill in the
        rest, so a model is still configured in exactly one place. A
        ``compute_dtype`` already set on the frame is kept when none is given.
        """
        if compute_dtype is None:
            compute_dtype = self.compute_dtype
        return replace(self, method=method, conv_tester=conv_tester, compute_dtype=compute_dtype)

    # -- hooks that a scheduling subclass overrides -------------------------- #

    def prepare(self, Y: Array, W: Array, nan_error: bool = False) -> tuple[Any, Any]:
        """Validate ``(Y, W)`` and place them however this frame wants them for the loop.

        The base frame converts them to JAX arrays in ``compute_dtype``, then
        treats entries where either is non-finite as missing (``Y = W = 0``,
        with a :class:`~robusta_hmf.warnings.DataWarning`), or raises
        ``ValueError`` for them if ``nan_error`` is set. Negative weights raise
        ``ValueError``. The return value is fed to both the initialiser and
        :meth:`run`, so a frame needing a resident device layout builds it
        once here rather than per grid point.
        """
        if W is None:
            raise ValueError("W (the inverse-variance weights) is required.")
        # Values too large for compute_dtype become inf here; sanitise() counts
        # and reports them, so numpy's overflow warning would only duplicate it.
        with np.errstate(over="ignore"):
            Y = jnp.asarray(Y, self.compute_dtype)
            W = jnp.asarray(W, self.compute_dtype)
        return sanitise(Y, W, nan_error)

    def requested_dtype(self):
        """The ``compute_dtype`` the user set on this frame, or None."""
        return self.compute_dtype

    def cast_state(self, state: RHMFState) -> RHMFState:
        """Return ``state`` with ``A`` and ``G`` in ``compute_dtype``."""
        if self.compute_dtype is None:
            return state
        return update_state(
            state,
            A=jnp.asarray(state.A, self.compute_dtype),
            G=jnp.asarray(state.G, self.compute_dtype),
        )

    def make_initialiser(self, N: int, M: int, K: int, strategy: InitStrategy) -> Initialiser:
        """The initialiser this frame can consume, when none was overridden."""
        return Initialiser(N=N, M=M, K=K, strategy=strategy)

    def adopt_initialiser(self, initialiser: Initialiser) -> Initialiser:
        """Adapt a user-supplied initialiser to this frame's execution model.

        The base frame takes any initialiser as-is. A frame that hands its
        initialiser something other than a plain array -- as the sharded one
        does -- uses this to swap in an equivalent it can actually consume,
        rather than failing deep inside ``execute()``.
        """
        return initialiser

    def initial_coefficients(
        self, Y: Array, W: Array, G: Array, ridge: float | None, row_excluded=None
    ) -> Array:
        """One weighted A step against a fixed basis, seeding :meth:`Robusta.infer`.

        Rows marked in ``row_excluded`` get zero weight, so they solve to zero
        rather than to an arbitrary point of a singular system.
        """
        A_dummy = jnp.zeros((Y.shape[0], G.shape[1]), dtype=G.dtype)
        W = mask_weights(W, row_excluded, None)
        return WeightedAStep(ridge=ridge)(Y, W, RHMFState(A=A_dummy, G=G, it=0)).A

    def max_diagonals(self, Y, W, state: RHMFState) -> tuple[np.ndarray, np.ndarray]:
        """Largest diagonal entry of each row's and each column's least-squares system.

        ``max_k (W_t @ G**2)[i, k]`` for the rows of ``A`` and
        ``max_k (W_tᵀ @ A**2)[j, k]`` for the rows of ``G``, with ``W_t`` the
        total (data times robust) weights at ``state``.
        """
        state = self.cast_state(state)
        Wt = self.method.likelihood.weights_total(Y, W, state.A, state.G)
        dA = jnp.max(Wt @ state.G**2, axis=1)
        dG = jnp.max(Wt.T @ state.A**2, axis=1)
        return np.asarray(dA), np.asarray(dG)

    def coverage(self, Y, W, row_active, col_active) -> tuple[np.ndarray, np.ndarray]:
        """Non-zero weights per row among active columns, and per column among active rows."""
        nz = W > 0
        row_cov = jnp.sum(nz & jnp.asarray(col_active)[None, :], axis=1)
        col_cov = jnp.sum(nz & jnp.asarray(row_active)[:, None], axis=0)
        return np.asarray(row_cov), np.asarray(col_cov)

    def z_moments(self, Y: Array, W: Array, state: RHMFState) -> ZMoments:
        """Held-out moments for ``state`` on ``(Y, W)``.

        Entries in rows or columns the state excludes are scored as zero-weight
        entries, with the excluded factors taken as 0.
        """
        if has_exclusions(state):
            W = mask_weights(W, state.row_excluded, state.col_excluded)
            state = fill_excluded(state, 0.0)
        return z_moments(Y, W, state.A, state.G, self.method.likelihood)

    # -- the loop ------------------------------------------------------------ #

    def run(
        self,
        Y: Array,
        W: Array,
        init_state: RHMFState,
        rotation_cadence: int,
        conv_check_cadence: int,
        max_iter: int,
        skip_G: bool = False,
    ) -> tuple[RHMFState, Array, FitInfo]:
        """Iterate ``step`` until converged or ``max_iter``.

        Returns the final state, the loss at every iteration, and a
        :class:`~robusta_hmf.fit_info.FitInfo` describing the run.

        Rows and columns marked in ``init_state.row_excluded`` /
        ``col_excluded`` get zero weight throughout, so their factors solve to
        zero and they leave the loss; they are returned as NaN, with the masks.
        """
        # Get the step function from the method
        step_fn = self.method.get_stepper()

        # Always rotate for als
        if self.method.opt_method == "als":
            rotation_cadence = 1

        # To store the loss history
        loss_history = []

        # Initialise states and loss
        state = self.cast_state(init_state)
        masked = has_exclusions(state)
        if masked:
            W = mask_weights(W, state.row_excluded, state.col_excluded)
            state = fill_excluded(state, 0.0)
        prev_state = deepcopy(state)
        prev_loss = jnp.inf
        converged = False
        monitor = self.make_monitor(state.A.shape[0], state.A.shape[1], skip_G)
        if monitor.check_rises:
            monitor.data_scale = self.zero_model_loss(Y, W)

        # Training loop
        for i in range(max_iter):
            # Do we rotate this iteration?
            rot = True if (i % rotation_cadence == 0 and i != 0) else False
            # Take an optimization step and record the loss
            state, loss = step_fn(Y=Y, W_data=W, state=state, rotate=rot, skip_G=skip_G)
            loss_history.append(loss)
            # Check convergence and print loss every conv_check_cadence iterations
            if i % conv_check_cadence == 0 and i != 0:
                monitor.update(loss_history)
                if self.conv_tester.is_converged(prev_state, state, prev_loss, loss):
                    print(f"Converged at iteration {i}")
                    converged = True
                    break
                prev_state = deepcopy(state)
                prev_loss = loss
                print(f"iter {state.it:03d} | loss {loss:.4f}", flush=True)
                if not np.isfinite(float(loss)):
                    raise FloatingPointError(nonfinite_loss_message(i, loss, self.compute_dtype))
                check_factors_finite(state.A, state.G, i, self.compute_dtype)

        loss_rises = monitor.finish(loss_history, converged, max_iter, self.conv_tester.strategy)
        if loss_history and not np.isfinite(float(loss_history[-1])):
            last = len(loss_history) - 1
            raise FloatingPointError(
                nonfinite_loss_message(last, loss_history[-1], self.compute_dtype)
            )
        if masked:
            state = fill_excluded(state, jnp.nan)
        info = make_fit_info(converged, loss_history, loss_rises)
        return state, jnp.array(loss_history), info

    def make_monitor(self, n_rows: int, rank: int, infer: bool) -> RunMonitor:
        """The :class:`~robusta_hmf.convergence.RunMonitor` for one run of the loop.

        Loss rises are checked only for ALS without a ridge: a ridge makes ALS
        minimise ``loss + ridge·(‖A‖² + ‖G‖²)``, and with rotation that is not
        monotone either, so a rising data loss says nothing about the numerics.
        """
        return RunMonitor(
            rank=rank,
            scale=getattr(self.method.likelihood, "scale", None),
            check_rises=self.method.opt_method == "als" and self.method.a_step.ridge is None,
            infer=infer,
            n_rows=n_rows,
        )

    def zero_model_loss(self, Y, W) -> float:
        """The loss of the zero model (``A = G = 0``), the scale of the loss's round-off."""
        zeros_A = jnp.zeros((Y.shape[0], 1), Y.dtype)
        zeros_G = jnp.zeros((Y.shape[1], 1), Y.dtype)
        return float(self.method.likelihood.loss(Y, W, zeros_A, zeros_G))


def check_factors_finite(A, G, iteration: int, dtype) -> None:
    """Raise ``FloatingPointError`` if ``A`` or ``G`` holds NaN or inf.

    Called inside the loop, where excluded rows are zero rather than NaN.
    """
    if not (np.isfinite(np.asarray(A)).all() and np.isfinite(np.asarray(G)).all()):
        raise FloatingPointError(nonfinite_factors_message(iteration, dtype))


def make_fit_info(converged: bool, loss_history: list, loss_rises=()) -> FitInfo:
    """The :class:`FitInfo` for a loop that ran ``len(loss_history)`` iterations."""
    return FitInfo(
        converged=converged,
        stop_reason="converged" if converged else "max_iter",
        n_iter=len(loss_history),
        final_loss=float(loss_history[-1]) if loss_history else float("nan"),
        loss_rises=tuple(loss_rises),
    )
