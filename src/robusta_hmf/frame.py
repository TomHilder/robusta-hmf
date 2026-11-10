# frame.py

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .als import WeightedAStep
from .convergence import ConvergenceTester
from .hmf import HMF
from .initialisation import Initialiser, InitStrategy
from .metrics import ZMoments, z_moments
from .precision import nonfinite_loss_message
from .state import RHMFState, update_state


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

    def prepare(self, Y: Array, W: Array) -> tuple[Any, Any]:
        """Place ``(Y, W)`` however this frame wants them for the loop.

        The base frame converts them to JAX arrays in ``compute_dtype``. The
        return value is fed to both the initialiser and :meth:`run`, so a frame
        needing a resident device layout builds it once here rather than per
        grid point.
        """
        if W is None:
            raise ValueError("W (the inverse-variance weights) is required.")
        return jnp.asarray(Y, self.compute_dtype), jnp.asarray(W, self.compute_dtype)

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

    def initial_coefficients(self, Y: Array, W: Array, G: Array, ridge: float | None) -> Array:
        """One weighted A step against a fixed basis, seeding :meth:`Robusta.infer`."""
        A_dummy = jnp.zeros((Y.shape[0], G.shape[1]), dtype=G.dtype)
        return WeightedAStep(ridge=ridge)(Y, W, RHMFState(A=A_dummy, G=G, it=0)).A

    def z_moments(self, Y: Array, W: Array, state: RHMFState) -> ZMoments:
        """Held-out moments for ``state`` on ``(Y, W)``."""
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
    ):
        # Get the step function from the method
        step_fn = self.method.get_stepper()

        # Always rotate for als
        if self.method.opt_method == "als":
            rotation_cadence = 1

        # To store the loss history
        loss_history = []

        # Initialise states and loss
        state = self.cast_state(init_state)
        prev_state = deepcopy(state)
        prev_loss = jnp.inf

        # Training loop
        for i in range(max_iter):
            # Do we rotate this iteration?
            rot = True if (i % rotation_cadence == 0 and i != 0) else False
            # Take an optimization step and record the loss
            state, loss = step_fn(Y=Y, W_data=W, state=state, rotate=rot, skip_G=skip_G)
            loss_history.append(loss)
            # Check convergence and print loss every conv_check_cadence iterations
            if i % conv_check_cadence == 0 and i != 0:
                if self.conv_tester.is_converged(prev_state, state, prev_loss, loss):
                    print(f"Converged at iteration {i}")
                    break
                prev_state = deepcopy(state)
                prev_loss = loss
                print(f"iter {state.it:03d} | loss {loss:.4f}", flush=True)
                if not np.isfinite(float(loss)):
                    raise FloatingPointError(nonfinite_loss_message(i, loss, self.compute_dtype))

        if loss_history and not np.isfinite(float(loss_history[-1])):
            last = len(loss_history) - 1
            raise FloatingPointError(
                nonfinite_loss_message(last, loss_history[-1], self.compute_dtype)
            )
        return state, jnp.array(loss_history)
