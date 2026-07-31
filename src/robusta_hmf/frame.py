# frame.py

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from .als import WeightedAStep
from .convergence import ConvergenceTester
from .hmf import HMF
from .initialisation import Initialiser, InitStrategy
from .metrics import ZMoments, z_moments
from .state import RHMFState


@dataclass
class OptFrame:
    """Owns the optimisation loop: data placement, initialisation, iteration.

    :class:`~robusta_hmf.Robusta` configures the model (an :class:`HMF`) and
    delegates execution to a frame. A subclass can change *how* the loop is
    scheduled -- see :class:`robusta_hmf.sharded.ShardedFrame`, which chunks
    and shards the row axis -- without re-declaring the model's parameter list.
    """

    method: HMF
    conv_tester: ConvergenceTester

    def bind(self, method: HMF, conv_tester: ConvergenceTester) -> "OptFrame":
        """Return a copy of this frame wired to a model and a convergence test.

        Lets a user construct a frame with only its scheduling parameters --
        ``ShardedFrame(row_block=4096)`` -- and have ``Robusta`` fill in the
        rest, so a model is still configured in exactly one place.
        """
        return replace(self, method=method, conv_tester=conv_tester)

    # -- hooks that a scheduling subclass overrides -------------------------- #

    def prepare(self, Y: Array, W: Array) -> tuple[Any, Any]:
        """Place ``(Y, W)`` however this frame wants them for the loop.

        The base frame passes them through unchanged. The return value is fed
        to both the initialiser and :meth:`run`, so a frame needing a resident
        device layout builds it once here rather than per grid point.
        """
        return Y, W

    def make_initialiser(self, N: int, M: int, K: int, strategy: InitStrategy) -> Initialiser:
        """The initialiser this frame can consume, when none was overridden."""
        return Initialiser(N=N, M=M, K=K, strategy=strategy)

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
        state = init_state
        prev_state = deepcopy(init_state)
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

        return state, jnp.array(loss_history)
