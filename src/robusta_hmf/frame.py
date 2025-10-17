# frame.py

from copy import deepcopy
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from .convergence import ConvergenceTester
from .hmf import HMF
from .state import RHMFState


@dataclass
class OptFrame:
    method: HMF
    conv_tester: ConvergenceTester

    def run(
        self,
        Y: Array,
        W: Array,
        init_state: RHMFState,
        rotation_cadence: int,
        conv_check_cadence: int,
        max_iter: int,
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
            state, loss = step_fn(Y=Y, W_data=W, state=state, rotate=rot)
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
