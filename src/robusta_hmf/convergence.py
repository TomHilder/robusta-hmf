# convergence.py

import warnings
from dataclasses import dataclass, field
from typing import Literal, get_args

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .state import RHMFState
from .warnings import ConvergenceWarning

ConvStrategy = Literal["max_frac_G", "max_frac_A", "rel_frac_loss", "none"]

DEFAULT_STRATEGY = "max_frac_G"
DEFAULT_TOL = 1e-6


def max_frac_mat(old_mat: Array, new_mat: Array, tol: float) -> bool:
    d_mat = old_mat - new_mat
    if jnp.max(d_mat * d_mat) / jnp.mean(new_mat * new_mat) < tol:
        # if jnp.max(d_mat * d_mat / (new_mat * new_mat)) < tol:
        return True
    else:
        return False


def rel_frac_loss(old_loss: float, new_loss: float, tol: float) -> bool:
    if jnp.abs(old_loss - new_loss) / jnp.abs(new_loss) < tol:
        return True
    else:
        return False


def never_converged(*args, **kwargs) -> bool:
    return False


@dataclass(frozen=True)
class ConvergenceTester:
    strategy: ConvStrategy = field(default=DEFAULT_STRATEGY)
    tol: float = field(default=DEFAULT_TOL)

    def __post_init__(self):
        if self.strategy not in get_args(ConvStrategy):
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be one of {get_args(ConvStrategy)}."
            )

    def is_converged(
        self,
        old_state: RHMFState,
        new_state: RHMFState,
        old_loss: float,
        new_loss: float,
    ) -> bool:
        if self.strategy == "max_frac_G":
            return max_frac_mat(old_state.G, new_state.G, tol=self.tol)
        elif self.strategy == "max_frac_A":
            return max_frac_mat(old_state.A, new_state.A, tol=self.tol)
        elif self.strategy == "rel_frac_loss":
            return rel_frac_loss(old_loss, new_loss, tol=self.tol)
        elif self.strategy == "none":
            return never_converged()


def rising_loss_iterations(
    losses, eps: float, start: int = 1, data_scale: float = 0.0
) -> list[int]:
    """Iterations ``t ≥ start`` at which the loss rose by more than ALS allows.

    ALS never increases the loss in exact arithmetic, so a rise beyond
    round-off means the iteration has gone numerically wrong. Iteration ``t``
    is flagged when

        L_t − L_{t−1} > max(2·D̄_t, 100·ε·|L_{t−1}| + 1000·ε²·S),
        D̄_t = (L_{t−1−w} − L_{t−1}) / w,  w = min(5, t − 1),  D̄_1 = 0:

    twice the recent average decrease, with a floor that keeps round-off
    from triggering it. ``S`` (``data_scale``) is ``Σ W·Y²``: residuals carry
    round-off of order ``ε·|Y|``, so a loss near zero, as in an exact fit,
    fluctuates on the scale of tens of ``ε²·S`` rather than ``ε·|L|``.
    """
    abs_floor = 1000 * eps**2 * data_scale
    L = np.asarray(losses, dtype=np.float64)
    flagged = []
    with np.errstate(invalid="ignore"):  # inf − inf once the loss has diverged
        for t in range(max(start, 1), L.size):
            w = min(5, t - 1)
            dbar = (L[t - 1 - w] - L[t - 1]) / w if w > 0 else 0.0
            if L[t] - L[t - 1] > max(2 * dbar, 100 * eps * abs(L[t - 1]) + abs_floor):
                flagged.append(t)
    return flagged


class RunMonitor:
    """Watches an optimisation loop's losses and reports on how it ended.

    :meth:`update` runs at each convergence check: it reads the losses
    recorded since the last check and issues one :class:`ConvergenceWarning`
    per run, at the first loss rise (ALS only). :meth:`finish` warns if the
    loop hit ``max_iter`` without converging, and returns every flagged
    iteration.
    """

    def __init__(
        self,
        *,
        rank: int,
        scale,
        check_rises: bool,
        infer: bool,
        n_rows: int,
        data_scale: float = 0.0,
    ):
        head = f"K={rank}" + (f", Q={scale:g}" if scale is not None else "")
        self.head = f"infer on {n_rows:,} rows ({head})" if infer else head
        self.check_rises = check_rises
        self.infer = infer
        self.data_scale = data_scale
        self.losses: list[float] = []
        self.rises: list[int] = []
        self.eps: float | None = None

    def update(self, loss_history: list) -> None:
        start = len(self.losses)
        new = loss_history[start:]
        if not new:
            return
        if self.eps is None:
            dtype = np.asarray(new[0]).dtype
            self.eps = float(np.finfo(dtype if dtype.kind == "f" else np.float64).eps)
        self.losses += [float(x) for x in new]
        if not self.check_rises:
            return
        rises = rising_loss_iterations(
            self.losses, self.eps, start=start, data_scale=self.data_scale
        )
        if rises and not self.rises:
            t = rises[0]
            see = "" if self.infer else " See model.fit_info.loss_rises."
            warnings.warn(
                f"{self.head}: loss rose at iteration {t} ({self.losses[t - 1]:.3g} → "
                f"{self.losses[t]:.3g}), which the ALS iteration should never do; the fit "
                f"is numerically unreliable.{see}",
                ConvergenceWarning,
                stacklevel=4,
            )
        self.rises += rises

    def finish(
        self, loss_history: list, converged: bool, max_iter: int, strategy: str
    ) -> tuple[int, ...]:
        self.update(loss_history)
        if not converged and strategy != "none" and len(self.losses) >= max_iter:
            see = "" if self.infer else "; see model.fit_info"
            warnings.warn(
                f"{self.head}: stopped at max_iter={max_iter} without converging "
                f"(final loss {self.losses[-1]:.5g}){see}",
                ConvergenceWarning,
                stacklevel=4,
            )
        return tuple(self.rises)
