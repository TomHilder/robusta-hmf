# fit_info.py

"""Diagnostics describing how a fit ran."""

from dataclasses import dataclass, field
from typing import Literal

StopReason = Literal["converged", "max_iter"]
ExclusionAxis = Literal["row", "col"]
ExclusionCause = Literal["empty", "low_coverage", "cascade"]


@dataclass(frozen=True)
class ExclusionEvent:
    """Rows or columns removed from a fit because the data could not determine them.

    Exclusion happens before the first iteration, from the weights.

    Attributes
    ----------
    axis : {"row", "col"}
        Whether ``indices`` are rows (of ``Y`` and ``A``) or columns (of ``Y``,
        rows of ``G``).
    indices : tuple of int
        The excluded indices, sorted.
    cause : {"empty", "low_coverage", "cascade"}
        ``"empty"``: no non-zero weights. ``"low_coverage"``: fewer non-zero
        weights than the rank. ``"cascade"``: fell below the rank after other
        rows or columns were excluded.
    """

    axis: ExclusionAxis
    indices: tuple[int, ...]
    cause: ExclusionCause


@dataclass(frozen=True)
class FitInfo:
    """How a call to :meth:`robusta_hmf.Robusta.fit` ran.

    Attributes
    ----------
    converged : bool
        Whether the convergence test passed.
    stop_reason : {"converged", "max_iter"}
        Why the loop ended.
    n_iter : int
        Number of iterations run.
    final_loss : float
        Loss at the last iteration (NaN if no iteration ran).
    loss_rises : tuple of int
        Iterations at which the loss rose by more than the optimiser allows.
    exclusions : tuple of ExclusionEvent
        Every exclusion, in the order it happened.
    """

    converged: bool
    stop_reason: StopReason
    n_iter: int
    final_loss: float
    loss_rises: tuple[int, ...] = field(default=())
    exclusions: tuple[ExclusionEvent, ...] = field(default=())
