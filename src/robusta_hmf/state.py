# state.py

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import optax
from jaxtyping import Array


def _as_mask(x):
    return None if x is None else jax.numpy.asarray(x, dtype=bool)


class RHMFState(eqx.Module):
    """Factors ``A`` (N, K) and ``G`` (M, K) of ``Y ≈ A Gᵀ``, plus loop bookkeeping.

    ``row_excluded`` (N,) and ``col_excluded`` (M,) mark rows of ``A`` and
    rows of ``G`` that a fit excluded because the data did not determine them;
    those rows are NaN. ``None`` means nothing is excluded.
    """

    A: Array = eqx.field(converter=jax.numpy.asarray)
    G: Array = eqx.field(converter=jax.numpy.asarray)
    it: int = eqx.field(converter=jax.numpy.asarray, default=0)
    opt_state: optax.OptState | None = eqx.field(default=None)
    row_excluded: Array | None = eqx.field(converter=_as_mask, default=None)
    col_excluded: Array | None = eqx.field(converter=_as_mask, default=None)


def update_state(
    state: RHMFState,
    A: Array | None = None,
    G: Array | None = None,
    it: int | None = None,
    opt_state: optax.OptState | None = None,
) -> RHMFState:
    A_ = state.A if A is None else A
    G_ = state.G if G is None else G
    it_ = state.it if it is None else it
    opt_ = state.opt_state if opt_state is None else opt_state
    return eqx.tree_at(
        lambda s: (s.A, s.G, s.it, s.opt_state),
        state,
        (A_, G_, it_, opt_),
        is_leaf=lambda x: x is None,
    )


def with_exclusions(
    state: RHMFState, row_excluded: Array | None, col_excluded: Array | None
) -> RHMFState:
    """Return ``state`` with its exclusion masks replaced (``None`` clears one)."""
    return eqx.tree_at(
        lambda s: (s.row_excluded, s.col_excluded),
        state,
        (_as_mask(row_excluded), _as_mask(col_excluded)),
        is_leaf=lambda x: x is None,
    )


def refresh_opt_state(state: RHMFState, opt: optax.GradientTransformation) -> RHMFState:
    opt_state = opt.init((state.A, state.G))
    state = eqx.tree_at(lambda s: s.opt_state, state, opt_state)
    return state


def save_state_to_npz(state: RHMFState, filepath: Path):
    """Save ``A``, ``G``, ``it`` and any exclusion masks (not the optimiser state)."""
    masks = {
        name: np.asarray(mask)
        for name, mask in (
            ("row_excluded", state.row_excluded),
            ("col_excluded", state.col_excluded),
        )
        if mask is not None
    }
    np.savez(filepath, A=state.A, G=state.G, it=state.it, **masks)


def load_state_from_npz(
    filepath: Path,
    opt_state: optax.OptState | None = None,
) -> RHMFState:
    """Load a state saved by :func:`save_state_to_npz`; absent masks load as ``None``."""
    data = np.load(filepath)
    return RHMFState(
        A=data["A"],
        G=data["G"],
        it=int(data["it"]),
        opt_state=opt_state,
        row_excluded=data["row_excluded"] if "row_excluded" in data.files else None,
        col_excluded=data["col_excluded"] if "col_excluded" in data.files else None,
    )
