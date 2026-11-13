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


def has_exclusions(state: RHMFState) -> bool:
    """Whether ``state`` carries an exclusion mask."""
    return state.row_excluded is not None or state.col_excluded is not None


def fill_excluded(state: RHMFState, value) -> RHMFState:
    """``state`` with its excluded rows of ``A`` and ``G`` set to ``value``."""
    A, G = state.A, state.G
    if state.row_excluded is not None:
        A = jax.numpy.where(state.row_excluded[:, None], jax.numpy.asarray(value, A.dtype), A)
    if state.col_excluded is not None:
        G = jax.numpy.where(state.col_excluded[:, None], jax.numpy.asarray(value, G.dtype), G)
    return update_state(state, A=A, G=G)


def mask_weights(W: Array, row_excluded: Array | None, col_excluded: Array | None) -> Array:
    """``W`` with excluded rows and columns set to zero weight."""
    if row_excluded is not None:
        W = jax.numpy.where(row_excluded[:, None], jax.numpy.zeros((), W.dtype), W)
    if col_excluded is not None:
        W = jax.numpy.where(col_excluded[None, :], jax.numpy.zeros((), W.dtype), W)
    return W


def working_form(W: Array, state: RHMFState) -> tuple[Array, RHMFState]:
    """``(W, state)`` as computations should see them when ``state`` has exclusions.

    Excluded rows and columns get zero weight and their factors are taken as 0,
    so they behave exactly like missing data: they leave every sum, and their
    robust weights come out as 1. Without exclusions both are returned as is.
    """
    if not has_exclusions(state):
        return W, state
    return mask_weights(W, state.row_excluded, state.col_excluded), fill_excluded(state, 0.0)


def check_state_shape(
    state: RHMFState, n_rows: int | None, n_cols: int, rank: int, what: str = "state"
) -> None:
    """Raise ``ValueError`` unless ``state`` fits data of shape ``(n_rows, n_cols)`` at ``rank``.

    ``n_rows=None`` skips the rows of ``A`` (``infer`` brings new rows).
    """
    expected = [("G", state.G.shape, (n_cols, rank))]
    if state.col_excluded is not None:
        expected.append(("col_excluded", state.col_excluded.shape, (n_cols,)))
    if n_rows is not None:
        expected.append(("A", state.A.shape, (n_rows, rank)))
        if state.row_excluded is not None:
            expected.append(("row_excluded", state.row_excluded.shape, (n_rows,)))
    for name, got, want in expected:
        if tuple(got) != want:
            raise ValueError(
                f"{what}.{name} has shape {tuple(got)}, but the data (shape "
                f"{(n_rows, n_cols) if n_rows is not None else ('N', n_cols)}) at rank={rank} "
                f"need {want}."
            )


def check_state_finite(state: RHMFState) -> None:
    """Raise ``ValueError`` if ``A`` or ``G`` is non-finite outside the exclusion masks."""
    for name, X, mask in (("A", state.A, state.row_excluded), ("G", state.G, state.col_excluded)):
        bad = ~np.isfinite(np.asarray(X)).all(axis=1)
        if mask is not None:
            bad &= ~np.asarray(mask)
        if bad.any():
            raise ValueError(
                f"state contains non-finite values outside its exclusion masks "
                f"({bad.sum()} rows of {name}, first {int(np.flatnonzero(bad)[0])}). Excluded "
                "rows are NaN by design, but only where state.row_excluded / "
                "state.col_excluded mark them."
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
