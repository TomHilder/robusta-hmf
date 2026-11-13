# coverage.py

"""Coverage of rows and columns, and the exclusion cascade.

The coverage of a row or column is its number of entries with non-zero weight.
A row of ``A`` (or of ``G``) is determined by the data only if its coverage is
at least the rank ``K``; otherwise its least-squares system is singular. Such
rows and columns are excluded from the fit. Excluding them lowers the coverage
of the columns and rows they cross, so exclusion is repeated until nothing
changes (the cascade).
"""

from collections.abc import Callable

import numpy as np

from .fit_info import ExclusionEvent

CoverageFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def cascade(
    coverage: CoverageFn,
    rank: int,
    row_active: np.ndarray,
    col_active: np.ndarray,
    *,
    update_cols: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[ExclusionEvent]]:
    """Exclude rows and columns with coverage below ``rank`` until nothing changes.

    Parameters
    ----------
    coverage : callable
        ``coverage(row_active, col_active) -> (row_cov, col_cov)``: the number
        of non-zero weights in each row among active columns, and in each
        column among active rows.
    rank : int
        The rank ``K``.
    row_active, col_active : numpy.ndarray of bool
        Rows and columns still in the fit at the start.
    update_cols : bool, default=True
        If False, columns are held fixed and only rows are excluded (inference,
        where ``G`` does not change).

    Returns
    -------
    row_active, col_active : numpy.ndarray of bool
        What remains.
    events : list of ExclusionEvent
        In the first round, cause ``"empty"`` (coverage 0) or
        ``"low_coverage"``; in later rounds, ``"cascade"``.
    """
    row_active = np.array(row_active, dtype=bool)
    col_active = np.array(col_active, dtype=bool)
    events: list[ExclusionEvent] = []
    first_round = True
    while True:
        row_cov, col_cov = coverage(row_active, col_active)
        new_rows = row_active & (np.asarray(row_cov) < rank)
        new_cols = col_active & (np.asarray(col_cov) < rank)
        if not update_cols:
            new_cols[:] = False
        if not (new_rows.any() or new_cols.any()):
            return row_active, col_active, events
        for axis, new, cov in (("row", new_rows, row_cov), ("col", new_cols, col_cov)):
            if first_round:
                empty = new & (np.asarray(cov) == 0)
                events += _events(axis, empty, "empty")
                events += _events(axis, new & ~empty, "low_coverage")
            else:
                events += _events(axis, new, "cascade")
        row_active &= ~new_rows
        col_active &= ~new_cols
        first_round = False


def under_covered(
    coverage: CoverageFn, rank: int, row_active: np.ndarray, col_active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of active rows and columns with coverage below ``rank``, without excluding any."""
    row_cov, col_cov = coverage(np.asarray(row_active, bool), np.asarray(col_active, bool))
    rows = np.flatnonzero(np.asarray(row_active, bool) & (np.asarray(row_cov) < rank))
    cols = np.flatnonzero(np.asarray(col_active, bool) & (np.asarray(col_cov) < rank))
    return rows, cols


def _events(axis, mask, cause) -> list[ExclusionEvent]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return [ExclusionEvent(axis=axis, indices=tuple(int(i) for i in idx), cause=cause)]


def indices_by_axis(events) -> tuple[np.ndarray, np.ndarray]:
    """All row indices and all column indices named in ``events``, sorted."""
    rows = [np.asarray(e.indices, int) for e in events if e.axis == "row"]
    cols = [np.asarray(e.indices, int) for e in events if e.axis == "col"]
    return (
        np.sort(np.concatenate(rows)) if rows else np.zeros(0, int),
        np.sort(np.concatenate(cols)) if cols else np.zeros(0, int),
    )


def _list(idx: np.ndarray, n_show: int = 5) -> str:
    shown = ", ".join(str(int(i)) for i in idx[:n_show])
    return shown + (", …" if idx.size > n_show else "")


def describe(rows: np.ndarray, cols: np.ndarray) -> tuple[str, str]:
    """``("3 rows and 352 columns", "rows 12, 40, 41; columns 1000, 1001, …")``."""
    counts, lists = [], []
    for name, idx in (("row", rows), ("column", cols)):
        if idx.size:
            counts.append(f"{idx.size:,} {name}{'s' if idx.size != 1 else ''}")
            lists.append(f"{name}{'s' if idx.size != 1 else ''} {_list(idx)}")
    return " and ".join(counts), "; ".join(lists)


def exclusion_message(rank: int, rows: np.ndarray, cols: np.ndarray, *, infer: bool) -> str:
    """Warning text for rows and columns excluded before a fit or an inference."""
    counts, lists = describe(rows, cols)
    if infer:
        return (
            f"infer (K={rank}): {counts} {'has' if rows.size == 1 else 'have'} fewer than "
            f"{rank} non-zero weights in the columns the model uses ({lists}). Excluded "
            "them; their rows of A are returned as NaN (see state.row_excluded)."
        )
    verb = "has" if rows.size + cols.size == 1 else "have"
    return (
        f"K={rank}: {counts} {verb} fewer than {rank} non-zero weights, directly or "
        f"after other exclusions ({lists}). Excluded them from the fit; their entries "
        "in A and G are returned as NaN. See model.fit_info.exclusions."
    )


def ridge_message(rank: int, rows: np.ndarray, cols: np.ndarray, ridge: float) -> str:
    """Warning text for under-covered rows and columns whose factors the ridge determines."""
    counts, lists = describe(rows, cols)
    verb = "has" if rows.size + cols.size == 1 else "have"
    return (
        f"K={rank}: {counts} {verb} fewer than {rank} non-zero weights ({lists}); "
        f"als_ridge={ridge:g} determines their factors, not the data."
    )


#: A ridge is negligible next to a system when ridge ≤ NEGLIGIBLE_RIDGE_FACTOR·ε·d.
NEGLIGIBLE_RIDGE_FACTOR = 10


def negligible_ridge(ridge: float, eps: float, diagonals: np.ndarray) -> np.ndarray:
    """Which systems a ridge cannot affect: ``ridge ≤ 10·ε·d``, d their largest diagonal."""
    return ridge <= NEGLIGIBLE_RIDGE_FACTOR * eps * np.asarray(diagonals)


def negligible_ridge_message(ridge, dtype, rows, cols, d_max) -> str:
    """Warning text for a ridge below the resolution of the systems it should determine."""
    counts, _ = describe(rows, cols)
    eps = float(np.finfo(dtype).eps)
    return (
        f"als_ridge={ridge:g} is below {np.dtype(dtype).name} resolution for {counts} with "
        f"fewer than K non-zero weights (largest diagonal {d_max:.1e}, so the ridge must "
        f"exceed ~{NEGLIGIBLE_RIDGE_FACTOR * eps * d_max:.1e} to have any effect); it is not "
        "determining their factors. Increase als_ridge, use float64, or set als_ridge=None "
        "to exclude them instead."
    )
