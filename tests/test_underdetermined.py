# test_underdetermined.py
#
# Under-determined and non-converging fits: input validation, coverage and
# exclusion, convergence reporting, and the scaffolding they report through
# (warning classes, FitInfo, exclusion masks on RHMFState).

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_low_rank_problem

import robusta_hmf
from robusta_hmf import (
    ConvergenceWarning,
    DataWarning,
    ExclusionEvent,
    FitInfo,
    Robusta,
    RobustaWarning,
    ShardedFrame,
    UnderdeterminedWarning,
    load_state_from_npz,
    save_state_to_npz,
)
from robusta_hmf.state import RHMFState, update_state, with_exclusions

jax.config.update("jax_enable_x64", True)

RANK = 3


@pytest.fixture(scope="module")
def problem():
    return make_low_rank_problem(N=60, M=30, K=RANK)


# ----------------------------
# Warning classes
# ----------------------------
@pytest.mark.parametrize("cls", [DataWarning, UnderdeterminedWarning, ConvergenceWarning])
def test_warnings_share_a_filterable_base(cls):
    assert issubclass(cls, RobustaWarning)
    assert issubclass(RobustaWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RobustaWarning)
        with pytest.raises(cls):
            warnings.warn("x", cls)


def test_new_names_are_exported_from_the_package_root():
    for name in [
        "RobustaWarning",
        "DataWarning",
        "UnderdeterminedWarning",
        "ConvergenceWarning",
        "FitInfo",
        "ExclusionEvent",
    ]:
        assert name in robusta_hmf.__all__
        assert hasattr(robusta_hmf, name)


# ----------------------------
# FitInfo / ExclusionEvent
# ----------------------------
def test_fit_info_defaults_and_immutability():
    info = FitInfo(converged=True, stop_reason="converged", n_iter=3, final_loss=1.0)
    assert info.loss_rises == ()
    assert info.exclusions == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.converged = False
    event = ExclusionEvent(axis="col", indices=(2, 5), cause="empty")
    assert event.axis == "col" and event.indices == (2, 5) and event.cause == "empty"
    # Comparable and hashable, so FitInfo is too.
    assert event == ExclusionEvent(axis="col", indices=(2, 5), cause="empty")
    assert len({event, ExclusionEvent(axis="col", indices=(2, 5), cause="empty")}) == 1
    a = FitInfo(True, "converged", 3, 1.0, exclusions=(event,))
    assert a == FitInfo(True, "converged", 3, 1.0, exclusions=(event,))


# ----------------------------
# Exclusion masks on RHMFState
# ----------------------------
def _state(N=6, M=5, K=RANK):
    return RHMFState(A=jnp.ones((N, K)), G=jnp.ones((M, K)), it=2)


def test_state_masks_default_to_none():
    state = _state()
    assert state.row_excluded is None
    assert state.col_excluded is None


def test_state_masks_are_converted_to_bool():
    state = RHMFState(
        A=jnp.ones((3, RANK)), G=jnp.ones((4, RANK)), row_excluded=[0, 1, 0], col_excluded=None
    )
    assert state.row_excluded.dtype == jnp.bool_
    assert state.row_excluded.tolist() == [False, True, False]


def test_with_exclusions_sets_and_clears_masks():
    state = with_exclusions(_state(), np.array([0, 0, 1, 0, 0, 0]), None)
    assert state.row_excluded.tolist() == [False, False, True, False, False, False]
    assert state.col_excluded is None
    assert int(state.it) == 2
    cleared = with_exclusions(state, None, None)
    assert cleared.row_excluded is None


def test_update_state_carries_masks():
    state = with_exclusions(_state(), np.eye(6)[0], np.eye(5)[4])
    updated = update_state(state, A=jnp.zeros((6, RANK)), it=9)
    assert updated.row_excluded.tolist() == state.row_excluded.tolist()
    assert updated.col_excluded.tolist() == state.col_excluded.tolist()
    assert int(updated.it) == 9


def test_npz_roundtrip_keeps_masks(tmp_path):
    state = with_exclusions(_state(), np.eye(6)[1], np.eye(5)[3])
    path = tmp_path / "s.npz"
    save_state_to_npz(state, path)
    loaded = load_state_from_npz(path)
    assert loaded.row_excluded.tolist() == state.row_excluded.tolist()
    assert loaded.col_excluded.tolist() == state.col_excluded.tolist()


def test_npz_without_masks_loads_as_none(tmp_path):
    # A file written before masks existed has only A, G and it.
    path = tmp_path / "old.npz"
    np.savez(path, A=np.ones((6, RANK)), G=np.ones((5, RANK)), it=4)
    loaded = load_state_from_npz(path)
    assert loaded.row_excluded is None and loaded.col_excluded is None
    assert int(loaded.it) == 4


def test_npz_without_masks_writes_no_mask_keys(tmp_path):
    path = tmp_path / "s.npz"
    save_state_to_npz(_state(), path)
    assert set(np.load(path).files) == {"A", "G", "it"}


# ----------------------------
# model.fit_info
# ----------------------------
def test_fit_info_none_before_fit():
    assert Robusta(rank=RANK).fit_info is None


@pytest.mark.parametrize("sharded", [False, True])
def test_fit_info_after_converged_fit(problem, sharded):
    frame = ShardedFrame(row_block=16, verbose=False) if sharded else None
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-4, override_frame=frame)
    _, loss_history = model.fit(problem["Y"], problem["W"], max_iter=500)
    info = model.fit_info
    assert info.converged is True
    assert info.stop_reason == "converged"
    assert info.n_iter == len(loss_history) < 500
    assert info.final_loss == float(loss_history[-1])
    assert info.loss_rises == () and info.exclusions == ()


@pytest.mark.parametrize("sharded", [False, True])
def test_fit_info_after_max_iter(problem, sharded):
    frame = ShardedFrame(row_block=16, verbose=False) if sharded else None
    model = Robusta(rank=RANK, robust=False, conv_strategy="none", override_frame=frame)
    _, loss_history = model.fit(problem["Y"], problem["W"], max_iter=7)
    info = model.fit_info
    assert info.converged is False
    assert info.stop_reason == "max_iter"
    assert info.n_iter == 7
    assert info.final_loss == float(loss_history[-1])


def test_infer_does_not_touch_fit_info(problem):
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-4)
    model.fit(problem["Y"], problem["W"], max_iter=500)
    before = model.fit_info
    model.infer(problem["Y"][:5], problem["W"][:5], conv_strategy="none", max_iter=3)
    assert model.fit_info is before
    assert not hasattr(model, "infer_info")


# ----------------------------
# Input validation (non-finite entries, negative weights, rank vs shape)
# ----------------------------


def _frame(sharded):
    return (
        ShardedFrame(row_block=16, devices=jax.devices("cpu")[:1], verbose=False)
        if sharded
        else None
    )


def _model(sharded=False, **kw):
    conv = kw.pop("conv_strategy", "none")
    return Robusta(
        rank=RANK, robust=False, conv_strategy=conv, override_frame=_frame(sharded), **kw
    )


def _data_warnings(record):
    return [w for w in record if issubclass(w.category, DataWarning)]


BAD_ENTRIES = [
    ("Y", np.nan),
    ("W", np.nan),
    ("Y", np.inf),
    ("Y", -np.inf),
    ("W", np.inf),
    ("W", -np.inf),
]


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("which,value", BAD_ENTRIES)
def test_nonfinite_entries_are_treated_as_missing(problem, sharded, which, value):
    Y, W = problem["Y"].copy(), problem["W"].copy()
    bad = [(3, 4), (20, 0), (41, 29)]  # rows 20 and 41 fall in later row blocks
    for r, c in bad:
        (Y if which == "Y" else W)[r, c] = value
    Y_ref, W_ref = Y.copy(), W.copy()
    for r, c in bad:
        Y_ref[r, c] = W_ref[r, c] = 0.0

    with pytest.warns(DataWarning) as record:
        state, loss = _model(sharded).fit(Y, W, max_iter=15)
    msgs = [str(w.message) for w in _data_warnings(record)]
    assert len(msgs) == 1
    assert "Found 3 non-finite entries" in msgs[0]
    assert "first at row 3, column 4" in msgs[0]
    assert "nan_error=True" in msgs[0]

    ref_state, ref_loss = _model(sharded).fit(Y_ref, W_ref, max_iter=15)
    np.testing.assert_array_equal(np.asarray(state.A), np.asarray(ref_state.A))
    np.testing.assert_array_equal(np.asarray(state.G), np.asarray(ref_state.G))
    np.testing.assert_array_equal(np.asarray(loss), np.asarray(ref_loss))


@pytest.mark.parametrize("sharded", [False, True])
def test_nan_where_weight_is_zero_gives_finite_fit(problem, sharded):
    # The case in issue #2: NaN flux on pixels that are already masked.
    Y, W = problem["Y"].copy(), problem["W"].copy()
    W[:, 7] = 0.0
    W[2, :] = 0.0
    Y[:, 7] = np.nan
    Y[5, 9] = np.nan
    W[5, 9] = 0.0
    with pytest.warns(DataWarning), pytest.warns(UnderdeterminedWarning, match="row 2; column 7"):
        state, loss = _model(sharded).fit(Y, W, max_iter=15)
    assert np.all(np.isfinite(np.asarray(loss)))
    rows = np.ones(Y.shape[0], bool)
    rows[2] = False
    cols = np.ones(Y.shape[1], bool)
    cols[7] = False
    assert np.all(np.isfinite(np.asarray(state.A)[rows]))
    assert np.all(np.isfinite(np.asarray(state.G)[cols]))


@pytest.mark.parametrize("sharded", [False, True])
def test_nan_error_raises_with_count_and_position(problem, sharded):
    Y = problem["Y"].copy()
    Y[33, 12] = np.nan
    Y[40, 2] = np.inf
    model = _model(sharded, nan_error=True)
    with pytest.raises(ValueError, match=r"2 non-finite entries .*first at row 33, column 12"):
        model.fit(Y, problem["W"], max_iter=15)
    assert model.fit_info is None and model.state is None


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("method", ["infer", "score"])
def test_infer_and_score_validate_too(problem, sharded, method):
    model = _model(sharded)
    model.fit(problem["Y"], problem["W"], max_iter=15)
    # score() evaluates the fitted state, so it needs the training rows.
    n = 20 if method == "infer" else problem["Y"].shape[0]
    Y = problem["Y"][:n].copy()
    W = problem["W"][:n].copy()
    Y[1, 1] = np.nan
    call = getattr(model, method)
    with pytest.warns(DataWarning, match="Found 1 non-finite"):
        call(Y, W)
    model.nan_error = True
    with pytest.raises(ValueError, match="1 non-finite"):
        call(Y, W)
    W[0, 0] = -1.0
    model.nan_error = False
    with pytest.raises(ValueError, match="negative"):
        call(problem["Y"][:n], W)


@pytest.mark.parametrize("sharded", [False, True])
def test_negative_weights_raise(problem, sharded):
    W = problem["W"].copy()
    W[17, 3] = -0.5
    with pytest.raises(ValueError, match=r"1 negative entries in W \(first at row 17, column 3\)"):
        _model(sharded).fit(problem["Y"], W, max_iter=15)


def test_sharded_handle_is_zeroed_in_both_arrays(problem):
    frame = _frame(True)
    Y, W = problem["Y"].copy(), problem["W"].copy()
    Y[20, 5] = np.nan  # second row block
    W[3, 8] = np.inf
    with pytest.warns(DataWarning):
        data, _ = frame.prepare(Y, W)
    Ys, Ws = np.asarray(data.Y), np.asarray(data.W)
    for r, c in [(20, 5), (3, 8)]:
        assert Ys[r, c] == 0 and Ws[r, c] == 0
    # Everything else is untouched (up to the float32 store).
    keep = np.ones(Y.shape, bool)
    keep[20, 5] = keep[3, 8] = False
    n = Y.shape[0]
    np.testing.assert_array_equal(Ys[:n][keep], Y.astype(np.float32)[keep])
    np.testing.assert_array_equal(Ws[:n][keep], W.astype(np.float32)[keep])


def test_value_overflowing_float32_is_treated_as_missing(problem):
    Y = problem["Y"].copy()
    Y[6, 6] = 1e60  # finite in float64, inf in float32
    with pytest.warns(DataWarning, match="Found 1 non-finite"):
        _, loss = _model(dtype="float32").fit(Y, problem["W"], max_iter=15)
    assert np.all(np.isfinite(np.asarray(loss)))


def test_clean_data_gives_no_data_warning(problem):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataWarning)
        _model().fit(problem["Y"], problem["W"], max_iter=15)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("N,M,which", [(2, 30, "rows"), (60, 2, "columns")])
def test_rank_above_shape_raises_in_fit(sharded, N, M, which):
    rng = np.random.default_rng(0)
    model = _model(sharded)
    with pytest.raises(
        ValueError, match=f"rank=3 is larger than the number of {which}.*shape mistake"
    ):
        model.fit(rng.normal(size=(N, M)), np.ones((N, M)), max_iter=2)


def test_transposed_data_is_caught(problem):
    with pytest.raises(ValueError, match="transposed"):
        _model().fit(problem["Y"][:, :2].T, problem["W"][:, :2].T, max_iter=2)


@pytest.mark.parametrize("sharded", [False, True])
def test_infer_allows_fewer_rows_than_rank(problem, sharded):
    model = _model(sharded)
    model.fit(problem["Y"], problem["W"], max_iter=15)
    state, _ = model.infer(problem["Y"][:1], problem["W"][:1], max_iter=5)
    assert state.A.shape == (1, RANK)
    assert np.all(np.isfinite(np.asarray(state.A)))
    with pytest.raises(ValueError, match="number of columns"):
        model.infer(problem["Y"][:5, :2], problem["W"][:5, :2])


def test_dense_shape_mismatch_raises(problem):
    with pytest.raises(ValueError, match="same shape"):
        _model().fit(problem["Y"], problem["W"][:, :-1], max_iter=2)


# ----------------------------
# Static coverage, the cascade, and what is returned
# ----------------------------
def _underdetermined(record):
    return [str(w.message) for w in record if issubclass(w.category, UnderdeterminedWarning)]


def _dead_row_and_col(problem, row=4, col=5):
    W = problem["W"].copy()
    W[row, :] = 0.0
    W[:, col] = 0.0
    return problem["Y"], W


def _cascade_weights(N=40, M=20):
    """K=3: column 0 has 2 weights; dropping it leaves row 1 with 2 (cols 1, 2);
    dropping row 1 leaves column 1 with 2 (rows 3, 4)."""
    W = np.ones((N, M))
    W[:, 0] = 0.0
    W[[1, 2], 0] = 1.0
    W[1, :] = 0.0
    W[1, [0, 1, 2]] = 1.0
    W[:, 1] = 0.0
    W[[1, 3, 4], 1] = 1.0
    return W


@pytest.mark.parametrize("sharded", [False, True])
def test_empty_row_and_column_are_excluded(problem, sharded):
    Y, W = _dead_row_and_col(problem)
    model = _model(sharded)
    with pytest.warns(UnderdeterminedWarning) as record:
        state, loss = model.fit(Y, W, max_iter=15)
    msgs = _underdetermined(record)
    assert len(msgs) == 1
    assert "1 row and 1 column" in msgs[0] and "rows 4" not in msgs[0]
    assert "row 4; column 5" in msgs[0]
    A, G = np.asarray(state.A), np.asarray(state.G)
    assert np.isnan(A[4]).all() and np.isnan(G[5]).all()
    assert np.isfinite(np.delete(A, 4, 0)).all() and np.isfinite(np.delete(G, 5, 0)).all()
    assert np.flatnonzero(np.asarray(state.row_excluded)).tolist() == [4]
    assert np.flatnonzero(np.asarray(state.col_excluded)).tolist() == [5]
    assert np.isfinite(np.asarray(loss)).all()
    events = model.fit_info.exclusions
    assert events == (
        ExclusionEvent(axis="row", indices=(4,), cause="empty"),
        ExclusionEvent(axis="col", indices=(5,), cause="empty"),
    )


def test_low_coverage_column_is_excluded(problem):
    W = problem["W"].copy()
    W[:, 9] = 0.0
    W[[0, 7], 9] = 1.0  # coverage 2 < K = 3
    model = _model()
    with pytest.warns(UnderdeterminedWarning):
        state, _ = model.fit(problem["Y"], W, max_iter=15)
    [event] = model.fit_info.exclusions
    assert event == ExclusionEvent(axis="col", indices=(9,), cause="low_coverage")
    assert np.isnan(np.asarray(state.G)[9]).all()
    assert state.row_excluded is None


def test_coverage_exactly_rank_is_kept(problem):
    W = problem["W"].copy()
    W[:, 9] = 0.0
    W[[0, 7, 11], 9] = 1.0  # coverage 3 == K
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnderdeterminedWarning)
        state, _ = _model().fit(problem["Y"], W, max_iter=15)
    assert state.col_excluded is None and state.row_excluded is None
    assert np.isfinite(np.asarray(state.G)).all()


@pytest.mark.parametrize("sharded", [False, True])
def test_cascade(sharded):
    rng = np.random.default_rng(3)
    W = _cascade_weights()
    Y = rng.normal(size=W.shape)
    model = _model(sharded)
    with pytest.warns(UnderdeterminedWarning) as record:
        state, _ = model.fit(Y, W, max_iter=5)
    assert len(_underdetermined(record)) == 1
    events = [(e.axis, e.cause, list(e.indices)) for e in model.fit_info.exclusions]
    assert events == [
        ("col", "low_coverage", [0]),
        ("row", "cascade", [1]),
        ("col", "cascade", [1]),
    ]
    assert np.flatnonzero(np.asarray(state.row_excluded)).tolist() == [1]
    assert np.flatnonzero(np.asarray(state.col_excluded)).tolist() == [0, 1]


def test_cascade_that_removes_everything_raises():
    rng = np.random.default_rng(0)
    W = np.zeros((10, 10))
    for i in range(10):  # every row and column has exactly 2 weights
        W[i, [i, (i + 1) % 10]] = 1.0
    with pytest.raises(ValueError, match="nothing left to fit"):
        _model().fit(rng.normal(size=W.shape), W, max_iter=5)


@pytest.mark.parametrize("sharded", [False, True])
def test_exclusion_equals_removing_by_hand(problem, sharded):
    """Excluding rows and columns is the same as fitting without them."""
    W = _cascade_weights(N=60, M=30) * problem["W"]
    Y = problem["Y"]
    kw = {"conv_strategy": "max_frac_G", "conv_tol": 1e-14}
    with pytest.warns(UnderdeterminedWarning):
        state, loss = _model(sharded, **kw).fit(Y, W, max_iter=2000)
    rows = ~np.asarray(state.row_excluded)
    cols = ~np.asarray(state.col_excluded)
    ref, ref_loss = _model(sharded, **kw).fit(Y[rows][:, cols], W[rows][:, cols], max_iter=2000)
    model_fit = np.asarray(state.A)[rows] @ np.asarray(state.G)[cols].T
    model_ref = np.asarray(ref.A) @ np.asarray(ref.G).T
    np.testing.assert_allclose(model_fit, model_ref, rtol=0, atol=1e-8)
    # The loss counts only the entries left in the fit.
    np.testing.assert_allclose(float(loss[-1]), float(ref_loss[-1]), rtol=1e-10)


@pytest.mark.parametrize("sharded", [False, True])
def test_masked_fit_tracks_reduced_fit_every_iteration(problem, sharded):
    """From the same start, excluded rows and columns take no part in any step.

    The final optimum alone cannot show this: an under-determined factor can
    absorb its own few entries exactly, leaving the rest of the fit unchanged.
    The path differs, though, from the first loss on.
    """
    W = _cascade_weights(N=60, M=30) * problem["W"]
    Y = problem["Y"]
    rng = np.random.default_rng(5)
    A0, G0 = rng.normal(size=(60, RANK)), rng.normal(size=(30, RANK))
    with pytest.warns(UnderdeterminedWarning):
        state, loss = _model(sharded).fit(Y, W, max_iter=20, init_state=RHMFState(A=A0, G=G0))
    rows = ~np.asarray(state.row_excluded)
    cols = ~np.asarray(state.col_excluded)
    _, ref_loss = _model(sharded).fit(
        Y[rows][:, cols],
        W[rows][:, cols],
        max_iter=20,
        init_state=RHMFState(A=A0[rows], G=G0[cols]),
    )
    np.testing.assert_allclose(np.asarray(loss), np.asarray(ref_loss), rtol=1e-9)


def test_ridge_reports_but_does_not_exclude(problem):
    Y, W = _dead_row_and_col(problem)
    with pytest.warns(UnderdeterminedWarning, match="als_ridge=1e-06 determines") as record:
        state, _ = _model(als_ridge=1e-6).fit(Y, W, max_iter=15)
    assert len(_underdetermined(record)) == 1
    assert state.row_excluded is None and state.col_excluded is None
    assert np.isfinite(np.asarray(state.A)).all() and np.isfinite(np.asarray(state.G)).all()
    assert np.allclose(np.asarray(state.A)[4], 0) and np.allclose(np.asarray(state.G)[5], 0)


@pytest.mark.parametrize("sharded", [False, True])
def test_full_coverage_has_no_masks_and_no_warning(problem, sharded):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnderdeterminedWarning)
        state, _ = _model(sharded).fit(problem["Y"], problem["W"], max_iter=15)
    assert state.row_excluded is None and state.col_excluded is None


def test_sgd_honours_static_exclusions(problem):
    Y, W = _dead_row_and_col(problem)
    model = Robusta(rank=RANK, method="sgd", robust=False, conv_strategy="none")
    with pytest.warns(UnderdeterminedWarning):
        state, loss = model.fit(Y, W, max_iter=10)
    A, G = np.asarray(state.A), np.asarray(state.G)
    assert np.isnan(A[4]).all() and np.isnan(G[5]).all()
    assert np.isfinite(np.delete(A, 4, 0)).all() and np.isfinite(np.delete(G, 5, 0)).all()
    assert np.isfinite(np.asarray(loss)).all()


@pytest.fixture(scope="module")
def excluded_fit(problem):
    Y, W = _dead_row_and_col(problem)
    model = _model()
    with pytest.warns(UnderdeterminedWarning):
        state, _ = model.fit(Y, W, max_iter=15)
    return model, state, Y, W


def test_accessors_are_nan_exactly_on_exclusions(excluded_fit):
    model, _, Y, _ = excluded_fit
    nan_rows = np.zeros(Y.shape, bool)
    nan_rows[4, :] = True
    nan_rows[:, 5] = True
    for X in (model.synthesize(), model.residuals(Y)):
        np.testing.assert_array_equal(np.isnan(np.asarray(X)), nan_rows)
    assert np.isnan(np.asarray(model.coefficients())).any(axis=1).tolist() == [
        i == 4 for i in range(Y.shape[0])
    ]
    assert np.isnan(np.asarray(model.basis_vectors())).any(axis=1).tolist() == [
        i == 5 for i in range(Y.shape[1])
    ]


def test_masks_survive_npz_and_infer_matches(excluded_fit, tmp_path):
    model, state, Y, W = excluded_fit
    path = tmp_path / "s.npz"
    save_state_to_npz(state, path)
    loaded = load_state_from_npz(path)
    np.testing.assert_array_equal(np.asarray(loaded.col_excluded), np.asarray(state.col_excluded))
    with pytest.warns(UnderdeterminedWarning):
        a, _ = model.infer(Y, W, state=state, max_iter=5)
    with pytest.warns(UnderdeterminedWarning):
        b, _ = model.infer(Y, W, state=loaded, max_iter=5)
    np.testing.assert_array_equal(np.asarray(a.A), np.asarray(b.A))


def test_fit_continues_from_returned_state(excluded_fit):
    model, state, Y, W = excluded_fit
    with warnings.catch_warnings():
        # Nothing new is under-covered, so nothing new is reported.
        warnings.simplefilter("error", UnderdeterminedWarning)
        new_state, loss = model.fit(Y, W, max_iter=5, init_state=state)
    np.testing.assert_array_equal(
        np.asarray(new_state.row_excluded), np.asarray(state.row_excluded)
    )
    np.testing.assert_array_equal(
        np.asarray(new_state.col_excluded), np.asarray(state.col_excluded)
    )
    assert np.isnan(np.asarray(new_state.G)[5]).all()
    assert np.isfinite(np.asarray(loss)).all()
    assert model.fit_info.exclusions == ()


def test_infer_keeps_training_exclusions_and_excludes_new_rows(excluded_fit):
    model, state, Y, W = excluded_fit
    W_new = W[:10].copy()
    W_new[2, :] = 0.0
    W_new[2, :2] = 1.0  # coverage 2 < K
    before = model.fit_info
    with pytest.warns(
        UnderdeterminedWarning, match=r"infer \(K=3\): 2 rows .*rows 2, 4"
    ) as record:
        inf, loss = model.infer(Y[:10], W_new, max_iter=5)
    assert len(_underdetermined(record)) == 1
    assert np.flatnonzero(np.asarray(inf.row_excluded)).tolist() == [2, 4]
    np.testing.assert_array_equal(np.asarray(inf.col_excluded), np.asarray(state.col_excluded))
    A = np.asarray(inf.A)
    assert np.isnan(A[[2, 4]]).all() and np.isfinite(np.delete(A, [2, 4], 0)).all()
    assert np.isnan(np.asarray(inf.G)[5]).all()
    assert np.isfinite(np.asarray(loss)).all()
    assert model.fit_info is before


@pytest.mark.parametrize("where", ["fit", "infer"])
def test_state_with_nan_outside_masks_raises(excluded_fit, where):
    model, state, Y, W = excluded_fit
    bad = update_state(state, A=state.A.at[0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="outside its exclusion masks"):
        if where == "fit":
            model.fit(Y, W, max_iter=2, init_state=bad)
        else:
            model.infer(Y, W, state=bad, max_iter=2)
    unmasked = with_exclusions(state, None, None)
    with pytest.raises(ValueError, match="outside its exclusion masks"):
        if where == "fit":
            model.fit(Y, W, max_iter=2, init_state=unmasked)
        else:
            model.infer(Y, W, state=unmasked, max_iter=2)


# ----------------------------
# A ridge too small to determine anything
# ----------------------------
def issue10_like(seed=0, N=200, M=400, K_true=4, n_sparse=40, keep_rows=3, sigma=0.01):
    """Scaled-down issue #10: a block of columns kept by only ``keep_rows`` rows."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, K_true))
    G = rng.normal(size=(M, K_true))
    Y = A @ G.T + sigma * rng.normal(size=(N, M))
    W = np.full((N, M), 1 / sigma**2)
    sparse = np.arange(M - n_sparse, M)
    W[:, sparse] = 0.0
    W[np.ix_(rng.choice(N, keep_rows, replace=False), sparse)] = 1 / sigma**2
    Y[W == 0] = 0.0
    return {"Y": Y, "W": W, "sparse_cols": sparse, "Y_clean": A @ G.T}


def _negligible(record):
    return [m for m in _underdetermined(record) if "below" in m and "resolution" in m]


def test_negligible_ridge_threshold():
    from robusta_hmf.coverage import negligible_ridge

    for dtype in (np.float32, np.float64):
        eps = float(np.finfo(dtype).eps)
        d = np.array([1e4])
        threshold = 10 * eps * d[0]
        assert negligible_ridge(threshold * 0.99, eps, d)[0]
        assert not negligible_ridge(threshold * 1.01, eps, d)[0]


def test_negligible_ridge_warns_in_float32():
    p = issue10_like()
    model = Robusta(rank=8, robust=False, als_ridge=1e-6, dtype="float32", conv_strategy="none")
    with pytest.warns(UnderdeterminedWarning) as record:
        # The warning comes from the check before the loop; one iteration is enough, and
        # more can reach NaN (which iteration depends on the CPU's float32 rounding).
        model.fit(p["Y"], p["W"], max_iter=1)
    [msg] = _negligible(record)
    assert "als_ridge=1e-06 is below float32 resolution for 40 columns" in msg
    # The threshold it reports is 10·ε·d for the largest such diagonal (~3 weights of 1e4).
    d_max = float(msg.split("largest diagonal ")[1].split(",")[0])
    assert 1e3 < d_max < 1e6
    threshold = float(msg.split("must exceed ~")[1].split(" ")[0])
    np.testing.assert_allclose(threshold, 10 * np.finfo(np.float32).eps * d_max, rtol=0.1)


@pytest.mark.parametrize(
    "dtype,ridge", [("float64", 1e-6), ("float32", 1e3)], ids=["float64", "float32-big-ridge"]
)
def test_resolvable_ridge_does_not_warn(dtype, ridge):
    p = issue10_like()
    model = Robusta(rank=8, robust=False, als_ridge=ridge, dtype=dtype, conv_strategy="none")
    with pytest.warns(UnderdeterminedWarning) as record:  # the coverage report itself
        model.fit(p["Y"], p["W"], max_iter=3)
    assert _negligible(record) == []
    assert any("determines their factors" in m for m in _underdetermined(record))


def test_negligible_ridge_with_full_coverage_is_not_checked(problem):
    # The ridge is negligible everywhere, but no system needs it.
    model = Robusta(
        rank=RANK, robust=False, als_ridge=1e-30, dtype="float32", conv_strategy="none"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnderdeterminedWarning)
        model.fit(problem["Y"], problem["W"], max_iter=3)


def test_sharded_max_diagonals_match_dense(problem):
    dense = Robusta(rank=RANK, robust=True, robust_scale=2.0)
    dense.fit(problem["Y"], problem["W"], max_iter=5)
    sharded = Robusta(rank=RANK, robust=True, robust_scale=2.0, override_frame=_frame(True))
    dA, dG = dense._frame.max_diagonals(
        *dense._frame.prepare(problem["Y"], problem["W"]), dense.state
    )
    sA, sG = sharded._frame.max_diagonals(
        *sharded._frame.prepare(problem["Y"], problem["W"]), dense.state
    )
    # The sharded store is float32, so agreement is to float32 round-off.
    np.testing.assert_allclose(sA, dA, rtol=1e-5)
    np.testing.assert_allclose(sG, dG, rtol=1e-5)


# ----------------------------
# Convergence reporting: rising loss, max_iter, non-finite factors
# ----------------------------
def _convergence(record):
    return [str(w.message) for w in record if issubclass(w.category, ConvergenceWarning)]


EPS32 = float(np.finfo(np.float32).eps)


def test_rising_loss_rule_on_fixed_sequences():
    from robusta_hmf.convergence import rising_loss_iterations

    # Strictly decreasing: nothing.
    assert rising_loss_iterations([10, 8, 6, 5, 4.5, 4.4, 4.39], EPS32, data_scale=10) == []
    # Round-off-sized rises near convergence, below 100·ε·√(L·S0) ≈ 15: nothing.
    L = 1.25e6 + np.array([5, 3, 2, 1.5, 1.4, 1.5, 1.4, 2.0, 1.3])
    assert rising_loss_iterations(L, EPS32, data_scale=1.25e6) == []
    # A real jump at t=3: flagged at 3 although the 5-step window isn't full yet.
    assert rising_loss_iterations([10, 9, 8.5, 50, 40, 39], EPS32, data_scale=10) == [3]
    # A rise smaller than twice the recent average decrease is tolerated...
    assert rising_loss_iterations([100, 90, 80, 70, 60, 50, 55], EPS32, data_scale=100) == []
    # ...but not one larger.
    assert rising_loss_iterations([100, 90, 80, 70, 60, 50, 71], EPS32, data_scale=100) == [6]
    # Only iterations from `start` on are evaluated.
    seq = [10, 9, 8.5, 50, 40, 60]
    assert rising_loss_iterations(seq, EPS32, start=4, data_scale=10) == [5]


def test_rising_loss_floor_handles_a_loss_at_round_off():
    from robusta_hmf.convergence import rising_loss_iterations

    # The exact fit in test_fit_shape_extremes[30-6-6]: S0 ≈ 1e5, the loss is round-off.
    L = [1.6e-26, 1.5e-25, 9e-26, 2e-25]
    assert rising_loss_iterations(L, 2.2e-16) != []  # no floor: false alarm
    assert rising_loss_iterations(L, 2.2e-16, data_scale=1e5) == []
    # A high-S/N fit near convergence: L ≈ 67 with S0 ≈ 2e7; a 13th-digit wobble.
    L = 66.6 + np.array([0.0, -1e-12, 2e-12, -1e-12, 3e-12])
    assert rising_loss_iterations(L, 2.2e-16, data_scale=2e7) == []


def test_exact_fit_does_not_warn():
    prob = make_low_rank_problem(seed=3, N=30, M=6, K=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        Robusta(rank=6, robust=False, conv_strategy="none").fit(prob["Y"], prob["W"], max_iter=20)


def tiny_weight_regularised(tiny=1e-6, **kw):
    p = issue10_like(**kw)
    W = p["W"].copy()
    W[:, p["sparse_cols"]] = np.where(W[:, p["sparse_cols"]] == 0, tiny, W[:, p["sparse_cols"]])
    return p["Y"], W


def _fit_reports(model, Y, W, max_iter=60):
    """(raised FloatingPointError?, rising-loss warning messages)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            model.fit(Y, W, max_iter=max_iter)
            raised = False
        except FloatingPointError:
            raised = True
    return raised, [m for m in _convergence(record) if "loss rose" in m]


@pytest.mark.parametrize("sharded", [False, True])
def test_unstable_float32_fit_is_reported(sharded):
    """Scaled-down #10 regularised by hand: the sparse columns' missing entries get
    W = 1e-6 instead of 0, so they count as covered and nothing is excluded. Their
    systems mix weights 1e4 and 1e-6, a ratio of 1e10: beyond float32's resolution
    (~1e7), well inside float64's (~1e16). In float32 the fit is unstable and must be
    reported, as a
    rising loss (ConvergenceWarning) or a non-finite one (FloatingPointError): which of
    the two depends on the CPU's rounding (arm64 and x86-64 differ)."""
    Y, W = tiny_weight_regularised()
    model = Robusta(
        rank=8, robust=True, dtype="float32", conv_strategy="none", override_frame=_frame(sharded)
    )
    raised, rises = _fit_reports(model, Y, W)
    assert raised or rises
    assert len(rises) <= 1  # at most one rising-loss warning per fit
    if rises:
        assert rises[0].startswith("K=8, Q=1: loss rose at iteration")


@pytest.mark.parametrize("sharded", [False, True])
def test_same_fit_in_float64_is_not_flagged(sharded):
    Y, W = tiny_weight_regularised()
    model = Robusta(
        rank=8, robust=True, dtype="float64", conv_strategy="none", override_frame=_frame(sharded)
    )
    assert _fit_reports(model, Y, W) == (False, [])


@pytest.mark.parametrize("sharded", [False, True])
def test_one_huge_outlier_does_not_hide_an_instability(sharded):
    """The floor's scale is the robust loss of the zero model, which caps outliers; a
    plain Σ W·Y² would let this one entry (W·Y² = 1e18) silence the rule."""
    Y, W = tiny_weight_regularised()
    Y[0, 0] = 1e7
    model = Robusta(
        rank=8, robust=True, dtype="float32", conv_strategy="none", override_frame=_frame(sharded)
    )
    raised, rises = _fit_reports(model, Y, W)
    assert raised or rises


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sharded", [False, True])
def test_healthy_fits_never_flag_a_rise(problem, dtype, sharded):
    model = Robusta(
        rank=RANK, robust=True, dtype=dtype, conv_tol=1e-8, override_frame=_frame(sharded)
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(problem["Y"], problem["W"], max_iter=300)
    assert [m for m in _convergence(record) if "loss rose" in m] == []
    assert model.fit_info.loss_rises == ()


@pytest.mark.parametrize("sharded", [False, True])
def test_max_iter_warns_with_details(problem, sharded):
    model = Robusta(rank=RANK, robust=True, robust_scale=3.0, override_frame=_frame(sharded))
    with pytest.warns(ConvergenceWarning) as record:
        _, loss = model.fit(problem["Y"], problem["W"], max_iter=4)
    [msg] = _convergence(record)
    assert msg == (
        f"K=3, Q=3: stopped at max_iter=4 without converging (final loss "
        f"{float(loss[-1]):.5g}); see model.fit_info"
    )
    info = model.fit_info
    assert info.converged is False and info.stop_reason == "max_iter" and info.n_iter == 4


def test_gaussian_max_iter_message_has_no_Q(problem):
    with pytest.warns(ConvergenceWarning, match=r"^K=3: stopped at max_iter=4"):
        Robusta(rank=RANK, robust=False).fit(problem["Y"], problem["W"], max_iter=4)


def test_converged_fit_does_not_warn(problem):
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(problem["Y"], problem["W"], max_iter=500)
    assert model.fit_info.converged is True


def test_conv_strategy_none_does_not_warn(problem):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        Robusta(rank=RANK, robust=False, conv_strategy="none").fit(
            problem["Y"], problem["W"], max_iter=4
        )


def test_different_K_in_one_loop_warn_separately(problem):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")  # Python's default once-per-location filtering
        for K in (2, 3):
            Robusta(rank=K, robust=False).fit(problem["Y"], problem["W"], max_iter=3)
    msgs = _convergence(record)
    assert len(msgs) == 2 and msgs[0].startswith("K=2") and msgs[1].startswith("K=3")


def test_convergence_warning_can_be_made_an_error(problem):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning, match="max_iter=3"):
            Robusta(rank=RANK, robust=False).fit(problem["Y"], problem["W"], max_iter=3)


def test_sgd_is_exempt_from_the_rising_loss_rule(problem):
    model = Robusta(rank=RANK, method="sgd", robust=False, learning_rate=0.5, conv_strategy="none")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, loss = model.fit(problem["Y"], problem["W"], max_iter=40)
    loss = np.asarray(loss)
    assert (np.diff(loss) > 1e-3 * np.abs(loss[:-1])).any()  # SGD did go uphill
    assert [m for m in _convergence(record) if "loss rose" in m] == []
    assert model.fit_info.loss_rises == ()


@pytest.mark.parametrize("sharded", [False, True])
def test_infer_max_iter_warning_names_rows_and_stores_nothing(problem, sharded):
    model = Robusta(rank=RANK, robust=False, conv_tol=1e-4, override_frame=_frame(sharded))
    model.fit(problem["Y"], problem["W"], max_iter=500)
    before = model.fit_info
    with pytest.warns(ConvergenceWarning) as record:
        model.infer(problem["Y"][:12], problem["W"][:12], max_iter=2, conv_tol=1e-30)
    [msg] = _convergence(record)
    assert msg.startswith("infer on 12 rows (K=3): stopped at max_iter=2 without converging")
    assert "fit_info" not in msg
    assert model.fit_info is before


def test_nonfinite_factors_raise():
    from robusta_hmf.frame import check_factors_finite

    check_factors_finite(np.ones((3, 2)), np.ones((4, 2)), 7, np.float64)
    with pytest.raises(FloatingPointError, match="A or G is not finite at iteration 7"):
        check_factors_finite(np.ones((3, 2)), np.array([[1.0, np.nan]] * 4), 7, np.float64)


def test_sharded_nonfinite_loss_is_caught_at_the_end(problem):
    # max_iter is below conv_check_cadence, so only the end-of-run check can see it.
    Y = problem["Y"].copy()
    Y[5, 7] = 1e200
    frame = ShardedFrame(
        row_block=16, devices=jax.devices("cpu")[:1], verbose=False, store_dtype=np.float64
    )
    model = Robusta(rank=RANK, init_strategy="random", override_frame=frame)
    with pytest.raises(FloatingPointError, match="Loss is not finite"):
        model.fit(Y, problem["W"], max_iter=3, conv_check_cadence=10)


@pytest.mark.parametrize("sharded", [False, True])
def test_instability_is_reported_at_the_first_check(capsys, sharded):
    """With ConvergenceWarning made an error, the unstable fit stops at the first
    convergence check (iteration 10) by either route, not at the end of the run."""
    Y, W = tiny_weight_regularised()
    model = Robusta(
        rank=8, robust=True, dtype="float32", conv_strategy="none", override_frame=_frame(sharded)
    )
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises((ConvergenceWarning, FloatingPointError), match="iteration"):
            model.fit(Y, W, max_iter=30, conv_check_cadence=10)
    # Checks run at iterations 10 and 20. Stopping at the first means the second never
    # prints its progress line. (The NaN route prints the first line, then raises.)
    assert "iter 021" not in capsys.readouterr().out


# ----------------------------
# score() with exclusions
# ----------------------------
def test_score_with_exclusions_equals_zeroing_by_hand(excluded_fit, problem):
    model, state, _, _ = excluded_fit
    Y, W = problem["Y"], problem["W"]  # full weights: row 4 and column 5 do carry data here
    m = model.score(Y, W, state=state)
    assert np.isfinite([m.kl, m.mean_z, m.std_z, m.chi2_red]).all()
    assert (m.n_excluded_rows, m.n_excluded_cols) == (1, 1)
    assert m.n_excluded_entries == 30 + 60 - 1
    assert m.n == Y.size  # excluded entries still count in n, like any zero-weight entry

    W_hand = W.copy()
    W_hand[4, :] = 0.0
    W_hand[:, 5] = 0.0
    A = np.nan_to_num(np.asarray(state.A))
    G = np.nan_to_num(np.asarray(state.G))
    ref = model.score(Y, W_hand, state=RHMFState(A=A, G=G))
    for key in ("sum_z", "sum_z2", "sum_chi2", "n"):
        np.testing.assert_allclose(getattr(m, key), getattr(ref, key), rtol=1e-12)
    assert ref.n_excluded_entries == 0
    assert m.as_dict()["n_excluded_entries"] == 89


def test_score_dense_and_sharded_agree_with_exclusions(excluded_fit, problem):
    model, state, _, _ = excluded_fit
    dense = model.score(problem["Y"], problem["W"], state=state)
    sharded = _model(True).score(problem["Y"], problem["W"], state=state)
    for key in ("sum_z", "sum_z2", "sum_chi2"):
        # The sharded store is float32.
        np.testing.assert_allclose(getattr(sharded, key), getattr(dense, key), rtol=1e-5)
    for key in ("n", "n_excluded_rows", "n_excluded_cols", "n_excluded_entries"):
        assert getattr(sharded, key) == getattr(dense, key)


def test_score_rejects_nan_outside_masks(excluded_fit, problem):
    model, state, _, _ = excluded_fit
    bad = update_state(state, G=state.G.at[0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="outside its exclusion masks"):
        model.score(problem["Y"], problem["W"], state=bad)


def test_score_without_exclusions_reports_zero_counts(problem):
    model = _model()
    model.fit(problem["Y"], problem["W"], max_iter=15)
    m = model.score(problem["Y"], problem["W"])
    assert (m.n_excluded_rows, m.n_excluded_cols, m.n_excluded_entries) == (0, 0, 0)


# ----------------------------
# Sharded parity
# ----------------------------
@pytest.mark.parametrize("n_devices", [1, 4])
def test_sharded_coverage_matches_dense(n_devices):
    rng = np.random.default_rng(2)
    N, M = 61, 23  # not a multiple of the chunk or device count: padding rows exist
    W = (rng.random((N, M)) < 0.3).astype(float)
    row_active = rng.random(N) < 0.8
    col_active = rng.random(M) < 0.8
    frame = ShardedFrame(row_block=8, devices=jax.devices("cpu")[:n_devices], verbose=False)
    data, _ = frame.prepare(np.zeros((N, M)), W)
    s_rows, s_cols = frame.coverage(data, None, row_active, col_active)
    assert s_rows.shape == (N,)  # padding rows are not reported
    np.testing.assert_array_equal(s_rows, (W > 0)[:, col_active].sum(axis=1))
    np.testing.assert_array_equal(s_cols, (W > 0)[row_active].sum(axis=0))


@pytest.mark.parametrize("n_devices", [1, 4])
def test_sharded_infer_keeps_training_exclusions(excluded_fit, n_devices):
    _, state, Y, W = excluded_fit
    W_new = W[:13].copy()
    W_new[2, :] = 0.0
    W_new[2, :2] = 1.0
    frame = ShardedFrame(row_block=4, devices=jax.devices("cpu")[:n_devices], verbose=False)
    model = Robusta(rank=RANK, robust=False, conv_strategy="none", override_frame=frame)
    with pytest.warns(UnderdeterminedWarning, match=r"rows 2, 4"):
        inf, _ = model.infer(Y[:13], W_new, state=state, max_iter=5)
    assert np.flatnonzero(np.asarray(inf.row_excluded)).tolist() == [2, 4]
    A = np.asarray(inf.A)
    assert A.shape == (13, RANK)
    assert np.isnan(A[[2, 4]]).all() and np.isfinite(np.delete(A, [2, 4], 0)).all()


# ----------------------------
# Regression: issue #10
# ----------------------------
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_issue10_regression(dtype):
    """Scaled-down #10: 40 columns kept by only 3 rows at K=8, no ridge.

    Before exclusion existed, float32 went NaN at iteration ~40 and float64 never
    converged, with the well-covered columns ending ~66σ from the truth.
    """
    p = issue10_like()
    good = np.setdiff1d(np.arange(p["Y"].shape[1]), p["sparse_cols"])
    model = Robusta(rank=8, robust=True, dtype=dtype)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        state, _ = model.fit(p["Y"], p["W"], max_iter=1000)
    assert _convergence(record) == []
    assert model.fit_info.converged
    assert np.flatnonzero(np.asarray(state.col_excluded)).tolist() == p["sparse_cols"].tolist()
    assert state.row_excluded is None
    pred = np.asarray(state.A) @ np.asarray(state.G)[good].T
    rms = np.sqrt(np.mean((pred - p["Y_clean"][:, good]) ** 2))
    assert rms < 0.01  # sigma = 0.01: the model is within the noise of the truth


def test_run_monitor_warns_once_and_records_every_rise():
    from robusta_hmf.convergence import RunMonitor

    monitor = RunMonitor(rank=4, scale=2.0, check_rises=True, infer=False, n_rows=10)
    losses = [100.0, 90.0, 85.0, 425.0, 80.0, 79.0, 300.0]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        monitor.update(losses[:4])  # a convergence check after the first rise
        monitor.update(losses)  # a later check, after the second
        rises = monitor.finish(losses, converged=False, max_iter=100, strategy="max_frac_G")
    expected = (
        "K=4, Q=2: loss rose at iteration 3 (85 → 425), which the ALS iteration "
        "should never do; the fit is numerically unreliable. See "
        "model.fit_info.loss_rises."
    )
    assert _convergence(record) == [expected]
    assert rises == (3, 6)


def test_run_monitor_exempts_sgd_and_reports_infer_rows():
    from robusta_hmf.convergence import RunMonitor

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sgd = RunMonitor(rank=3, scale=None, check_rises=False, infer=False, n_rows=5)
        assert sgd.finish([1.0, 5.0, 2.0], converged=True, max_iter=3, strategy="none") == ()
        inf = RunMonitor(rank=3, scale=None, check_rises=True, infer=True, n_rows=1234)
        inf.finish([3.0, 2.0], converged=False, max_iter=2, strategy="max_frac_A")
    assert _convergence(record) == [
        "infer on 1,234 rows (K=3): stopped at max_iter=2 without converging (final loss 2)"
    ]


@pytest.mark.parametrize("sharded", [False, True])
def test_ridge_fits_are_exempt_from_the_rising_loss_rule(sharded):
    """With a ridge, ALS minimises loss + ridge·(‖A‖² + ‖G‖²), so the data loss can rise
    on a correct fit (and with rotation the penalised objective is not monotone either)."""
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(60, 3)) @ rng.normal(size=(80, 3)).T + 0.1 * rng.normal(size=(60, 80))
    W = np.full(Y.shape, 100.0)
    model = Robusta(rank=3, als_ridge=1.0, conv_strategy="none", override_frame=_frame(sharded))
    _, loss = model.fit(Y, W, max_iter=60)
    loss = np.asarray(loss)
    assert (np.diff(loss) > 0).any()  # the data loss does rise...
    assert model.fit_info.loss_rises == ()  # ...and that is not reported


@pytest.mark.parametrize("sharded", [False, True])
def test_max_iter_zero_returns_an_empty_history(problem, sharded):
    model = Robusta(rank=RANK, override_frame=_frame(sharded))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        _, loss = model.fit(problem["Y"], problem["W"], max_iter=0)
        _, loss_inf = model.infer(problem["Y"][:5], problem["W"][:5], max_iter=0)
    assert np.asarray(loss).size == 0 and np.asarray(loss_inf).size == 0
    assert model.fit_info.n_iter == 0 and model.fit_info.converged is False
