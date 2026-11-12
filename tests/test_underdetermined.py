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
    event = ExclusionEvent(iteration=-1, axis="col", indices=np.array([2, 5]), cause="empty")
    assert event.iteration == -1 and event.axis == "col" and event.cause == "empty"


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
    with pytest.warns(DataWarning):
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
