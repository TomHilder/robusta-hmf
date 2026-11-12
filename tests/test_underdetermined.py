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
