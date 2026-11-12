# test_precision.py

import jax
import numpy as np
import pytest
from conftest import make_low_rank_problem, subspace_alignment

from robusta_hmf import PrecisionError, Robusta, ShardedFrame
from robusta_hmf.precision import check_precision, resolve_dtype

FIT_KW = dict(max_iter=60, conv_check_cadence=5)


@pytest.fixture
def x64_disabled():
    """Turn JAX's float64 support off for one test, as in a fresh user session."""
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def problem():
    return make_low_rank_problem(seed=3, N=120, M=40, K=3)


def one_device_frame(**kwargs):
    return ShardedFrame(row_block=64, devices=jax.devices("cpu")[:1], verbose=False, **kwargs)


# ---------------------------------------------------------------------------- #
# The default is float64, and it refuses to run silently in float32
# ---------------------------------------------------------------------------- #


def test_default_dtype_is_float64():
    assert Robusta(rank=3).dtype == np.float64


def test_default_fit_raises_when_x64_is_off(problem, x64_disabled):
    model = Robusta(rank=3)
    with pytest.raises(PrecisionError) as excinfo:
        model.fit(problem["Y"], problem["W"], **FIT_KW)
    message = str(excinfo.value)
    # The message has to carry both ways out, so the user needs no docs.
    assert 'jax.config.update("jax_enable_x64", True)' in message
    assert 'Robusta(rank=3, dtype="float32")' in message


def test_infer_and_score_also_raise_when_x64_is_off(problem):
    model = Robusta(rank=3)
    model.fit(problem["Y"], problem["W"], **FIT_KW)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(PrecisionError):
            model.infer(problem["Y"], problem["W"])
        with pytest.raises(PrecisionError):
            model.score(problem["Y"], problem["W"])
    finally:
        jax.config.update("jax_enable_x64", True)


def test_sharded_default_fit_raises_when_x64_is_off(problem, x64_disabled):
    model = Robusta(rank=3, override_frame=one_device_frame())
    with pytest.raises(PrecisionError):
        model.fit(problem["Y"], problem["W"], **FIT_KW)


def test_check_precision_passes_with_x64_on():
    check_precision(np.dtype("float64"))
    check_precision(np.dtype("float32"))


# ---------------------------------------------------------------------------- #
# float32 is an explicit opt-in, and the dtype asked for is the dtype computed in
# ---------------------------------------------------------------------------- #


def test_float32_opt_in_runs_with_x64_off_and_matches_float64(problem):
    ref = Robusta(rank=3)
    ref.fit(problem["Y"], problem["W"], **FIT_KW)

    jax.config.update("jax_enable_x64", False)
    try:
        model = Robusta(rank=3, dtype="float32")
        state, loss = model.fit(problem["Y"], problem["W"], **FIT_KW)
    finally:
        jax.config.update("jax_enable_x64", True)

    assert state.A.dtype == np.float32 and state.G.dtype == np.float32
    assert np.all(np.isfinite(np.asarray(loss)))
    assert subspace_alignment(state.G, ref.G) > 1 - 1e-5


def test_float64_inputs_are_cast_down_to_an_explicit_float32(problem):
    # x64 is on here, so nothing but the explicit dtype keeps this in float32.
    model = Robusta(rank=3, dtype="float32")
    state, _ = model.fit(problem["Y"], problem["W"], **FIT_KW)
    assert state.A.dtype == np.float32 and state.G.dtype == np.float32


def test_float32_inputs_are_cast_up_to_the_float64_default(problem):
    Y32 = problem["Y"].astype(np.float32)
    W32 = problem["W"].astype(np.float32)
    state, _ = Robusta(rank=3).fit(Y32, W32, **FIT_KW)
    assert state.A.dtype == np.float64 and state.G.dtype == np.float64


def test_infer_uses_the_model_dtype_whatever_the_state_dtype(problem):
    ref = Robusta(rank=3)
    state64, _ = ref.fit(problem["Y"], problem["W"], **FIT_KW)
    model32 = Robusta(rank=3, dtype="float32")
    inferred, _ = model32.infer(problem["Y"], problem["W"], state=state64)
    assert inferred.A.dtype == np.float32 and inferred.G.dtype == np.float32


@pytest.mark.parametrize("dtype", ["float32", np.float32, np.dtype("float32"), jax.numpy.float32])
def test_dtype_spellings(dtype):
    assert Robusta(rank=2, dtype=dtype).dtype == np.float32


@pytest.mark.parametrize("dtype", ["float16", "int32", "complex64", "not-a-dtype"])
def test_unsupported_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="dtype must be one of"):
        resolve_dtype(dtype)
    with pytest.raises(ValueError, match="dtype must be one of"):
        Robusta(rank=2, dtype=dtype)


# ---------------------------------------------------------------------------- #
# The sharded frame takes its dtype from the same place
# ---------------------------------------------------------------------------- #


def test_sharded_frame_inherits_the_model_dtype(problem):
    model = Robusta(rank=3, dtype="float32", override_frame=one_device_frame())
    assert model._frame.compute_dtype == np.float32
    state, _ = model.fit(problem["Y"], problem["W"], **FIT_KW)
    assert state.A.dtype == np.float32 and state.G.dtype == np.float32


def test_sharded_frame_compute_dtype_counts_as_the_explicit_choice():
    model = Robusta(rank=3, override_frame=one_device_frame(compute_dtype=np.float32))
    assert model.dtype == np.float32
    assert model._frame.compute_dtype == np.float32


def test_conflicting_model_and_frame_dtypes_are_rejected():
    with pytest.raises(ValueError, match="conflicts with the frame's compute_dtype"):
        Robusta(rank=3, dtype="float64", override_frame=one_device_frame(compute_dtype=np.float32))


def test_matching_model_and_frame_dtypes_are_accepted():
    frame = one_device_frame(compute_dtype=np.float32)
    assert Robusta(rank=3, dtype="float32", override_frame=frame).dtype == np.float32


def test_sharded_float32_opt_in_runs_with_x64_off(problem, x64_disabled):
    model = Robusta(rank=3, dtype="float32", override_frame=one_device_frame())
    state, loss = model.fit(problem["Y"], problem["W"], **FIT_KW)
    assert state.G.dtype == np.float32
    assert np.all(np.isfinite(np.asarray(loss)))


# ---------------------------------------------------------------------------- #
# A loss that goes non-finite stops the fit with a diagnosis
# ---------------------------------------------------------------------------- #


def nan_problem(problem, dtype="float64"):
    # A finite value whose squared residual overflows the dtype, so the loss
    # goes non-finite inside the loop. (NaN inputs no longer reach the loop.)
    Y = problem["Y"].copy()
    Y[5, 7] = 1e30 if dtype == "float32" else 1e200
    return Y, problem["W"]


def test_dense_fit_stops_at_the_first_check_after_the_loss_goes_nonfinite(problem):
    # The loss is NaN from iteration 0; with conv_check_cadence=5 the fit must
    # stop at iteration 5, not run on to max_iter.
    Y, W = nan_problem(problem)
    with pytest.raises(FloatingPointError, match="Loss is not finite at iteration 5:"):
        Robusta(rank=3, init_strategy="random").fit(Y, W, **FIT_KW)


def test_dense_nonfinite_loss_is_caught_before_the_first_convergence_check(problem):
    # max_iter is below conv_check_cadence, so only the end-of-run check can see it.
    Y, W = nan_problem(problem)
    with pytest.raises(FloatingPointError, match="Loss is not finite"):
        Robusta(rank=3, init_strategy="random").fit(Y, W, max_iter=3, conv_check_cadence=10)


def test_float32_nonfinite_loss_suggests_float64(problem):
    Y, W = nan_problem(problem, "float32")
    with pytest.raises(FloatingPointError, match="try float64 first"):
        Robusta(rank=3, dtype="float32", init_strategy="random").fit(Y, W, **FIT_KW)


def test_sharded_fit_raises_on_a_nonfinite_loss_with_the_same_diagnosis(problem):
    Y, W = nan_problem(problem)
    # Stored in float64: a float32 store would turn the value into inf, which
    # is then treated as missing rather than reaching the loop.
    frame = one_device_frame(store_dtype=np.float64)
    model = Robusta(rank=3, init_strategy="random", override_frame=frame)
    with pytest.raises(FloatingPointError, match="large enough to overflow"):
        model.fit(Y, W, **FIT_KW)


def test_dense_fit_requires_weights(problem):
    with pytest.raises(ValueError, match="W .* is required"):
        Robusta(rank=3).fit(problem["Y"], None, **FIT_KW)
