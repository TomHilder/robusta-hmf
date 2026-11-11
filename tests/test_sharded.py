# test_sharded.py
#
# The sharded frame is a scheduling variant, not a second model: it must take
# the same mathematical steps as the dense path and land in the same place, to
# round-off. Almost everything here is an equivalence test against Robusta's
# default OptFrame.
#
# The 4-device mesh runs on forced host CPU devices; conftest.py sets
# XLA_FLAGS before jax initialises so that works without an accelerator.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_low_rank_problem, make_outlier_problem

from robusta_hmf import Robusta
from robusta_hmf.metrics import z_moments
from robusta_hmf.sharded import (
    GramInitialiser,
    RowPlan,
    ShardedData,
    ShardedFrame,
    plan_rows,
    shard_data,
    shard_rows,
)

jax.config.update("jax_enable_x64", True)

RANK = 3  # true rank of the synthetic problems in conftest
DEVICE_COUNTS = [1, 4]


def cpu_devices(n):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(f"needs {n} CPU devices, have {len(devices)}")
    return devices[:n]


def align_signs(G_ref, G):
    """Each basis vector's sign is arbitrary between the two SVD routes."""
    G_ref, G = np.asarray(G_ref), np.asarray(G)
    return G * np.sign(np.sum(G_ref * G, axis=0))


def sharded_model(n_devices=1, row_block=64, verbose=False, **kwargs):
    return Robusta(
        override_frame=ShardedFrame(
            row_block=row_block,
            devices=cpu_devices(n_devices),
            store_dtype=np.float64,
            verbose=verbose,
        ),
        **kwargs,
    )


# ----------------------------
# Row plan
# ----------------------------
@pytest.mark.parametrize("n_rows", [1, 7, 199, 499, 501, 512, 4097])
@pytest.mark.parametrize("n_devices", [1, 3, 4])
@pytest.mark.parametrize("row_block", [1, 64, 4096])
def test_plan_rows_covers_every_row(n_rows, n_devices, row_block):
    plan = plan_rows(n_rows, n_devices, row_block)
    assert plan.n_padded >= n_rows
    assert plan.n_padded == plan.chunk * plan.n_chunks * n_devices
    assert plan.n_pad == plan.n_padded - n_rows
    assert plan.chunk <= row_block or plan.n_chunks == 1
    # Padding never exceeds one chunk per device, which is what keeps it small.
    assert plan.n_pad < plan.chunk * n_devices + n_devices


@pytest.mark.parametrize("bad", [(0, 1, 8), (10, 1, 0)])
def test_plan_rows_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        plan_rows(*bad)


def test_row_plan_properties():
    plan = RowPlan(n_rows=10, n_devices=2, chunk=3, n_chunks=2)
    assert plan.rows_per_device == 6
    assert plan.n_padded == 12
    assert plan.n_pad == 2


# ----------------------------
# Data placement
# ----------------------------
@pytest.mark.parametrize("n_devices", DEVICE_COUNTS)
@pytest.mark.parametrize("n_rows", [199, 200, 501])
def test_shard_rows_preserves_values_and_zero_pads(n_devices, n_rows):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_rows, 7))
    mesh = ShardedFrame(devices=cpu_devices(n_devices)).mesh
    plan = plan_rows(n_rows, n_devices, 64)
    out = np.asarray(shard_rows(x, plan, mesh, np.float64))
    assert out.shape == (plan.n_padded, 7)
    assert np.allclose(out[:n_rows], x)
    assert np.all(out[n_rows:] == 0.0)


def test_shard_data_rejects_mismatched_shapes():
    mesh = ShardedFrame(devices=cpu_devices(1)).mesh
    with pytest.raises(ValueError, match="same shape"):
        shard_data(np.zeros((4, 3)), np.zeros((4, 2)), mesh, 64)
    with pytest.raises(ValueError, match="W must be provided"):
        shard_data(np.zeros((4, 3)), None, mesh, 64)


def test_prepared_data_exposes_unpadded_shape():
    prob = make_low_rank_problem(N=199, M=50, K=RANK)
    frame = ShardedFrame(row_block=64, devices=cpu_devices(4), verbose=False)
    data, W = frame.prepare(prob["Y"], prob["W"])
    assert isinstance(data, ShardedData)
    assert W is None
    assert data.shape == (199, 50)
    assert data.n_rows == 199
    assert data.Y.shape[0] == data.plan.n_padded > 199


# ----------------------------
# Shared reference fits
# ----------------------------
@pytest.fixture(scope="module")
def problem():
    return make_outlier_problem(N=199, M=50, K=RANK)


@pytest.fixture(scope="module")
def dense_fit(problem):
    """Robusta's default dense path, the thing everything is compared against."""
    model = Robusta(rank=RANK, robust=True, robust_scale=3.0, conv_tol=0.0)
    state, loss = model.fit(
        jnp.array(problem["Y"]), jnp.array(problem["W"]), max_iter=15, conv_check_cadence=99
    )
    return model, state, loss


# ----------------------------
# Fit equivalence
# ----------------------------
@pytest.mark.parametrize("n_devices", DEVICE_COUNTS)
@pytest.mark.parametrize("row_block", [7, 64, 4096])
def test_fit_matches_dense_path(problem, dense_fit, n_devices, row_block):
    """Same steps, different schedule: agreement must be at round-off."""
    _, ref_state, ref_loss = dense_fit
    model = sharded_model(
        n_devices, row_block, rank=RANK, robust=True, robust_scale=3.0, conv_tol=0.0
    )
    state, loss = model.fit(problem["Y"], problem["W"], max_iter=15, conv_check_cadence=99)

    assert state.A.shape == (199, RANK)  # padding stripped
    assert np.max(np.abs(align_signs(ref_state.G, state.G) - np.asarray(ref_state.G))) < 1e-12
    assert np.max(np.abs(np.asarray(ref_loss) - np.asarray(loss))) < 1e-9
    assert int(state.it) == int(ref_state.it)


@pytest.mark.parametrize("n_rows", [199, 200, 256, 501])
def test_fit_handles_padded_and_exact_row_counts(n_rows):
    """N that divides the mesh evenly and N that does not must agree."""
    prob = make_low_rank_problem(N=n_rows, M=50, K=RANK)
    ref = Robusta(rank=RANK, robust=False, conv_tol=0.0)
    ref_state, ref_loss = ref.fit(
        jnp.array(prob["Y"]), jnp.array(prob["W"]), max_iter=10, conv_check_cadence=99
    )
    model = sharded_model(4, 64, rank=RANK, robust=False, conv_tol=0.0)
    state, loss = model.fit(prob["Y"], prob["W"], max_iter=10, conv_check_cadence=99)
    assert state.A.shape == (n_rows, RANK)
    assert np.max(np.abs(align_signs(ref_state.G, state.G) - np.asarray(ref_state.G))) < 1e-12
    assert np.max(np.abs(np.asarray(ref_loss) - np.asarray(loss))) < 1e-9


def test_chunk_and_device_invariance(problem):
    """Changing row_block or device count must not change the answer."""
    results = []
    for n_devices, row_block in [(1, 7), (1, 4096), (4, 13), (4, 64)]:
        model = sharded_model(n_devices, row_block, rank=RANK, robust=True, conv_tol=0.0)
        state, loss = model.fit(problem["Y"], problem["W"], max_iter=10, conv_check_cadence=99)
        results.append((np.asarray(state.G), np.asarray(loss)))
    base_G, base_loss = results[0]
    for G, loss in results[1:]:
        assert np.max(np.abs(align_signs(base_G, G) - base_G)) < 1e-12
        assert np.max(np.abs(loss - base_loss)) < 1e-9


@pytest.mark.parametrize("ridge", [None, 1e-6, 0.1])
def test_ridge_matches_dense_path(problem, ridge):
    ref = Robusta(rank=RANK, robust=True, als_ridge=ridge, conv_tol=0.0)
    ref_state, _ = ref.fit(
        jnp.array(problem["Y"]), jnp.array(problem["W"]), max_iter=10, conv_check_cadence=99
    )
    model = sharded_model(4, 64, rank=RANK, robust=True, als_ridge=ridge, conv_tol=0.0)
    state, _ = model.fit(problem["Y"], problem["W"], max_iter=10, conv_check_cadence=99)
    assert np.max(np.abs(align_signs(ref_state.G, state.G) - np.asarray(ref_state.G))) < 1e-12


@pytest.mark.parametrize("robust", [True, False])
def test_both_likelihoods_match_dense_path(problem, robust):
    ref = Robusta(rank=RANK, robust=robust, robust_scale=3.0, conv_tol=0.0)
    ref_state, ref_loss = ref.fit(
        jnp.array(problem["Y"]), jnp.array(problem["W"]), max_iter=10, conv_check_cadence=99
    )
    model = sharded_model(4, 64, rank=RANK, robust=robust, robust_scale=3.0, conv_tol=0.0)
    state, loss = model.fit(problem["Y"], problem["W"], max_iter=10, conv_check_cadence=99)
    assert np.max(np.abs(align_signs(ref_state.G, state.G) - np.asarray(ref_state.G))) < 1e-12
    assert np.max(np.abs(np.asarray(ref_loss) - np.asarray(loss))) < 1e-9


def test_float32_runs_and_tracks_float64(problem):
    """Reduced precision is a supported knob, not an accident."""
    model = Robusta(
        rank=RANK,
        robust=True,
        conv_tol=0.0,
        override_frame=ShardedFrame(
            row_block=64,
            devices=cpu_devices(4),
            store_dtype=np.float32,
            compute_dtype=np.float32,
            verbose=False,
        ),
    )
    state, loss = model.fit(problem["Y"], problem["W"], max_iter=10, conv_check_cadence=99)
    assert state.G.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(state.G)))
    assert np.all(np.isfinite(np.asarray(loss)))

    ref = sharded_model(4, 64, rank=RANK, robust=True, conv_tol=0.0)
    ref_state, _ = ref.fit(problem["Y"], problem["W"], max_iter=10, conv_check_cadence=99)
    aligned = align_signs(ref_state.G, state.G)
    assert np.max(np.abs(aligned - np.asarray(ref_state.G))) < 1e-4


def test_store_dtype_float32_keeps_float64_arithmetic(problem):
    """Storing the data in fp32 must not demote the accumulations."""
    model = sharded_model(4, 64, rank=RANK, robust=True, conv_tol=0.0)
    model._frame.store_dtype = np.dtype(np.float32)
    state, _ = model.fit(problem["Y"], problem["W"], max_iter=10, conv_check_cadence=99)
    assert state.G.dtype == jnp.float64


# ----------------------------
# Degenerate rows and pixels
# ----------------------------
def test_dead_row_and_pixel_do_not_poison_the_batch():
    prob = make_low_rank_problem(N=199, M=50, K=RANK)
    W = prob["W"].copy()
    W[4, :] = 0.0  # a spectrum masked in every pixel
    W[:, 5] = 0.0  # a pixel masked in every spectrum
    model = sharded_model(4, 64, rank=RANK, robust=True, conv_tol=0.0)
    state, loss = model.fit(prob["Y"], W, max_iter=10, conv_check_cadence=99)
    assert np.all(np.isfinite(np.asarray(state.A)))
    assert np.all(np.isfinite(np.asarray(state.G)))
    assert np.all(np.isfinite(np.asarray(loss)))
    assert np.allclose(np.asarray(state.A)[4], 0.0)
    assert np.allclose(np.asarray(state.G)[5], 0.0)


def test_padding_rows_are_inert(problem):
    """The whole padding scheme rests on zero-weight rows contributing nothing.

    A mesh that pads and one that does not must give the same loss.
    """
    model_padded = sharded_model(4, 64, rank=RANK, robust=True, conv_tol=0.0)
    model_exact = sharded_model(1, 199, rank=RANK, robust=True, conv_tol=0.0)
    _, loss_padded = model_padded.fit(
        problem["Y"], problem["W"], max_iter=8, conv_check_cadence=99
    )
    _, loss_exact = model_exact.fit(problem["Y"], problem["W"], max_iter=8, conv_check_cadence=99)
    assert plan_rows(199, 4, 64).n_pad > 0  # the padded case really does pad
    assert plan_rows(199, 1, 199).n_pad == 0
    assert np.max(np.abs(np.asarray(loss_padded) - np.asarray(loss_exact))) < 1e-9


# ----------------------------
# Initialisation
# ----------------------------
@pytest.mark.parametrize("n_devices", DEVICE_COUNTS)
def test_gram_init_matches_svd_init(problem, n_devices):
    """The Gram route targets the same subspace as svd_init.

    Equivalent in exact arithmetic, but forming ``Y.T @ Y`` squares the
    condition number, so agreement is at ~sqrt(eps) rather than eps. The
    difference is washed out by the first few ALS iterations -- the fitted G
    agrees to 1e-12, see the fit-equivalence tests above.
    """
    from robusta_hmf.initialisation import Initialiser

    N, M = problem["Y"].shape
    ref = Initialiser(N=N, M=M, K=RANK, strategy="svd").execute(Y=jnp.array(problem["Y"]))

    frame = ShardedFrame(
        row_block=64, devices=cpu_devices(n_devices), store_dtype=np.float64, verbose=False
    )
    data, _ = frame.prepare(problem["Y"], problem["W"])
    got = GramInitialiser(N=N, M=M, K=RANK, frame=frame).execute(Y=data)

    assert np.max(np.abs(align_signs(ref.G, got.G) - np.asarray(ref.G))) < 1e-6
    A = np.asarray(got.A)[:N]
    assert np.max(np.abs(align_signs(ref.A, A) - np.asarray(ref.A))) < 1e-6
    # A covers the padded row grid, and the padding is zero.
    assert got.A.shape[0] == data.plan.n_padded
    assert np.allclose(np.asarray(got.A)[N:], 0.0)


def test_random_init_is_reproducible_and_sharded(problem):
    frame = ShardedFrame(row_block=64, devices=cpu_devices(4), verbose=False)
    data, _ = frame.prepare(problem["Y"], problem["W"])
    init = GramInitialiser(N=199, M=50, K=RANK, strategy="random", frame=frame)
    a = init.execute(seed=3, Y=data)
    b = init.execute(seed=3, Y=data)
    assert np.allclose(np.asarray(a.A), np.asarray(b.A))
    assert a.A.shape == (data.plan.n_padded, RANK)
    with pytest.raises(ValueError, match="seed"):
        init.execute(seed=None, Y=data)


def test_gram_init_needs_prepared_data():
    frame = ShardedFrame(devices=cpu_devices(1), verbose=False)
    init = GramInitialiser(N=4, M=3, K=2, frame=frame)
    with pytest.raises(TypeError, match="ShardedData"):
        init.execute(Y=np.zeros((4, 3)))


def test_gram_init_needs_a_frame():
    with pytest.raises(ValueError, match="mesh"):
        GramInitialiser(N=4, M=3, K=2).execute(Y=None)


def test_bare_gram_initialiser_is_bound_by_the_frame(problem):
    """The documented spelling: neither the mesh nor the shapes stated up front."""
    model = Robusta(
        rank=RANK,
        robust=True,
        conv_tol=0.0,
        override_frame=ShardedFrame(
            row_block=64, devices=cpu_devices(4), store_dtype=np.float64, verbose=False
        ),
        override_initialiser=GramInitialiser(),
    )
    state, _ = model.fit(problem["Y"], problem["W"], max_iter=8, conv_check_cadence=99)
    assert state.A.shape == (199, RANK)

    ref = sharded_model(4, 64, rank=RANK, robust=True, conv_tol=0.0)
    ref_state, _ = ref.fit(problem["Y"], problem["W"], max_iter=8, conv_check_cadence=99)
    assert np.max(np.abs(align_signs(ref_state.G, state.G) - np.asarray(ref_state.G))) < 1e-12


def test_plain_initialiser_is_rerouted_through_the_gram_path(problem):
    """A dense Initialiser would be handed a ShardedData and die inside svd()."""
    from robusta_hmf.initialisation import Initialiser

    model = Robusta(
        rank=RANK,
        robust=True,
        conv_tol=0.0,
        override_frame=ShardedFrame(
            row_block=64, devices=cpu_devices(4), store_dtype=np.float64, verbose=False
        ),
        override_initialiser=Initialiser(strategy="svd"),
    )
    state, _ = model.fit(problem["Y"], problem["W"], max_iter=8, conv_check_cadence=99)
    assert isinstance(model._initialiser, GramInitialiser)
    assert state.A.shape == (199, RANK)
    assert np.all(np.isfinite(np.asarray(state.G)))


def test_rank_above_min_dimension_is_rejected(problem):
    model = sharded_model(1, 64, rank=99, robust=False)
    with pytest.raises(ValueError, match="exceeds"):
        model.fit(problem["Y"][:10], problem["W"][:10], max_iter=1)


# ----------------------------
# infer() and score()
# ----------------------------
@pytest.mark.parametrize("n_devices", DEVICE_COUNTS)
def test_infer_matches_dense_path(problem, dense_fit, n_devices):
    """Same global stopping rule as Robusta.infer, so the two agree exactly."""
    ref_model, ref_state, _ = dense_fit
    ref_infer, ref_loss = ref_model.infer(
        jnp.array(problem["Y"]),
        jnp.array(problem["W"]),
        state=ref_state,
        max_iter=50,
        conv_tol=1e-10,
    )

    model = sharded_model(n_devices, 64, rank=RANK, robust=True, robust_scale=3.0, conv_tol=0.0)
    infer, loss = model.infer(
        problem["Y"], problem["W"], state=ref_state, max_iter=50, conv_tol=1e-10
    )

    assert infer.A.shape == (199, RANK)
    assert np.allclose(np.asarray(infer.G), np.asarray(ref_state.G))  # basis stays fixed
    assert len(loss) == len(ref_loss)  # same number of iterations
    assert np.max(np.abs(np.asarray(infer.A) - np.asarray(ref_infer.A))) < 1e-10
    assert np.max(np.abs(np.asarray(loss) - np.asarray(ref_loss))) < 1e-9


@pytest.mark.parametrize("n_devices", DEVICE_COUNTS)
def test_score_matches_dense_metrics(problem, dense_fit, n_devices):
    """Chunked accumulation must give the same moments as the dense route."""
    ref_model, ref_state, _ = dense_fit
    ref = z_moments(
        jnp.array(problem["Y"]),
        jnp.array(problem["W"]),
        ref_state.A,
        ref_state.G,
        ref_model._hmf.likelihood,
    )

    model = sharded_model(n_devices, 64, rank=RANK, robust=True, robust_scale=3.0, conv_tol=0.0)
    got = model.score(problem["Y"], problem["W"], state=ref_state)

    assert got.n == ref.n  # padding rows excluded from the count
    assert abs(got.kl - ref.kl) < 1e-10
    assert abs(got.std_z - ref.std_z) < 1e-12
    assert abs(got.chi2_red - ref.chi2_red) < 1e-10
    assert abs(got.mean_z - ref.mean_z) < 1e-12


def test_score_needs_a_state(problem):
    model = sharded_model(1, 64, rank=RANK)
    with pytest.raises(ValueError, match="fit"):
        model.score(problem["Y"], problem["W"])


# ----------------------------
# Reusing placed data across a (K, Q) grid
# ----------------------------
def test_prepared_data_is_reusable_across_models(problem):
    """The reason ShardedData is public: a grid must not re-transfer the data."""
    frame = ShardedFrame(
        row_block=64, devices=cpu_devices(4), store_dtype=np.float64, verbose=False
    )
    data, _ = frame.prepare(problem["Y"], problem["W"])

    fits = {}
    for rank in (2, 3):
        model = Robusta(
            rank=rank,
            robust=True,
            conv_tol=0.0,
            override_frame=ShardedFrame(
                row_block=64, devices=cpu_devices(4), store_dtype=np.float64, verbose=False
            ),
        )
        state, _ = model.fit(data, max_iter=8, conv_check_cadence=99)
        fits[rank] = state
        assert state.A.shape == (199, rank)

    # Same Y and W arrays served both fits.
    direct = sharded_model(4, 64, rank=3, robust=True, conv_tol=0.0)
    ref_state, _ = direct.fit(problem["Y"], problem["W"], max_iter=8, conv_check_cadence=99)
    assert np.max(np.abs(align_signs(ref_state.G, fits[3].G) - np.asarray(ref_state.G))) < 1e-12


# ----------------------------
# Unsupported configurations fail loudly
# ----------------------------
def test_sgd_is_rejected(problem):
    model = Robusta(
        rank=RANK,
        method="sgd",
        override_frame=ShardedFrame(row_block=64, devices=cpu_devices(1), verbose=False),
    )
    with pytest.raises(NotImplementedError, match="sgd"):
        model.fit(problem["Y"], problem["W"], max_iter=2)


def test_unbound_frame_is_rejected(problem):
    frame = ShardedFrame(row_block=64, devices=cpu_devices(1), verbose=False)
    with pytest.raises(ValueError, match="not bound"):
        frame.run(problem["Y"], problem["W"], max_iter=1)


def test_wrong_A_shape_is_rejected(problem):
    frame = ShardedFrame(row_block=64, devices=cpu_devices(4), verbose=False)
    with pytest.raises(ValueError, match="rows, expected"):
        frame._pad_A(np.zeros((7, RANK)), plan_rows(199, 4, 64))


# ----------------------------
# Chunked robust weights
# ----------------------------
def test_robust_weights_blocks_tile_the_full_matrix(problem, dense_fit):
    ref_model, ref_state, _ = dense_fit
    expected = np.asarray(
        ref_model.robust_weights(jnp.array(problem["Y"]), jnp.array(problem["W"]), state=ref_state)
    )
    model = sharded_model(1, 64, rank=RANK, robust=True, robust_scale=3.0)
    blocks = list(
        model._frame.robust_weights_blocks(problem["Y"], problem["W"], ref_state, row_block=37)
    )
    got = np.concatenate(blocks, axis=0)
    assert got.shape == expected.shape
    assert np.max(np.abs(got - expected)) < 1e-12
