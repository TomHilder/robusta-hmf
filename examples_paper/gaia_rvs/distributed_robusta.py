"""Multi-GPU, memory-bounded variant of :class:`robusta_hmf.Robusta`.

Why this exists
---------------
``Robusta.fit`` evaluates every ALS step over the whole ``(N, M)`` matrix at
once. Each step materialises roughly five full-size intermediates -- ``A @ G.T``,
the squared residuals, the IRLS weights, ``W * Y``, and the loss residuals --
on top of the resident ``Y`` and ``W``. For the full Gaia RVS training set
(N = 499_822 rows, M = 2321 pixels) in float64 a single ``(N, M)`` array is
8.6 GiB, so a step needs ~43 GiB of scratch on top of 17 GiB of data. That is
why subsampling every 5th spectrum fits in a 48 GiB A6000 and every 4th does
not.

Two independent changes fix it, and this module applies both:

1. **Row chunking.** Every quantity the ALS steps need is either row-local
   (the A step) or a *sum over rows* (the G step, the loss). So a single pass
   over blocks of rows can compute the new ``A`` block by block while
   accumulating the G-step normal equations, whose size is ``(M, K, K)`` --
   1.8 MiB at K=10, independent of N. Peak scratch drops from ``O(N * M)`` to
   ``O(row_block * M)``: 73 MiB per temporary at ``row_block=4096``.

2. **Row sharding.** With the row axis chunked it is trivially also
   *shardable*: ``Y``, ``W`` and ``A`` are split across devices along rows,
   ``G`` is replicated (0.18 MiB), and a single ``psum`` of the small ``(M, K, K)``
   and ``(M, K)`` blocks per iteration is the only communication. The loss is
   one more ``psum`` of a scalar.

A third, free win: ``Y`` and ``W`` are *stored* in float32 and cast to float64
per chunk. The HDF5 fluxes and uncertainties are float32 to begin with, so
nothing is lost, while the resident data halves to 8.6 GiB total and every
accumulation (normal equations, loss) still happens in float64.

The SVD initialiser is replaced by a mathematically equivalent Gram-matrix
route, because ``jnp.linalg.svd`` on a (499_822, 2321) matrix wants a
``U`` the size of ``Y`` itself. Instead we accumulate ``C = Y.T @ Y``
(41 MiB), take its top-K eigenvectors, and recover ``A`` in one chunked pass;
for ``Y = U S V.T`` this gives exactly the same ``A = U sqrt(S)``,
``G = V sqrt(S)`` as :func:`robusta_hmf.initialisation.svd_init`.

Everything else -- the Student-t IRLS weights, the ALS normal equations, the
rotation, the convergence test -- is imported from the library rather than
reimplemented, so this stays a scheduling variant and not a second model.

Usage
-----
    model = DistributedRobusta(rank=10, robust_scale=2.0, row_block=4096)
    state, loss = model.fit(Y, W, max_iter=100, conv_check_cadence=1)

``Y`` and ``W`` are plain numpy arrays; the sharded device arrays are built
shard by shard so the host never holds a second copy.
"""

import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from robusta_hmf.convergence import ConvergenceTester
from robusta_hmf.likelihoods import GaussianLikelihood, StudentTLikelihood
from robusta_hmf.rotations import get_rotation_cls
from robusta_hmf.state import RHMFState

ROW_AXIS = "rows"


# ---------------------------------------------------------------------------- #
# Row plan: how to split N rows over devices and chunks
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RowPlan:
    """How ``n_rows`` real rows are laid out over devices and scan chunks.

    ``shard_map`` needs every device to hold the same number of rows, and the
    inner ``lax.scan`` needs that number to be an exact multiple of the chunk
    size, so the row axis is padded up to ``n_padded``. The chunk size is
    shrunk from the requested ``row_block`` where that makes the padding
    smaller -- for the full RVS sample on 4 devices this is 18 padded rows
    rather than the 8082 a fixed 8192-row block would need.

    Padding rows are all-zero in both ``Y`` and ``W``. Zero weights make them
    inert: they contribute nothing to the G-step sums and nothing to the loss
    (``log1p(0) == 0``), and their A-step normal equations are exactly zero,
    which :func:`_solve_batch` detects and replaces with the identity so the
    batched solve stays non-singular.
    """

    n_rows: int
    n_devices: int
    chunk: int
    n_chunks: int

    @property
    def rows_per_device(self) -> int:
        return self.chunk * self.n_chunks

    @property
    def n_padded(self) -> int:
        return self.rows_per_device * self.n_devices

    @property
    def n_pad(self) -> int:
        return self.n_padded - self.n_rows


def plan_rows(n_rows: int, n_devices: int, row_block: int) -> RowPlan:
    per_device = -(-n_rows // n_devices)
    n_chunks = max(1, -(-per_device // row_block))
    chunk = -(-per_device // n_chunks)
    return RowPlan(n_rows=n_rows, n_devices=n_devices, chunk=chunk, n_chunks=n_chunks)


def build_mesh(devices=None) -> Mesh:
    """A 1-D mesh over the row axis. Works unchanged with a single device."""
    devices = list(jax.devices()) if devices is None else list(devices)
    return Mesh(np.asarray(devices), (ROW_AXIS,))


def shard_rows(x: np.ndarray, plan: RowPlan, mesh: Mesh, dtype) -> Array:
    """Place ``x`` on the mesh, split along rows and zero-padded to the plan.

    Built with ``make_array_from_callback`` so only one device's shard is
    materialised at a time; ``jnp.asarray`` followed by ``device_put`` would
    stage a full second copy of an 8.6 GiB array on the host.
    """
    sharding = NamedSharding(mesh, P(ROW_AXIS, None))
    shape = (plan.n_padded, *x.shape[1:])

    def callback(index):
        rows = index[0]
        start, stop, _ = rows.indices(plan.n_padded)
        out = np.zeros((stop - start, *x.shape[1:]), dtype=dtype)
        real = min(stop, x.shape[0])
        if real > start:
            out[: real - start] = x[start:real]
        return out

    return jax.make_array_from_callback(shape, sharding, callback)


# ---------------------------------------------------------------------------- #
# Building blocks (row-local, so they are identical on and off the mesh)
# ---------------------------------------------------------------------------- #


def _outer(X: Array) -> Array:
    """Row-wise outer products, flattened: (n, K) -> (n, K*K)."""
    return (X[:, :, None] * X[:, None, :]).reshape(X.shape[0], -1)


def _solve_batch(M: Array, b: Array, ridge: float | None) -> Array:
    """Batched solve of the ALS normal equations, tolerant of empty rows.

    A row (or, in the G step, a pixel) whose weights are all zero gives
    ``M == 0`` exactly. That happens for padding rows, for spectra that are
    NaN throughout, and for pixels masked in every spectrum. Substituting the
    identity there returns a zero factor for that row instead of poisoning the
    whole batched solve with NaNs, and is exact: such rows enter no downstream
    sum with non-zero weight.
    """
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    if ridge is not None:
        M = M + ridge * eye
    empty = jnp.all(M == 0, axis=(-2, -1))
    M = jnp.where(empty[:, None, None], eye, M)
    b = jnp.where(empty[:, None], jnp.zeros_like(b), b)
    return jnp.linalg.solve(M, b[..., None])[..., 0]


# ---------------------------------------------------------------------------- #
# The model
# ---------------------------------------------------------------------------- #


class DistributedRobusta:
    """Row-sharded, row-chunked ALS. Mirrors the ``Robusta`` ALS path.

    Parameters match :class:`robusta_hmf.Robusta` where they overlap. The
    extra ones are:

    row_block : int
        Target rows per scan chunk. Sets peak scratch memory:
        ``~6 * row_block * M * 8`` bytes per device. 4096 is ~440 MiB at
        M=2321. Larger is marginally faster, smaller is safer.
    devices : list | None
        Devices to shard over. Defaults to all visible devices.
    store_dtype : np.dtype
        On-device dtype for Y and W. float32 by default -- the inputs are
        float32 measurements, and all arithmetic still happens in
        ``compute_dtype``.
    compute_dtype : np.dtype | None
        Dtype of the arithmetic and of A/G. Defaults to float64 when
        ``jax_enable_x64`` is set, float32 otherwise.
    """

    def __init__(
        self,
        rank: int,
        robust: bool = True,
        robust_nu: float = 1.0,
        robust_scale: float = 1.0,
        als_ridge: float | None = None,
        rotation: str = "fast",
        conv_strategy: str = "max_frac_G",
        conv_tol: float = 1e-3,
        row_block: int = 4096,
        devices=None,
        store_dtype=np.float32,
        compute_dtype=None,
        **rotation_kwargs,
    ):
        self.rank = rank
        self.ridge = als_ridge
        self.row_block = row_block
        self.mesh = build_mesh(devices)
        self.store_dtype = np.dtype(store_dtype)
        if compute_dtype is None:
            compute_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        self.compute_dtype = np.dtype(compute_dtype)

        if robust:
            self.likelihood = StudentTLikelihood(nu=float(robust_nu), scale=float(robust_scale))
        else:
            self.likelihood = GaussianLikelihood()
        self.rotation = get_rotation_cls(method=rotation)(**rotation_kwargs)
        self.conv_tester = ConvergenceTester(strategy=conv_strategy, tol=conv_tol)

        self._state = None
        self._loss_history = None

    # -- properties mirroring Robusta ---------------------------------------- #

    @property
    def n_devices(self) -> int:
        return self.mesh.size

    @property
    def A(self):
        return self._state.A if self._state is not None else None

    @property
    def G(self):
        return self._state.G if self._state is not None else None

    @property
    def state(self):
        return self._state

    @property
    def loss_history(self):
        return self._loss_history

    # -- one fused ALS pass over the rows ------------------------------------ #

    def _make_pass_one(self, plan: RowPlan):
        """A step + G-step accumulation in a single sweep over the rows.

        One pass suffices because the G step needs the *new* A: within a chunk
        we form the IRLS weights from the old (A, G), solve that chunk's rows
        of A, then immediately fold those rows into the running ``(M, K, K)``
        and ``(M, K)`` accumulators. Both accumulators are tiny and are summed
        across devices with a single ``psum``.
        """
        K, chunk, cdtype = self.rank, plan.chunk, self.compute_dtype
        likelihood, ridge = self.likelihood, self.ridge

        def local(Y_l, W_l, A_l, G):
            n_local, M = Y_l.shape
            nc = n_local // chunk
            GG = _outer(G)  # (M, K*K)

            def body(carry, xs):
                MG, bG = carry
                y, w, a = xs
                y = y.astype(cdtype)
                w = w.astype(cdtype)
                # Total weights = data weights * robust IRLS downweighting.
                Wt = likelihood.weights_total(y, w, a, G)  # (chunk, M)
                WY = Wt * y
                # A step: M[i] = G.T diag(W[i]) G, b[i] = G.T (W[i] * Y[i]).
                a_new = _solve_batch((Wt @ GG).reshape(chunk, K, K), WY @ G, ridge)
                # G step, accumulated: MG[m] = sum_n W[n,m] a[n] a[n].T
                #                      bG[m] = sum_n (W*Y)[n,m] a[n]
                MG = MG + Wt.T @ _outer(a_new)
                bG = bG + WY.T @ a_new
                return (MG, bG), a_new

            init = (jnp.zeros((Y_l.shape[1], K * K), cdtype), jnp.zeros((Y_l.shape[1], K), cdtype))
            (MG, bG), A_new = jax.lax.scan(
                body,
                init,
                (Y_l.reshape(nc, chunk, M), W_l.reshape(nc, chunk, M), A_l.reshape(nc, chunk, K)),
            )
            return (
                jax.lax.psum(MG, ROW_AXIS),
                jax.lax.psum(bG, ROW_AXIS),
                A_new.reshape(n_local, K),
            )

        return shard_map(
            local,
            mesh=self.mesh,
            in_specs=(P(ROW_AXIS, None), P(ROW_AXIS, None), P(ROW_AXIS, None), P()),
            out_specs=(P(), P(), P(ROW_AXIS, None)),
            # jnp.linalg.solve lowers to custom_linear_solve, which has no
            # replication rule, so shard_map cannot statically prove the psum'd
            # outputs are replicated. They are -- psum guarantees it -- so
            # switch the check off rather than hand-rolling the batched solve.
            check_rep=False,
        )

    def _make_pass_two(self, plan: RowPlan):
        """Chunked, summed loss at the post-rotation (A, G)."""
        K, chunk, cdtype = self.rank, plan.chunk, self.compute_dtype
        likelihood = self.likelihood

        def local(Y_l, W_l, A_l, G):
            n_local, M = Y_l.shape
            nc = n_local // chunk

            def body(total, xs):
                y, w, a = xs
                return total + likelihood.loss(y.astype(cdtype), w.astype(cdtype), a, G), None

            total, _ = jax.lax.scan(
                body,
                jnp.zeros((), cdtype),
                (Y_l.reshape(nc, chunk, M), W_l.reshape(nc, chunk, M), A_l.reshape(nc, chunk, K)),
            )
            return jax.lax.psum(total, ROW_AXIS)

        return shard_map(
            local,
            mesh=self.mesh,
            in_specs=(P(ROW_AXIS, None), P(ROW_AXIS, None), P(ROW_AXIS, None), P()),
            out_specs=P(),
        )

    def _make_step(self, plan: RowPlan):
        pass_one = self._make_pass_one(plan)
        pass_two = self._make_pass_two(plan)
        K, ridge, rotation = self.rank, self.ridge, self.rotation

        def step(Y, W, A, G, rotate):
            MG, bG, A_new = pass_one(Y, W, A, G)
            G_new = _solve_batch(MG.reshape(MG.shape[0], K, K), bG, ridge)
            state = RHMFState(A=A_new, G=G_new, it=0)
            if rotate:
                # Rotation is driven by G (replicated, (M, K)); the only
                # sharded work is A @ R_inv, which is row-local.
                state = rotation(state)
            loss = pass_two(Y, W, state.A, state.G)
            return state.A, state.G, loss

        return jax.jit(step, static_argnums=(4,))

    # -- initialisation ------------------------------------------------------ #

    def _make_gram(self, plan: RowPlan):
        chunk, cdtype = plan.chunk, self.compute_dtype

        def local(Y_l):
            nc, M = Y_l.shape[0] // chunk, Y_l.shape[1]

            def body(C, y):
                y = y.astype(cdtype)
                return C + y.T @ y, None

            C, _ = jax.lax.scan(body, jnp.zeros((M, M), cdtype), Y_l.reshape(nc, chunk, M))
            return jax.lax.psum(C, ROW_AXIS)

        return jax.jit(
            shard_map(local, mesh=self.mesh, in_specs=(P(ROW_AXIS, None),), out_specs=P())
        )

    def _make_project(self, plan: RowPlan):
        """A = (Y @ V) * scale, chunked so Y is never cast to fp64 wholesale."""
        chunk, cdtype = plan.chunk, self.compute_dtype

        def local(Y_l, V, scale):
            nc, M = Y_l.shape[0] // chunk, Y_l.shape[1]

            def body(_, y):
                return None, (y.astype(cdtype) @ V) * scale

            _, A = jax.lax.scan(body, None, Y_l.reshape(nc, chunk, M))
            return A.reshape(Y_l.shape[0], -1)

        return jax.jit(
            shard_map(
                local,
                mesh=self.mesh,
                in_specs=(P(ROW_AXIS, None), P(), P()),
                out_specs=P(ROW_AXIS, None),
            )
        )

    def _init_svd(self, Y_dev, plan: RowPlan):
        """Top-K SVD initialisation via the Gram matrix.

        Identical in exact arithmetic to ``svd_init``: for ``Y = U S V.T``,
        ``Y.T @ Y = V S^2 V.T``, so the top-K eigenvectors of the Gram matrix
        are ``V[:, :K]`` and ``Y @ V = U S``. We return ``A = U sqrt(S)`` and
        ``G = V sqrt(S)``, matching the library's convention, without ever
        forming a (N, min(N, M)) ``U``.
        """
        K, cdtype = self.rank, self.compute_dtype
        C = np.asarray(self._make_gram(plan)(Y_dev))
        # eigh on the host: (M, M) is 41 MiB, this runs once, and LAPACK
        # avoids any dependence on cuSOLVER's float64 workspace behaviour.
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1][:K]
        V = np.ascontiguousarray(evecs[:, order]).astype(cdtype)
        sv = np.sqrt(np.clip(evals[order], 0.0, None)).astype(cdtype)  # singular values
        root = np.sqrt(sv)
        scale = np.where(root > 0, 1.0 / np.where(root > 0, root, 1.0), 0.0).astype(cdtype)
        A = self._make_project(plan)(Y_dev, jnp.asarray(V), jnp.asarray(scale))
        G = jnp.asarray(V * root)
        return RHMFState(A=A, G=G, it=0)

    def _init_random(self, plan: RowPlan, M: int, seed: int):
        key = jax.random.key(seed) if hasattr(jax.random, "key") else jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (plan.n_padded, self.rank), dtype=self.compute_dtype)
        A = jax.device_put(A, NamedSharding(self.mesh, P(ROW_AXIS, None)))
        G = jax.random.normal(k2, (M, self.rank), dtype=self.compute_dtype)
        return RHMFState(A=A, G=G, it=0)

    # -- fit ----------------------------------------------------------------- #

    def fit(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        max_iter: int = 1000,
        conv_check_cadence: int = 10,
        seed: int = 0,
        init_state: RHMFState | None = None,
        init_strategy: str = "svd",
        verbose: bool = True,
    ) -> tuple[RHMFState, Array]:
        """Fit on ``Y``, ``W`` (numpy, shape (N, M)). Returns (state, losses).

        The returned ``state.A`` has exactly ``N`` rows; the padding used on
        device is stripped.
        """
        n_rows, M = Y.shape
        if W.shape != Y.shape:
            raise ValueError(f"Y and W must have the same shape, got {Y.shape} and {W.shape}")
        plan = plan_rows(n_rows, self.n_devices, self.row_block)

        if verbose:
            per_dev = plan.rows_per_device * M
            print(
                f"Sharding {n_rows} x {M} over {self.n_devices} device(s): "
                f"{plan.n_chunks} chunk(s) of {plan.chunk} rows each, "
                f"{plan.rows_per_device} rows/device (+{plan.n_pad} padded)\n"
                f"  resident per device: {2 * per_dev * self.store_dtype.itemsize / 2**30:.2f} GiB"
                f"  (Y+W as {self.store_dtype.name})\n"
                f"  scratch per device:  "
                f"~{6 * plan.chunk * M * self.compute_dtype.itemsize / 2**30:.2f} GiB"
                f"  (compute in {self.compute_dtype.name})",
                flush=True,
            )

        Y_dev = shard_rows(Y, plan, self.mesh, self.store_dtype)
        W_dev = shard_rows(W, plan, self.mesh, self.store_dtype)

        if init_state is None:
            t0 = time.time()
            if verbose:
                print(f"Initialising ({init_strategy})... ", end="", flush=True)
            if init_strategy == "svd":
                state = self._init_svd(Y_dev, plan)
            elif init_strategy == "random":
                state = self._init_random(plan, M, seed)
            else:
                raise ValueError(f"Unknown init_strategy: {init_strategy!r}")
            if verbose:
                print(f"done in {time.time() - t0:.1f} s.", flush=True)
        else:
            A = init_state.A
            if A.shape[0] == n_rows:
                A = shard_rows(np.asarray(A), plan, self.mesh, self.compute_dtype)
            elif A.shape[0] != plan.n_padded:
                raise ValueError(
                    f"init_state.A has {A.shape[0]} rows, expected {n_rows} or {plan.n_padded}"
                )
            state = RHMFState(A=A, G=jnp.asarray(init_state.G, self.compute_dtype), it=0)

        step = self._make_step(plan)
        A, G = state.A, state.G
        loss_history = []
        prev_state, prev_loss = state, jnp.inf
        it0 = int(state.it)

        for i in range(max_iter):
            # Matches OptFrame: ALS rotates every iteration except the first.
            rotate = i != 0
            t0 = time.time()
            A, G, loss = step(Y_dev, W_dev, A, G, rotate)
            loss_history.append(loss)
            if i % conv_check_cadence == 0 and i != 0:
                loss = float(loss)  # forces the step to complete before timing
                state = RHMFState(A=A, G=G, it=it0 + i + 1)
                if self.conv_tester.is_converged(prev_state, state, prev_loss, loss):
                    if verbose:
                        print(f"Converged at iteration {i}", flush=True)
                    break
                prev_state, prev_loss = state, loss
                if verbose:
                    print(
                        f"iter {it0 + i + 1:03d} | loss {loss:.6f} | {time.time() - t0:.2f} s/it",
                        flush=True,
                    )
                if not np.isfinite(loss):
                    raise FloatingPointError(f"Loss is not finite at iteration {i}: {loss}")

        state = RHMFState(A=A[:n_rows], G=G, it=it0 + len(loss_history))
        self._state = state
        self._loss_history = jnp.array(loss_history)
        return state, self._loss_history

    # -- inference helpers --------------------------------------------------- #

    def robust_weights_chunked(self, Y, W, state=None, row_block=None):
        """IRLS weights, returned one row-block at a time as numpy arrays.

        The full weight matrix is the same size as ``Y``, so this is a
        generator rather than a single array: consume it to compute per-row
        summaries (outlier scores) without ever holding an (N, M) fp64 array.
        """
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        row_block = row_block or self.row_block
        cdtype = self.compute_dtype

        @jax.jit
        def block(y, w, a):
            return self.likelihood.weights_irls(y.astype(cdtype), w.astype(cdtype), a, state.G)

        for start in range(0, Y.shape[0], row_block):
            stop = min(start + row_block, Y.shape[0])
            yield np.asarray(block(Y[start:stop], W[start:stop], state.A[start:stop]))

    def synthesize(self, state=None, indices=None):
        state = state if state is not None else self._state
        if state is None:
            raise ValueError("No trained state available. Call fit() first.")
        A = state.A if indices is None else state.A[indices]
        return A @ state.G.T


# ---------------------------------------------------------------------------- #
# Reference check
# ---------------------------------------------------------------------------- #


def compare_to_reference(n=512, m=97, K=6, Q=2.0, iters=8, row_block=64, seed=0):
    """Run this model and the library ``Robusta`` on the same small problem.

    Returns ``(max_abs_G_diff, max_abs_loss_diff)``. Both should be at
    round-off level: the two take the same mathematical steps, only scheduled
    differently. Handy under
    ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` to exercise the
    sharded path without GPUs.
    """
    from robusta_hmf import Robusta

    rng = np.random.default_rng(seed)
    A_true = rng.normal(size=(n, K))
    G_true = rng.normal(size=(m, K))
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    Y = (A_true @ G_true.T + 0.1 * rng.normal(size=(n, m))).astype(dtype)
    W = np.exp(rng.normal(size=(n, m))).astype(dtype)
    Y[rng.random((n, m)) < 0.01] += 25.0  # outliers, so the robust weights bite

    kwargs = dict(rank=K, robust_scale=Q, conv_tol=0.0, rotation="fast", target="G", whiten=True)
    ref = Robusta(conv_strategy="max_frac_G", init_strategy="svd", **kwargs)
    ref_state, ref_loss = ref.fit(
        jnp.asarray(Y), jnp.asarray(W), max_iter=iters, conv_check_cadence=iters + 1
    )

    dist = DistributedRobusta(row_block=row_block, store_dtype=dtype, **kwargs)
    dist_state, dist_loss = dist.fit(
        Y, W, max_iter=iters, conv_check_cadence=iters + 1, verbose=False
    )

    # Sign of each basis vector is arbitrary between the two SVD routes.
    ref_G, dist_G = np.asarray(ref_state.G), np.asarray(dist_state.G)
    sign = np.sign(np.sum(ref_G * dist_G, axis=0))
    dG = np.max(np.abs(ref_G - dist_G * sign))
    dL = np.max(np.abs(np.asarray(ref_loss) - np.asarray(dist_loss)))
    return float(dG), float(dL)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}")
    dG, dL = compare_to_reference()
    print(f"max |dG| = {dG:.3e}   max |dloss| = {dL:.3e}")
