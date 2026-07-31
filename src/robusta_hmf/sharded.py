# sharded.py

"""Row-chunked, row-sharded scheduling of the ALS loop.

Why this exists
---------------
:class:`~robusta_hmf.Robusta` evaluates every ALS step over the whole ``(N, M)``
matrix at once. Each step materialises roughly five full-size intermediates --
``A @ G.T``, the squared residuals, the IRLS weights, ``W * Y``, and the loss
residuals -- on top of the resident ``Y`` and ``W``. For the full Gaia RVS
training set (N = 496_955 rows, M = 2321 pixels) in float64 a single ``(N, M)``
array is 8.6 GiB, so a step needs ~43 GiB of scratch on top of 17 GiB of data.
That is why subsampling every 5th spectrum fits in a 48 GiB A6000 and every 4th
does not.

Two independent changes fix it, and :class:`ShardedFrame` applies both:

1. **Row chunking.** Every quantity the ALS steps need is either row-local (the
   A step) or a *sum over rows* (the G step, the loss). So a single pass over
   blocks of rows can compute the new ``A`` block by block while accumulating
   the G-step normal equations, whose size is ``(M, K, K)`` -- 1.8 MiB at K=10,
   independent of N. Peak scratch drops from ``O(N * M)`` to
   ``O(row_block * M)``: 73 MiB per temporary at ``row_block=4096``.

   **Chunking alone is what fixes the out-of-memory failure**, and it needs no
   second device: a one-device mesh is the default and is fully supported.

2. **Row sharding.** With the row axis chunked it is trivially also
   *shardable*: ``Y``, ``W`` and ``A`` are split across devices along rows,
   ``G`` is replicated (0.18 MiB), and a single ``psum`` of the small
   ``(M, K, K)`` and ``(M, K)`` blocks per iteration is the only
   communication. The loss is one more ``psum`` of a scalar. Sharding is a
   speed knob on top of the memory fix, not a prerequisite for it.

A third, free win: ``Y`` and ``W`` are *stored* in float32 and cast to float64
per chunk. Flux and uncertainty measurements are typically float32 to begin
with, so nothing is lost, while the resident data halves and every accumulation
(normal equations, loss) still happens in ``compute_dtype``.

The SVD initialiser is replaced by a mathematically equivalent Gram-matrix
route (:class:`GramInitialiser`), because ``jnp.linalg.svd`` on a
(496_955, 2321) matrix wants a ``U`` the size of ``Y`` itself.

Usage
-----
::

    from robusta_hmf import Robusta
    from robusta_hmf.sharded import GramInitialiser, ShardedFrame

    model = Robusta(
        rank=10,
        robust_scale=5.0,
        override_frame=ShardedFrame(row_block=4096),  # all visible devices
    )
    state, loss_history = model.fit(Y, W)             # numpy arrays, (N, M)

``GramInitialiser`` is selected automatically when the frame is a
``ShardedFrame``; pass it explicitly via ``override_initialiser`` only if you
want to configure it.

Contracts
---------
- **Zero-weight rows and pixels must be inert.** Row padding depends on it: a
  padded row is all-zero in ``Y`` and ``W``, contributes nothing to the G-step
  sums and nothing to the loss (``log1p(0) == 0``), and its A-step normal
  equations are exactly zero, which
  :func:`~robusta_hmf.als.solve_normal_equations` detects. This holds for
  :class:`~robusta_hmf.likelihoods.GaussianLikelihood`,
  :class:`~robusta_hmf.likelihoods.CauchyLikelihood` and
  :class:`~robusta_hmf.likelihoods.StudentTLikelihood`, and is a contract on
  any :class:`~robusta_hmf.likelihoods.Likelihood` used with this frame.
- **Results are not bitwise reproducible across ``row_block`` or device
  count**, because the summation order changes. The observed spread is ~1e-12
  in the loss and ~1e-14 in ``G``, i.e. round-off, but it is not zero.
- **``method="sgd"`` is not supported** and raises. Neither is
  ``RHMFState.opt_state``, which is not carried through the loop.
"""

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from .als import normal_equations, solve_normal_equations
from .frame import OptFrame
from .initialisation import Initialiser, InitStrategy
from .metrics import ZMoments
from .state import RHMFState

ROW_AXIS = "rows"

# ---------------------------------------------------------------------------- #
# shard_map portability
# ---------------------------------------------------------------------------- #

try:  # jax >= 0.6
    from jax import shard_map as _shard_map
except ImportError:  # jax < 0.6
    from jax.experimental.shard_map import shard_map as _shard_map

# The replication-check keyword was renamed check_rep -> check_vma in jax 0.8.
# Detect it rather than branching on a version string, so this keeps working
# whichever side of the rename the installed jax is on.
_CHECK_KW = "check_vma" if "check_vma" in inspect.signature(_shard_map).parameters else "check_rep"


def _shmap(f, mesh: Mesh, in_specs, out_specs, check: bool = False):
    """``shard_map`` with the replication check normalised, and off by default.

    Every kernel here either returns a row-sharded array or a ``psum`` of a
    small accumulator, so replication of the outputs is guaranteed by the
    ``psum`` itself. The static check is nonetheless switched off, because it
    rejects both of the shapes this module is built from, and for different
    reasons on either side of the jax 0.8 rename:

    - before 0.8, ``jnp.linalg.solve`` lowers to ``custom_linear_solve``, which
      had no replication rule, so ``check_rep`` could not see through the ALS
      solve;
    - from 0.8, ``check_vma`` rejects a ``lax.scan`` whose carry starts as a
      plain ``jnp.zeros`` and accumulates values derived from sharded inputs,
      unless the initial carry is wrapped in ``jax.lax.pvary`` -- which does
      not exist on the older jax the Gaia GPU pipeline runs on.

    Rather than carry two spellings of every kernel, the check is off and the
    equivalence to the dense path is asserted by the test suite instead.
    """
    return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **{_CHECK_KW: check})


# ---------------------------------------------------------------------------- #
# Row plan: how to split N rows over devices and chunks
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RowPlan:
    """How ``n_rows`` real rows are laid out over devices and scan chunks.

    ``shard_map`` needs every device to hold the same number of rows, and the
    inner ``lax.scan`` needs that number to be an exact multiple of the chunk
    size, so the row axis is padded up to :attr:`n_padded`. The chunk size is
    shrunk from the requested ``row_block`` where that makes the padding
    smaller -- for the full RVS sample on 4 devices this is 18 padded rows
    rather than the 8082 a fixed 8192-row block would need.

    Attributes
    ----------
    n_rows : int
        Real rows, before padding.
    n_devices : int
        Devices in the mesh.
    chunk : int
        Rows per ``lax.scan`` step. Sets peak scratch memory.
    n_chunks : int
        Scan steps per device.
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
    """Choose a chunk size and chunk count that minimise padding.

    Parameters
    ----------
    n_rows : int
        Real rows in the data matrix.
    n_devices : int
        Devices to split the row axis over.
    row_block : int
        Target rows per chunk. The realised chunk is at most this.

    Returns
    -------
    plan : RowPlan
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if row_block < 1:
        raise ValueError(f"row_block must be positive, got {row_block}")
    per_device = -(-n_rows // n_devices)
    n_chunks = max(1, -(-per_device // row_block))
    chunk = -(-per_device // n_chunks)
    return RowPlan(n_rows=n_rows, n_devices=n_devices, chunk=chunk, n_chunks=n_chunks)


def build_mesh(devices: Sequence | None = None) -> Mesh:
    """A 1-D mesh over the row axis. Works unchanged with a single device."""
    devices = list(jax.devices()) if devices is None else list(devices)
    return Mesh(np.asarray(devices), (ROW_AXIS,))


def shard_rows(x: np.ndarray, plan: RowPlan, mesh: Mesh, dtype) -> Array:
    """Place ``x`` on the mesh, split along rows and zero-padded to ``plan``.

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


@dataclass(frozen=True)
class ShardedData:
    """``Y`` and ``W`` resident on a mesh, plus the plan describing their layout.

    Built once by :meth:`ShardedFrame.prepare` and reused: a (K, Q) grid over
    the same sample would otherwise re-transfer the full matrices for every
    grid point.

    Attributes
    ----------
    Y, W : Array, shape (n_padded, n_pixels)
        Sharded, zero-padded data and weights, in ``store_dtype``.
    plan : RowPlan
        The row layout.
    mesh : Mesh
        The mesh they are resident on.
    n_pixels : int
        Columns of the original matrix.
    """

    Y: Array
    W: Array
    plan: RowPlan
    mesh: Mesh
    n_pixels: int

    @property
    def n_rows(self) -> int:
        """Real rows, excluding padding."""
        return self.plan.n_rows

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_pixels)`` of the original, unpadded matrix."""
        return (self.n_rows, self.n_pixels)


def shard_data(Y, W, mesh: Mesh, row_block: int, dtype=np.float32) -> ShardedData:
    """Place ``Y`` and ``W`` (numpy, ``(N, M)``) on ``mesh``, split along rows."""
    if W is None:
        raise ValueError("W must be provided alongside Y (or pass a prepared ShardedData)")
    if W.shape != Y.shape:
        raise ValueError(f"Y and W must have the same shape, got {Y.shape} and {W.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y.shape}")
    plan = plan_rows(Y.shape[0], mesh.size, row_block)
    return ShardedData(
        Y=shard_rows(Y, plan, mesh, dtype),
        W=shard_rows(W, plan, mesh, dtype),
        plan=plan,
        mesh=mesh,
        n_pixels=Y.shape[1],
    )


# ---------------------------------------------------------------------------- #
# The frame
# ---------------------------------------------------------------------------- #


@dataclass
class ShardedFrame(OptFrame):
    """An :class:`~robusta_hmf.frame.OptFrame` that chunks and shards the rows.

    Reads the likelihood, rotation and ALS ridge off the
    :class:`~robusta_hmf.hmf.HMF` it is bound to, so it never duplicates the
    model's parameter list. Hand it to
    :class:`~robusta_hmf.Robusta` via ``override_frame``.

    Parameters
    ----------
    row_block : int, default=4096
        Target rows per scan chunk. Sets peak scratch memory, roughly
        ``6 * row_block * M * itemsize`` bytes per device -- ~440 MiB at
        M=2321, K=10, float64. Larger is marginally faster, smaller is safer.
    devices : Sequence | None, default=None
        Devices to shard over. Defaults to all visible devices; a single
        device gives pure chunking, which is the memory fix on its own.
    store_dtype : dtype, default=numpy.float32
        On-device dtype for ``Y`` and ``W``. Measurements are usually float32
        to begin with, and all arithmetic still happens in ``compute_dtype``.
    compute_dtype : dtype | None, default=None
        Dtype of the arithmetic and of ``A``/``G``. Defaults to float64 when
        ``jax_enable_x64`` is set, float32 otherwise.
    verbose : bool, default=True
        Print the sharding and memory summary when the data is placed.

    Notes
    -----
    Results are not bitwise identical to :class:`~robusta_hmf.Robusta`'s dense
    path, nor across different ``row_block`` or device counts: the arithmetic
    is the same but the summation order is not. See the module docstring.
    """

    # Redeclared with defaults so a user can construct the frame with only its
    # scheduling parameters; Robusta fills these in via OptFrame.bind().
    method: Any = None
    conv_tester: Any = None

    row_block: int = 4096
    devices: Sequence | None = None
    store_dtype: Any = np.float32
    compute_dtype: Any = None
    verbose: bool = True

    mesh: Mesh = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.mesh = build_mesh(self.devices)
        self.store_dtype = np.dtype(self.store_dtype)
        if self.compute_dtype is None:
            self.compute_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        self.compute_dtype = np.dtype(self.compute_dtype)

    @property
    def n_devices(self) -> int:
        return self.mesh.size

    def _check_method(self):
        if self.method is None:
            raise ValueError("ShardedFrame is not bound to a model; pass it to Robusta.")
        if self.method.opt_method != "als":
            raise NotImplementedError(
                f"ShardedFrame supports method='als' only, got {self.method.opt_method!r}. "
                "The sharded loop has no gradient path and does not carry opt_state."
            )

    @property
    def _ridge(self) -> float | None:
        return self.method.a_step.ridge

    # -- data placement ------------------------------------------------------ #

    def prepare(self, Y, W=None) -> tuple[ShardedData, None]:
        """Place ``(Y, W)`` on the mesh, or pass through an existing handle.

        Returns ``(data, None)``: the weights live inside the
        :class:`ShardedData`, so there is no second array to carry.
        """
        if isinstance(Y, ShardedData):
            return Y, None
        data = shard_data(
            np.asarray(Y), np.asarray(W), self.mesh, self.row_block, self.store_dtype
        )
        if self.verbose:
            self._print_layout(data)
        return data, None

    def _print_layout(self, data: ShardedData):
        plan, n_pix = data.plan, data.n_pixels
        per_dev = plan.rows_per_device * n_pix
        print(
            f"Sharding {data.n_rows} x {n_pix} over {self.n_devices} device(s): "
            f"{plan.n_chunks} chunk(s) of {plan.chunk} rows each, "
            f"{plan.rows_per_device} rows/device (+{plan.n_pad} padded)\n"
            f"  resident per device: "
            f"{2 * per_dev * self.store_dtype.itemsize / 2**30:.2f} GiB"
            f"  (Y+W as {self.store_dtype.name})\n"
            f"  scratch per device:  "
            f"~{6 * plan.chunk * n_pix * self.compute_dtype.itemsize / 2**30:.2f} GiB"
            f"  (compute in {self.compute_dtype.name})",
            flush=True,
        )

    def make_initialiser(self, N: int, M: int, K: int, strategy: InitStrategy) -> Initialiser:
        """A :class:`GramInitialiser`, which is the only kind this frame can use."""
        return GramInitialiser(N=N, M=M, K=K, strategy=strategy, frame=self)

    def _pad_A(self, A, plan: RowPlan) -> Array:
        """Coerce an ``A`` of ``n_rows`` or ``n_padded`` rows onto the mesh."""
        if A.shape[0] == plan.n_padded and isinstance(A, jax.Array):
            return A.astype(self.compute_dtype)
        if A.shape[0] not in (plan.n_rows, plan.n_padded):
            raise ValueError(f"A has {A.shape[0]} rows, expected {plan.n_rows} or {plan.n_padded}")
        return shard_rows(np.asarray(A), plan, self.mesh, self.compute_dtype)

    # -- one fused ALS pass over the rows ------------------------------------ #

    def _make_pass_one(self, plan: RowPlan, skip_G: bool):
        """A step, plus G-step accumulation, in a single sweep over the rows.

        One pass suffices because the G step needs the *new* A: within a chunk
        we form the IRLS weights from the old ``(A, G)``, solve that chunk's
        rows of ``A``, then immediately fold those rows into the running
        ``(M, K, K)`` and ``(M, K)`` accumulators. Both accumulators are tiny
        and are summed across devices with a single ``psum``.
        """
        chunk, cdtype = plan.chunk, self.compute_dtype
        likelihood, ridge = self.method.likelihood, self._ridge

        def local(Y_l, W_l, A_l, G):
            n_local, n_pix = Y_l.shape
            K = G.shape[1]
            nc = n_local // chunk

            def body(carry, xs):
                y, w, a = xs
                y = y.astype(cdtype)
                w = w.astype(cdtype)
                # Total weights = data weights * robust IRLS downweighting.
                Wt = likelihood.weights_total(y, w, a, G)  # (chunk, n_pix)
                WY = Wt * y
                # A step: row-local normal equations against the fixed G.
                a_new = solve_normal_equations(*normal_equations(Wt, WY, G), ridge)
                if skip_G:
                    return carry, a_new
                # G step, accumulated: the same kernel with the roles of the
                # two factors swapped, summed over chunks and then over devices.
                MG, bG = carry
                MG_c, bG_c = normal_equations(Wt.T, WY.T, a_new)
                return (MG + MG_c, bG + bG_c), a_new

            blocks = (
                Y_l.reshape(nc, chunk, n_pix),
                W_l.reshape(nc, chunk, n_pix),
                A_l.reshape(nc, chunk, K),
            )
            if skip_G:
                init = None
            else:
                init = (
                    jnp.zeros((n_pix, K, K), cdtype),
                    jnp.zeros((n_pix, K), cdtype),
                )
            carry, A_new = jax.lax.scan(body, init, blocks)
            A_new = A_new.reshape(n_local, K)
            if skip_G:
                return A_new
            MG, bG = carry
            return jax.lax.psum(MG, ROW_AXIS), jax.lax.psum(bG, ROW_AXIS), A_new

        in_specs = (P(ROW_AXIS, None), P(ROW_AXIS, None), P(ROW_AXIS, None), P())
        out_specs = P(ROW_AXIS, None) if skip_G else (P(), P(), P(ROW_AXIS, None))
        return _shmap(local, self.mesh, in_specs, out_specs)

    def _make_pass_two(self, plan: RowPlan):
        """Chunked, summed loss at the post-rotation ``(A, G)``."""
        chunk, cdtype = plan.chunk, self.compute_dtype
        likelihood = self.method.likelihood

        def local(Y_l, W_l, A_l, G):
            n_local, n_pix = Y_l.shape
            nc = n_local // chunk

            def body(total, xs):
                y, w, a = xs
                return total + likelihood.loss(y.astype(cdtype), w.astype(cdtype), a, G), None

            total, _ = jax.lax.scan(
                body,
                jnp.zeros((), cdtype),
                (
                    Y_l.reshape(nc, chunk, n_pix),
                    W_l.reshape(nc, chunk, n_pix),
                    A_l.reshape(nc, chunk, G.shape[1]),
                ),
            )
            return jax.lax.psum(total, ROW_AXIS)

        in_specs = (P(ROW_AXIS, None), P(ROW_AXIS, None), P(ROW_AXIS, None), P())
        return _shmap(local, self.mesh, in_specs, P())

    def _make_step(self, plan: RowPlan, skip_G: bool):
        pass_one = self._make_pass_one(plan, skip_G)
        pass_two = self._make_pass_two(plan)
        ridge, rotation = self._ridge, self.method.rotation

        def step(Y, W, A, G, rotate):
            if skip_G:
                state = RHMFState(A=pass_one(Y, W, A, G), G=G, it=0)
            else:
                MG, bG, A_new = pass_one(Y, W, A, G)
                state = RHMFState(A=A_new, G=solve_normal_equations(MG, bG, ridge), it=0)
                if rotate:
                    # Rotation is driven by G (replicated, (M, K)); the only
                    # sharded work is A @ R_inv, which is row-local.
                    state = rotation(state)
            loss = pass_two(Y, W, state.A, state.G)
            return state.A, state.G, loss

        return jax.jit(step, static_argnums=(4,))

    # -- the loop ------------------------------------------------------------ #

    def run(
        self,
        Y,
        W=None,
        init_state: RHMFState = None,
        rotation_cadence: int = 1,
        conv_check_cadence: int = 10,
        max_iter: int = 1000,
        skip_G: bool = False,
    ) -> tuple[RHMFState, Array]:
        """Run the ALS loop over sharded, chunked rows.

        Mirrors :meth:`robusta_hmf.frame.OptFrame.run`. ``Y`` may be a
        :class:`ShardedData` from :meth:`prepare`, in which case ``W`` is
        ignored. The returned ``state.A`` has exactly ``n_rows`` rows: the
        padding used on device is stripped.
        """
        self._check_method()
        data, _ = self.prepare(Y, W)
        plan = data.plan

        if self.method.opt_method == "als":
            rotation_cadence = 1

        A = self._pad_A(init_state.A, plan)
        G = jnp.asarray(init_state.G, self.compute_dtype)
        step = self._make_step(plan, skip_G)

        loss_history = []
        it0 = int(init_state.it)
        prev_state = RHMFState(A=A[: plan.n_rows], G=G, it=it0)
        prev_loss = jnp.inf

        for i in range(max_iter):
            rot = i % rotation_cadence == 0 and i != 0
            A, G, loss = step(data.Y, data.W, A, G, rot)
            loss_history.append(loss)
            if i % conv_check_cadence == 0 and i != 0:
                # Compare on the unpadded A: the zero padding rows would
                # otherwise dilute mean(A**2) and make max_frac_A stricter
                # than the same test on the dense path.
                state = RHMFState(A=A[: plan.n_rows], G=G, it=it0 + i + 1)
                if self.conv_tester.is_converged(prev_state, state, prev_loss, loss):
                    print(f"Converged at iteration {i}")
                    break
                prev_state = state
                prev_loss = loss
                print(f"iter {state.it:03d} | loss {loss:.4f}", flush=True)
                if not np.isfinite(float(loss)):
                    raise FloatingPointError(f"Loss is not finite at iteration {i}: {loss}")

        state = RHMFState(A=A[: plan.n_rows], G=G, it=it0 + len(loss_history))
        return state, jnp.array(loss_history)

    # -- held-out inference and scoring -------------------------------------- #

    def initial_coefficients(self, Y, W, G: Array, ridge: float | None) -> Array:
        """One weighted A step against a fixed basis, chunked over the rows.

        Uses the data weights alone, matching the dense path's initial guess.
        Returns the padded ``A`` so :meth:`run` can consume it without a
        host round trip.
        """
        data, _ = self.prepare(Y, W)
        chunk, cdtype = data.plan.chunk, self.compute_dtype

        def local(Y_l, W_l, G):
            n_local, n_pix = Y_l.shape
            nc = n_local // chunk

            def body(_, xs):
                y, w = xs
                y = y.astype(cdtype)
                w = w.astype(cdtype)
                return None, solve_normal_equations(*normal_equations(w, w * y, G), ridge)

            _, A = jax.lax.scan(
                body, None, (Y_l.reshape(nc, chunk, n_pix), W_l.reshape(nc, chunk, n_pix))
            )
            return A.reshape(n_local, G.shape[1])

        in_specs = (P(ROW_AXIS, None), P(ROW_AXIS, None), P())
        f = _shmap(local, self.mesh, in_specs, P(ROW_AXIS, None))
        return jax.jit(f)(data.Y, data.W, jnp.asarray(G, cdtype))

    def _make_moments(self, plan: RowPlan):
        """Chunked sums of z, z^2 and chi^2 over the held-out residuals."""
        chunk, cdtype = plan.chunk, self.compute_dtype
        likelihood = self.method.likelihood

        def local(Y_l, W_l, A_l, G):
            n_local, n_pix = Y_l.shape
            nc = n_local // chunk

            def body(carry, xs):
                s1, s2, chi2 = carry
                y, w, a = xs
                y = y.astype(cdtype)
                w = w.astype(cdtype)
                r = y - a @ G.T
                z = r * jnp.sqrt(w * likelihood.weights_irls(y, w, a, G))
                return (s1 + jnp.sum(z), s2 + jnp.sum(z * z), chi2 + jnp.sum(w * r * r)), None

            zero = jnp.zeros((), cdtype)
            (s1, s2, chi2), _ = jax.lax.scan(
                body,
                (zero, zero, zero),
                (
                    Y_l.reshape(nc, chunk, n_pix),
                    W_l.reshape(nc, chunk, n_pix),
                    A_l.reshape(nc, chunk, G.shape[1]),
                ),
            )
            return (
                jax.lax.psum(s1, ROW_AXIS),
                jax.lax.psum(s2, ROW_AXIS),
                jax.lax.psum(chi2, ROW_AXIS),
            )

        in_specs = (P(ROW_AXIS, None), P(ROW_AXIS, None), P(ROW_AXIS, None), P())
        return jax.jit(_shmap(local, self.mesh, in_specs, (P(), P(), P())))

    def z_moments(self, Y, W, state: RHMFState) -> ZMoments:
        """Held-out moments, accumulated chunk by chunk.

        Identical in definition to :func:`robusta_hmf.metrics.z_moments`, but
        never materialises an ``(N, M)`` array. Padding rows contribute nothing
        to the sums and are excluded from the count. Statistics run over every
        pixel, including those masked to zero weight.
        """
        data, _ = self.prepare(Y, W)
        A = self._pad_A(state.A, data.plan)
        s1, s2, chi2 = self._make_moments(data.plan)(
            data.Y, data.W, A, jnp.asarray(state.G, self.compute_dtype)
        )
        return ZMoments(
            n=data.n_rows * data.n_pixels,
            sum_z=float(s1),
            sum_z2=float(s2),
            sum_chi2=float(chi2),
        )

    def robust_weights_blocks(self, Y, W, state: RHMFState, row_block: int | None = None):
        """Yield IRLS weights one row-block at a time, as numpy arrays.

        The full weight matrix is the same size as ``Y``, so this is a
        generator rather than a single array: consume it to compute per-row
        summaries (outlier scores) without ever holding an ``(N, M)`` array in
        the compute dtype.
        """
        row_block = row_block or self.row_block
        cdtype = self.compute_dtype
        likelihood = self.method.likelihood
        G = jnp.asarray(state.G, cdtype)

        @jax.jit
        def block(y, w, a):
            return likelihood.weights_irls(y.astype(cdtype), w.astype(cdtype), a, G)

        for start in range(0, Y.shape[0], row_block):
            stop = min(start + row_block, Y.shape[0])
            yield np.asarray(block(Y[start:stop], W[start:stop], state.A[start:stop]))


# ---------------------------------------------------------------------------- #
# Initialisation
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GramInitialiser(Initialiser):
    """SVD initialisation via the Gram matrix, for data too large to ``svd``.

    Identical in exact arithmetic to
    :func:`~robusta_hmf.initialisation.svd_init`: for ``Y = U S V.T``,
    ``Y.T @ Y = V S^2 V.T``, so the top-K eigenvectors of the Gram matrix are
    ``V[:, :K]`` and ``Y @ V = U S``. This returns ``A = U sqrt(S)`` and
    ``G = V sqrt(S)``, matching the library's convention, without ever forming
    the ``(N, min(N, M))`` ``U`` that ``jnp.linalg.svd`` wants -- which for the
    Gaia RVS sample is as large as ``Y`` itself.

    Equivalent does not mean identical in floating point: forming ``Y.T @ Y``
    squares the condition number, so the recovered subspace agrees with
    ``svd_init`` at around ``sqrt(eps)`` (~1e-8 in float64) rather than
    ``eps``. This is an initialiser, and the first few ALS iterations wash the
    difference out -- fits started from either route agree to ~1e-12 -- but do
    not expect bitwise agreement at iteration zero.

    Parameters
    ----------
    N, M, K : int | None
        Filled in by :meth:`Robusta.fit` when left unset.
    strategy : {"svd", "random", "custom"}, default="svd"
        ``"svd"`` takes the Gram route; the others behave as usual but place
        their output on the mesh.
    frame : ShardedFrame | None
        The frame supplying the mesh and row plan. Filled in automatically
        when ``Robusta`` builds the initialiser.
    """

    frame: Any = None

    def _require_frame(self) -> ShardedFrame:
        if self.frame is None:
            raise ValueError(
                "GramInitialiser needs the ShardedFrame that owns the mesh. "
                "Pass override_frame=ShardedFrame(...) to Robusta and let it "
                "build the initialiser, or construct GramInitialiser(frame=...)."
            )
        return self.frame

    def execute(self, seed=None, A=None, G=None, Y=None, opt=None) -> RHMFState:
        """Seed ``(A, G)`` from data already resident on the mesh.

        Parameters
        ----------
        seed : int | None
            Used by ``strategy="random"``.
        A, G : Array | None
            Used by ``strategy="custom"``.
        Y : ShardedData
            Prepared by :meth:`ShardedFrame.prepare`.
        opt : optax.GradientTransformation | None
            Must be None; the sharded loop has no SGD path.

        Returns
        -------
        state : RHMFState
            ``state.A`` covers the padded row grid.
        """
        frame = self._require_frame()
        if opt is not None:
            raise NotImplementedError("ShardedFrame does not support method='sgd'.")
        if not isinstance(Y, ShardedData):
            raise TypeError(
                f"GramInitialiser needs a ShardedData, got {type(Y).__name__}. "
                "It is produced by ShardedFrame.prepare()."
            )

        if self.strategy == "svd":
            return self._init_svd(frame, Y)
        if self.strategy == "random":
            return self._init_random(frame, Y, seed)
        if self.strategy == "custom":
            if A is None or G is None:
                raise ValueError("Must provide A and G for custom initialisation.")
            return RHMFState(
                A=frame._pad_A(A, Y.plan),
                G=jnp.asarray(G, frame.compute_dtype),
                it=0,
            )
        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def _init_svd(self, frame: ShardedFrame, data: ShardedData) -> RHMFState:
        K, cdtype = self.K, frame.compute_dtype
        if K > min(data.n_rows, data.n_pixels):
            raise ValueError(
                f"K={K} exceeds min(N, M)={min(data.n_rows, data.n_pixels)} in SVD initialisation."
            )
        C = np.asarray(_make_gram(frame, data.plan)(data.Y))
        # eigh on the host: (M, M) is 41 MiB at M=2321, this runs once, and
        # LAPACK avoids any dependence on cuSOLVER's float64 workspace
        # behaviour.
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1][:K]
        V = np.ascontiguousarray(evecs[:, order]).astype(cdtype)
        sv = np.sqrt(np.clip(evals[order], 0.0, None)).astype(cdtype)  # singular values
        root = np.sqrt(sv)
        scale = np.where(root > 0, 1.0 / np.where(root > 0, root, 1.0), 0.0).astype(cdtype)
        A = _make_project(frame, data.plan)(data.Y, jnp.asarray(V), jnp.asarray(scale))
        return RHMFState(A=A, G=jnp.asarray(V * root), it=0)

    def _init_random(self, frame: ShardedFrame, data: ShardedData, seed: int | None) -> RHMFState:
        if seed is None:
            raise ValueError("Must provide seed for random initialisation.")
        key = jax.random.key(seed)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (data.plan.n_padded, self.K), dtype=frame.compute_dtype)
        A = jax.device_put(A, NamedSharding(frame.mesh, P(ROW_AXIS, None)))
        G = jax.random.normal(k2, (data.n_pixels, self.K), dtype=frame.compute_dtype)
        return RHMFState(A=A, G=G, it=0)


def _make_gram(frame: ShardedFrame, plan: RowPlan):
    """``C = Y.T @ Y``, accumulated chunk by chunk and summed over devices."""
    chunk, cdtype = plan.chunk, frame.compute_dtype

    def local(Y_l):
        n_pix = Y_l.shape[1]

        def body(C, y):
            y = y.astype(cdtype)
            return C + y.T @ y, None

        C, _ = jax.lax.scan(body, jnp.zeros((n_pix, n_pix), cdtype), Y_l.reshape(-1, chunk, n_pix))
        return jax.lax.psum(C, ROW_AXIS)

    return jax.jit(_shmap(local, frame.mesh, (P(ROW_AXIS, None),), P()))


def _make_project(frame: ShardedFrame, plan: RowPlan):
    """``A = (Y @ V) * scale``, chunked so Y is never cast to fp64 wholesale."""
    chunk, cdtype = plan.chunk, frame.compute_dtype

    def local(Y_l, V, scale):
        n_pix = Y_l.shape[1]

        def body(_, y):
            return None, (y.astype(cdtype) @ V) * scale

        _, A = jax.lax.scan(body, None, Y_l.reshape(-1, chunk, n_pix))
        return A.reshape(Y_l.shape[0], -1)

    in_specs = (P(ROW_AXIS, None), P(), P())
    return jax.jit(_shmap(local, frame.mesh, in_specs, P(ROW_AXIS, None)))
