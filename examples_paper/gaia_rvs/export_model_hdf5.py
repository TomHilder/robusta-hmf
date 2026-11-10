"""Pack the final full-RVS fit into one self-contained HDF5 file.

``fit_final_full_rvs.py`` saves only what the figures need: the factors (A, G)
and per-spectrum weight summaries. Anything that wants the actual spectra, the
model spectra or the per-pixel robust weights has to rebuild them from the
HDF5, the state file and the metadata CSV, which is exactly the three-way
alignment that is easy to get wrong. This script does it once and writes:

    data/flux            (N, P) float32  observed flux, masked pixels 0
    data/ivar            (N, P) float32  inverse variance, masked pixels 0
    data/wavelength      (P,)   float64  vacuum wavelength (nm), edges clipped
    model/model_flux     (N, P) float32  A @ G.T
    model/robust_weights (N, P) float32  IRLS weights, masked pixels 1
    meta/source_id       (N,)   int64    Gaia DR3 source_id
    meta/gaia_id         (N,)   int64    alias of source_id, for convenience
    meta/<column>        (N,)            every column of the metadata CSV

ROW ORDER is the catalogue order of the sample -- the order of the spectra in
``gaia-dr3-rvs-all.hdf5``, not the seeded shuffle the fit ran in. The A rows are
permuted back accordingly (see ``inverse_row_order``), so row i of every dataset
here is the same star in ``data/``, ``model/`` and ``meta/``.

MASKING follows ``20260731_oom_tests.load_split``, which is what the fit used:
a pixel with non-finite flux or non-finite 1/sigma^2 gets flux = ivar = 0. With
ivar = 0 the IRLS weight is 1 by construction (the residual carries no weight),
which is the same convention as ``analysis_funcs.compute_outlier_scores``.

The robust weights are recomputed here rather than read from disk -- the fit
never stores the (N, P) weight matrix, only per-spectrum summaries of it. They
come from the same ``StudentTLikelihood`` the model was fit with, evaluated in
float64, so ``--check`` can compare the 1st-percentile-per-spectrum score
against ``full_rvs_final_weights.npz`` and should agree to round-off.

SIZE: four (N, P) float32 arrays, ~9.3 GiB each, so ~37 GiB uncompressed for
the full sample. Nothing (N, P) is ever resident: the file is filled a
row-block at a time.

USAGE
    uv run python export_model_hdf5.py                    # K, Q from the weights npz
    uv run python export_model_hdf5.py --K 16 --Q 7.5     # or state them
    uv run python export_model_hdf5.py --limit 5000 --out /tmp/test.hdf5 --check
"""

import argparse
import time
from pathlib import Path

import gaia_config as cfg
import h5py
import numpy as np
import polars as pl
from analysis_funcs import clip_edge_pix, get_test_train_split_idx
from collect import META
from train_full_ms import RESULTS_DIR, build_sample

from robusta_hmf.likelihoods import StudentTLikelihood

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Student-t degrees of freedom. Not a CLI option in the fitting scripts either:
# every model in gaia_rvs_results was fit at the DistributedRobusta default.
ROBUST_NU = 1.0

# Rows per read/compute/write block. 4096 x 2321 in float64 is ~76 MiB per
# intermediate, and there are a handful of them alive at once.
BLOCK = 4096

# HDF5 chunk shape for the (N, P) datasets: whole spectra, so reading one star
# touches one chunk.
CHUNK_ROWS = 64

# ============================================================================ #


def resolve_KQ(args, weights_file):
    """(K, Q) from the CLI if given, else from the fit's weights file."""
    if args.K is not None and args.Q is not None:
        return args.K, args.Q
    if args.K is not None or args.Q is not None:
        raise SystemExit("Pass both --K and --Q, or neither (to read the weights file)")
    if not weights_file.exists():
        raise SystemExit(f"{weights_file} does not exist -- pass --K and --Q explicitly.")
    d = np.load(weights_file)
    K, Q = int(d["best_K"]), float(d["best_Q"])
    print(f"Read K={K}, Q={Q:g} from {weights_file.name}")
    return K, Q


def inverse_row_order(n_rows, source_id, state_source_id):
    """Map catalogue row -> row of A in the state file.

    The fit runs on the seeded shuffle from ``get_test_train_split_idx`` with
    ``train_frac=1.0``, so ``A[i]`` belongs to catalogue row ``row_idx[i]``.
    Inverting that puts everything back in catalogue order. The state file
    carries its own source ids, so the mapping is checked rather than trusted.
    """
    row_idx, _ = get_test_train_split_idx(n_rows, train_frac=1.0)
    if not np.array_equal(state_source_id, source_id[row_idx]):
        raise SystemExit(
            "The state file's source ids do not match the recomputed split -- the state "
            "was fit on a different sample than build_sample('all') returns."
        )
    inv = np.empty(n_rows, dtype=np.int64)
    inv[row_idx] = np.arange(n_rows)
    return inv


def read_metadata(source_id):
    """Every column of the metadata CSV, in the given source_id order.

    A left join, so a spectrum with no CSV row still gets a slot (null, which
    becomes NaN / False / "" below). With both filters off the join is 1:1 in
    practice; the null counts are recorded per dataset either way.
    """
    print(f"Reading {META}...", flush=True)
    t0 = time.time()
    meta = pl.read_csv(META)
    order = pl.DataFrame({"source_id": source_id}).with_row_index("_row")
    meta = order.join(meta, on="source_id", how="left").sort("_row").drop("_row")
    n_missing = int(meta["designation"].null_count()) if "designation" in meta.columns else 0
    print(
        f"  {meta.height} rows x {meta.width} columns in {time.time() - t0:.0f} s"
        + (f", {n_missing} with no CSV match" if n_missing else "")
    )
    return meta


def column_for_h5(series):
    """(values, dtype, note) for one metadata column, ready for h5py.

    Nulls survive as NaN wherever the column can hold one, which means integer
    columns with nulls are widened to float64 rather than getting a sentinel.
    Booleans and strings have no NaN, so nulls become False / "" and the count
    is returned for the dataset attributes.
    """
    n_null = int(series.null_count())
    dt = series.dtype
    if dt == pl.String:
        values = series.fill_null("").to_numpy().astype(object)
        return values, h5py.string_dtype(), n_null
    if dt == pl.Boolean:
        return series.fill_null(False).to_numpy(), None, n_null
    if dt.is_integer() and n_null:
        return series.cast(pl.Float64).to_numpy(), None, 0
    return series.to_numpy(), None, 0


def write_metadata(group, meta, source_id):
    """meta/ -- source_id, its gaia_id alias, and every CSV column."""
    group.create_dataset("source_id", data=source_id)
    group["source_id"].attrs["description"] = "Gaia DR3 source_id, aligned with data/ and model/"
    group.create_dataset("gaia_id", data=source_id)
    group["gaia_id"].attrs["description"] = "identical to meta/source_id; kept as an alias"

    for name in meta.columns:
        if name in ("source_id", "gaia_id"):
            continue  # already written, from the spectra file rather than the CSV
        values, dtype, n_null = column_for_h5(meta[name])
        ds = group.create_dataset(name, data=values, dtype=dtype)
        ds.attrs["source"] = META.name
        if n_null:
            ds.attrs["n_null"] = n_null
            ds.attrs["null_fill"] = "" if dtype is not None else False
    print(f"  wrote {len(group)} metadata datasets")


def fill_spectra(f, data, idx, A, G, Q, block, chunk_rows, compression=None):
    """Fill data/ and model/ a row-block at a time, in catalogue order.

    Blocks are contiguous in catalogue order, which is HDF5 order, so
    ``get_flux_batch`` reads whole slabs and decompresses each chunk once.
    """
    n, K = A.shape
    P = G.shape[0]
    q2 = np.float64(Q) ** 2
    nu = np.float64(ROBUST_NU)
    likelihood = StudentTLikelihood(nu=float(ROBUST_NU), scale=float(Q))
    assert K == G.shape[1]

    kw = dict(
        shape=(n, P), dtype=np.float32, chunks=(min(chunk_rows, n), P), compression=compression
    )
    out = {
        "flux": f["data"].create_dataset("flux", **kw),
        "ivar": f["data"].create_dataset("ivar", **kw),
        "model_flux": f["model"].create_dataset("model_flux", **kw),
        "robust_weights": f["model"].create_dataset("robust_weights", **kw),
    }

    n_masked, t0 = 0, time.time()
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = np.arange(start, stop)

        flux, u_flux = clip_edge_pix(*data.get_flux_batch(idx[rows]))
        flux = flux.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = 1.0 / (u_flux.astype(np.float64) ** 2)
        # isfinite, not isnan: u_flux == 0 gives an infinite weight, which a
        # nan_to_num route would turn into 1.8e308 rather than masking.
        bad = ~(np.isfinite(flux) & np.isfinite(ivar))
        n_masked += int(bad.sum())
        Y = np.where(bad, 0.0, flux)
        W = np.where(bad, 0.0, ivar)

        model = A[rows] @ G.T
        # StudentTLikelihood.weights_irls, written out so this stays on the
        # host in float64: nu*s^2 / (nu*s^2 + ivar*residual^2). Masked pixels
        # have ivar = 0 and therefore weight 1.
        rw = (nu * q2) / (nu * q2 + W * (Y - model) ** 2)

        out["flux"][start:stop] = Y.astype(np.float32)
        out["ivar"][start:stop] = W.astype(np.float32)
        out["model_flux"][start:stop] = model.astype(np.float32)
        out["robust_weights"][start:stop] = rw.astype(np.float32)

        elapsed = time.time() - t0
        rate = stop / elapsed
        print(
            f"  {stop}/{n} rows, {elapsed / 60:.1f} min elapsed, "
            f"{(n - stop) / rate / 60:.1f} min left   ",
            end="\r",
            flush=True,
        )

    print(
        f"  {n} rows x {P} pixels in {(time.time() - t0) / 60:.1f} min, "
        f"{n_masked / (n * P):.3%} of pixels masked" + " " * 20,
        flush=True,
    )
    _check_likelihood_agrees(likelihood, Y, W, A[rows], G, rw)


def _check_likelihood_agrees(likelihood, Y, W, A_rows, G, rw):
    """The last block, through the library's own IRLS weights.

    Cheap insurance that the inlined expression above is the same function the
    model was fit with, evaluated on real data rather than a toy array.
    """
    import jax

    # Nothing above this point touches jax, so flipping x64 on here is safe and
    # keeps the comparison in the precision the fit ran at.
    jax.config.update("jax_enable_x64", True)
    ref = np.asarray(likelihood.weights_irls(Y, W, A_rows, G))
    max_diff = np.max(np.abs(ref - rw))
    print(f"  IRLS weights agree with the library to {max_diff:.2e}")


def check_against_weights_file(out_file, weights_file, n_sample=2000, seed=0):
    """Compare the 1st-percentile-per-spectrum score with the fit's own npz."""
    if not weights_file.exists():
        print(f"Skipping check: {weights_file} does not exist")
        return
    d = np.load(weights_file)
    with h5py.File(out_file, "r") as f:
        source_id = f["meta/source_id"][:]
        if len(source_id) != len(d["source_id"]):
            print("Skipping check: the weights file covers a different number of spectra")
            return
        # The npz is in the fit's shuffled order; row_idx maps it back.
        want = np.empty(len(source_id))
        want[d["row_idx"]] = d["score"]
        rows = np.sort(np.random.default_rng(seed).choice(len(source_id), n_sample, replace=False))
        got = np.array([np.percentile(f["model/robust_weights"][int(i)], 1) for i in rows])
    max_diff = np.max(np.abs(got - want[rows]))
    print(
        f"Check vs {weights_file.name}: max |score difference| over {n_sample} random "
        f"spectra = {max_diff:.2e}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sample", default="all", choices=("ms", "all", "all_filtered"))
    p.add_argument("--K", type=int, default=None, help="rank; default: read the weights file")
    p.add_argument("--Q", type=float, default=None, help="robust scale; default: as --K")
    p.add_argument("--state", type=Path, default=None, help="state npz (default: the allrows fit)")
    p.add_argument("--out", type=Path, default=None, help="output HDF5 file")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--block", type=int, default=BLOCK, help="rows per read/compute/write block")
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS, help="rows per HDF5 chunk")
    p.add_argument(
        "--compression",
        default="none",
        choices=("none", "lzf", "gzip"),
        help="gzip is ~10x slower to write for a few percent on float32 spectra",
    )
    p.add_argument("--limit", type=int, default=0, help="first N spectra only (for testing)")
    p.add_argument(
        "--check", action="store_true", help="compare scores against the fit's weights npz"
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    tag = {"ms": "full_ms", "all": "full_rvs", "all_filtered": "full_rvs"}[args.sample]
    weights_file = args.results_dir / f"{tag}_final_weights.npz"
    K, Q = resolve_KQ(args, weights_file)

    state_file = args.state or (
        args.results_dir / f"converged_state_R{K}_Q{Q:.2f}_bin_{tag}_allrows.npz"
    )
    if not state_file.exists():
        raise SystemExit(f"{state_file} does not exist")
    out_file = args.out or (args.results_dir / f"{tag}_K{K}_Q{Q:.2f}_model.hdf5")
    if out_file.exists() and not args.overwrite:
        raise SystemExit(f"{out_file} exists -- pass --overwrite to replace it")

    print(f"Building sample {args.sample!r}...", flush=True)
    data, idx, ids, _ = build_sample(args.sample)
    print(f"Sample {tag!r}: {len(idx)} spectra")

    print(f"Loading {state_file}", flush=True)
    state = np.load(state_file)
    A, G = state["A"], state["G"]
    if A.shape[0] != len(idx):
        raise SystemExit(f"{state_file} has {A.shape[0]} rows, but the sample has {len(idx)}")
    if A.shape[1] != K:
        raise SystemExit(f"{state_file} has rank {A.shape[1]}, not K={K}")

    # Catalogue order, so data/, model/ and meta/ line up row for row and the
    # HDF5 reads stay sequential.
    inv = inverse_row_order(len(idx), ids, state["source_id"])
    A = A[inv]

    if args.limit:
        n = min(args.limit, len(idx))
        print(f"Limiting to the first {n} spectra")
        idx, ids, A = idx[:n], ids[:n], A[:n]

    meta = read_metadata(ids)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with h5py.File(out_file, "w") as f:
        f.attrs["description"] = "Gaia DR3 RVS spectra, RHMF model and per-pixel robust weights"
        f.attrs["sample"] = args.sample
        f.attrs["tag"] = tag
        f.attrs["K"] = K
        f.attrs["Q"] = Q
        f.attrs["robust_nu"] = ROBUST_NU
        f.attrs["state_file"] = str(state_file)
        f.attrs["n_clip_pix"] = cfg.N_CLIP_PIX
        f.attrs["row_order"] = "catalogue (spectra file) order, not the fit's shuffled order"
        f.attrs["masking"] = "non-finite flux or ivar -> flux = ivar = 0, robust weight = 1"

        f.create_group("data")
        f.create_group("model")
        f.create_group("meta")

        # Same grid MatchedData exposes, with the same edge clipping as the flux.
        lam = data.λ_grid[cfg.N_CLIP_PIX : -cfg.N_CLIP_PIX]
        ds = f["data"].create_dataset("wavelength", data=lam)
        ds.attrs["units"] = "nm"

        print("Writing spectra, model and robust weights...", flush=True)
        fill_spectra(
            f,
            data,
            idx,
            A,
            G,
            Q,
            args.block,
            args.chunk_rows,
            compression=None if args.compression == "none" else args.compression,
        )

        print("Writing metadata...", flush=True)
        write_metadata(f["meta"], meta, ids)

    size = out_file.stat().st_size
    print(f"\nWrote {out_file} ({size / 2**30:.1f} GiB) in {(time.time() - t0) / 60:.1f} min")

    if args.check:
        check_against_weights_file(out_file, weights_file)


if __name__ == "__main__":
    main()
