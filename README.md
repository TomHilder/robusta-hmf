# Robusta-HMF

`jax` implementation of robust heteroskedastic matrix factorisation. Robusta like the coffee bean, get it?

## Installation

Easiest is from PyPI either with `pip`

```sh
pip install robusta-hmf
```

or `uv` (recommended)

```sh
uv add robusta-hmf
```

Or, you can clone and build from source

```sh
git clone git@github.com:TomHilder/robusta-hmf.git
cd robusta-hmf
pip install -e .
```

## Usage

Given a data matrix `Y` of shape `(N, M)` and a matching matrix of inverse-variance
weights `W`, fit a rank-`K` factorisation `Y ≈ A @ G.T`:

```python
from robusta_hmf import Robusta

# rank-5 robust fit; robust_scale controls how aggressively outliers are downweighted
model = Robusta(rank=5, robust=True, robust_scale=2.0)
state, loss_history = model.fit(Y, W, max_iter=1000)

basis = state.G          # (M, K) shared basis vectors
coeffs = state.A         # (N, K) per-row coefficients
reconstruction = model.synthesize()   # A @ G.T, shape (N, M)
```

Missing or masked data is handled by setting the corresponding entries of `W` to `0`.
See `examples_paper/` for complete, worked pipelines (synthetic validation in `toy/` and
Gaia RVS spectra in `gaia_rvs/`).

### Model selection

`score()` returns the held-out statistics used in the paper. The headline number is the KL
divergence of the standardised residuals from `N(0, 1)` (Eq. 20) — zero is perfect, lower is
better — which is what to minimise over a `(K, Q)` grid:

```python
infer_state, _ = model.infer(Y_test, W_test, state=state)
scores = model.score(Y_test, W_test, state=infer_state)

scores.kl        # KL divergence from N(0, 1); the model-selection criterion
scores.std_z     # sigma_z, target 1
scores.chi2_red  # reduced chi-squared, target 1
```

### Fitting data larger than one GPU

Each ALS step over the whole `(N, M)` matrix materialises about five full-size
intermediates on top of `Y` and `W`. For 496,955 × 2321 spectra in float64 that is ~43 GiB
of scratch, which will not fit in a 48 GiB card.

**Chunking the row axis is what fixes this**, and it needs no second device — peak scratch
drops to ~0.4 GiB, set by `row_block` rather than by `N`. Sharding across devices is a
speed knob layered on top. Both come from swapping in one frame:

```python
from robusta_hmf import Robusta, ShardedFrame

model = Robusta(
    rank=10,
    robust_scale=5.0,
    override_frame=ShardedFrame(row_block=4096),  # devices default to all visible
)
state, loss_history = model.fit(Y, W)   # plain numpy arrays; never fully copied to host
```

Everything else — `infer`, `synthesize`, `score`, `coefficients`, `residuals` — works
unchanged, and the results match the ordinary path to round-off. `Y` and `W` are stored on
device in float32 by default while all arithmetic stays in float64, halving resident memory.

To fit several models to one sample (say a `(K, Q)` grid) without re-transferring the data,
place it once and reuse the handle:

```python
frame = ShardedFrame(row_block=4096)
data, _ = frame.prepare(Y, W)          # one transfer to device
for rank in ranks:
    model = Robusta(rank=rank, override_frame=frame)   # bind() copies, so reuse is safe
    state, _ = model.fit(data)
```

## Citation

If you use `robusta-hmf` in your research, please cite the accompanying paper
([arXiv:2607.08081](https://arxiv.org/abs/2607.08081)):

```bibtex
@article{hilder2026robusta,
    title         = {Robust Heteroskedastic Matrix Factorization: A Generalization
                     of PCA that Flags Outliers and Handles Missing Data},
    author        = {Hilder, Thomas and Hogg, David W. and Casey, Andrew R.
                     and Rix, Hans-Walter},
    year          = {2026},
    eprint        = {2607.08081},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.IM},
    url           = {https://arxiv.org/abs/2607.08081},
}
```

## Help

Found a bug, or have a question or feature request? Please
[open an issue](https://github.com/TomHilder/robusta-hmf/issues) on GitHub.

## TODOs

- [ ] More interpretable loss, maybe normalised in some sensible way (maybe proper NLL)
- [ ] Type checking with `mypy`
