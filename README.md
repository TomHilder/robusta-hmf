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
