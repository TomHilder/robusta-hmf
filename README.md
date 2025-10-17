# Robusta-HMF

`jax` implementation of robust heteroskedastic matrix factorisation. Robusta like the coffee bean, get it?

## Installation

Easiest is from PyPI either with `pip` (CURRENTLY NOT ON PYPI YET)

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

TODO

## Citation

TODO

## Help

TODO

## TODOs

- [x] Port Hogg's existing code and make sure it builds/installs*
- [x] Port to `equinox`*
- [ ] Type checking with `mypy`*
- [x] Add dependency injection for the following:*
  - [x] Optimisation method, IRLS, SGD (directly optimising objective, see robust_hmf_notes.pdf)
    - [ ] Potentially `dask` and batching support for SGD
  - [x] w-steps. Each w-step corresponds to a different likelihood. Hogg's is Cauchy. We should let this flexible*
  - [x] Initialisation.
  - [x] Re-orientation. Can easily imagine wanting something cheaper for really big data.
- [ ] Add a save and restore method. Probably avoid pickle/dill and instead encapsulate info in serialisable way and then rebuild model upon loading
  - [ ] Eh maybe, maybe not
- [x] Tests!*
- [ ] CI, automated tests, automated relases, and PyPI*
- [ ] Relax version requirements since uv by default is newest everything

(*) = Priority
