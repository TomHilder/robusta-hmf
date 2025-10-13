# Robusta-HMF

`jax` implementation of robust heteroskedastic matrix factorisation. Robusta like the coffee bean, get it?

## TODOs

- [x] Port Hogg's existing code and make sure it builds/installs*
- [x] Port to `equinox`*
- [ ] Type checking with `mypy`*
- [ ] Add dependency injection for the following:*
  - [x] Optimisation method, IRLS, SGD (directly optimising objective, see robust_hmf_notes.pdf)
    - [ ] Potentially `dask` and batching support for SGD
  - [x] w-steps. Each w-step corresponds to a different likelihood. Hogg's is Cauchy. We should let this flexible*
  - [x] Initialisation.
  - [x] Re-orientation. Can easily imagine wanting something cheaper for really big data.
- [ ] Add a save and restore method. Probably avoid pickle/dill and instead encapsulate info in serialisable way and then rebuild model upon loading*
- [ ] Tests!*
- [ ] CI, automated tests, automated relases, and PyPI*
- [ ] Relax version requirements since uv by default is newest everything

Upon reaching some critical "done-ness", this package should be moved to it's own repo while keeping the git history. Probably just before whenever we want automated tests, releases, PyPI, etc.

(*) = Priority
