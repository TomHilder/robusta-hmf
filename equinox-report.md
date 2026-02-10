# Equinox Dependency Report

## What is Equinox?

[Equinox](https://docs.kidger.site/equinox/) is a lightweight JAX library that brings PyTorch-like OOP convenience to JAX without sacrificing performance or purity. It provides:

- **PyTree-based models**: Classes that act as JAX PyTrees, enabling composition and transformation
- **Filtered transformations**: JAX transformations (JIT, grad) that intelligently handle non-array types
- **PyTree manipulation**: Tools for surgical modifications to module structure

The core philosophy: **models are PyTrees**. Unlike frameworks that require explicit parameter extraction, Equinox lets you define models as classes where both data arrays and configuration are stored as module fields.

## Key APIs Used in robusta-hmf

### eqx.Module

Base class that registers a user-defined class as a JAX PyTree. All annotated fields become PyTree leaves.

```python
class RHMFState(eqx.Module):
    A: Array = eqx.field(converter=jax.numpy.asarray)
    G: Array = eqx.field(converter=jax.numpy.asarray)
    it: int = eqx.field(default=0)
    opt_state: optax.OptState | None = eqx.field(default=None)
```

### eqx.field()

Field descriptor with metadata:

- `converter`: Transform field values (e.g., `converter=jax.numpy.asarray` ensures JAX arrays)
- `static=True`: Field is a compile-time constant, not part of PyTree (never traced by JIT/grad)
- `default`: Default value

Used heavily in robusta-hmf for static configuration:

```python
target: Literal["A", "G", "none"] = eqx.field(static=True, default="G")
ridge: float | None = eqx.field(static=True, default=None)
```

### @eqx.filter_jit

Wrapper around `jax.jit` that automatically handles non-array types. All JAX/NumPy arrays are traced (dynamic), everything else is held static. Eliminates the need for `static_argnums`.

```python
@eqx.filter_jit
def step_als(self, Y, W_data, state, rotate=True, skip_G=False):
    # self.likelihood, self.rotation (static) -> never retraced
    # Y, W_data, state.A, state.G (arrays) -> traced normally
    ...
```

### eqx.filter_value_and_grad

Equinox's `jax.value_and_grad` that automatically filters non-array fields from gradient computation.

```python
loss, grads = eqx.filter_value_and_grad(loss_fn)(params, Y)
```

### eqx.tree_at

Surgical, immutable PyTree updates — modify specific fields while preserving the rest.

```python
# Single field update
state = eqx.tree_at(lambda s: s.it, state, state.it + 1)

# Multi-field update
return eqx.tree_at(
    lambda s: (s.A, s.G, s.it, s.opt_state),
    state,
    (A_, G_, it_, opt_),
)
```

### eqx.partition / eqx.combine / eqx.filter

Not currently used in robusta-hmf but important Equinox concepts:

- `eqx.partition`: Split PyTree into two by predicate (e.g., arrays vs non-arrays)
- `eqx.combine`: Merge split PyTrees back together
- `eqx.filter`: Keep/remove leaves based on predicate

## How robusta-hmf Uses Equinox

Six key modules inherit from `eqx.Module`:

| Module | File | Purpose | Static Fields |
|--------|------|---------|---------------|
| `RHMFState` | state.py | Optimization state (A, G, iteration, opt_state) | — |
| `HMF` | hmf.py | Main optimization engine | likelihood, rotation, opt_method, opt |
| `Likelihood` hierarchy | likelihoods.py | Loss functions (Gaussian, Student-t, Cauchy) | nu, scale |
| `Rotation` hierarchy | rotations.py | Symmetry handling (Identity, FastAffine, SlowAffine) | target, whiten, eps |
| `WeightedAStep` / `WeightedGStep` | als.py | ALS linear solvers | ridge |
| `Regulariser` hierarchy | regularisers.py | Placeholder for regularization | — |

### Design Patterns

1. **Static config fields**: Likelihood type, rotation method, ridge parameter — marked `static=True` so JIT doesn't retrace when only data changes
2. **Immutable state updates**: `eqx.tree_at` for functional updates to `RHMFState`
3. **Filtered JIT on methods**: `@eqx.filter_jit` on `step_als()` and `step_sgd()` — self (static config) + data (dynamic arrays) handled automatically
4. **Abstract base classes**: `Likelihood`, `Rotation`, `Regulariser` as abstract `eqx.Module` subclasses for polymorphism
5. **Integration with optax**: Optimizer state lives inside `RHMFState`, gradients from `eqx.filter_value_and_grad` flow to `opt.update()`

## References

- [Equinox Documentation](https://docs.kidger.site/equinox/)
- [Equinox GitHub](https://github.com/patrick-kidger/equinox)
- [Equinox Paper (arXiv:2111.00254)](https://arxiv.org/abs/2111.00254)
