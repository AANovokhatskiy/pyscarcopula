# Vine API

This page is the API reference entry point for vine classes and helper types.
Usage examples and conceptual details live in the user guide:

- [Vine Copulas](../guide/vine.md)
- [Prediction Semantics](../guide/prediction-semantics.md)
- [R-vine Conditioning](../guide/rvine-conditioning.md)
- [Performance Tuning](../guide/performance.md)

## Public Options

`CVineCopula.predict(...)` and `VineCopula.predict(...)` both support:

- `given={var_index: u_value}` for conditional generation in
  pseudo-observation space;
- `horizon='current'|'next'` for dynamic edge prediction;
- `predictive_r_mode='grid'|'histogram'|None` for SCAR-TM predictive
  parameter sampling. No other string values are supported;
- `rng=np.random.default_rng(seed)` for reproducible Monte Carlo output.

`VineCopula.predict(...)` additionally supports:

- `predict_config=PredictConfig(...)`;
- `dynamic_conditioning='ignore'|'given_only'`;
- `mcmc_steps=<non-negative int>` and `mcmc_burnin=<non-negative int>` for
  approximate conditional prediction;
- `return_diagnostics=True`.

`VineCopula.fit(...)` additionally supports fit-time conditional-structure
targeting:

- `given_vars=[...]`;
- `conditional_strict=True|False`;
- `conditional_mode='suffix'`;
- `structure_search='beam'|'multi-start'`;
- `beam_width=<positive int>`.

For detailed behavior of these options, see the guide pages linked above. The
API signatures below are generated from the source docstrings.

`candidates=` and `copulas=` have different meanings for vine fitting.
Pass `candidates=[BivariateGaussianCopula, ...]` to define the family pool
used for automatic selection. Pass `copulas=[[(CopulaClass, rotation), ...],
...]` only when the family and rotation of every edge are fixed in advance.
`copulas=` does not accept fitted copula instances.

Use `vine.to_rvine_matrix()` or `RVineMatrix.from_model(vine)` when you need
the fitted R-vine structure as an `RVineMatrix`.

For new code, construct all generic modes through one class:

```python
from pyscarcopula import VineCopula
from pyscarcopula.vine import RVineMatrix

auto = VineCopula()
c_vine = VineCopula.cvine(d=5, order=[0, 1, 2, 3, 4])
d_vine = VineCopula.dvine(d=5, order=[0, 1, 2, 3, 4])
arbitrary = VineCopula(
    structure=RVineMatrix.from_trees(d=5, trees=trees)
)
```

`model.structure` is the canonical public structure.
`model.natural_order_matrix` exposes the numerical matrix convention;
`model.matrix` is retained as a compatibility property.

## VineCopula

::: pyscarcopula.vine.vine.VineCopula
    options:
      members:
        - cvine
        - dvine
        - rvine
        - fit
        - log_likelihood
        - sample
        - predict
        - summary
        - to_rvine_matrix

## CVineCopula

`CVineCopula` is the supported legacy implementation. Prefer
`VineCopula.cvine(...)` for new code. It does not currently emit a runtime
deprecation warning; its C-vine-specific conditional algorithms intentionally
differ from the generic matrix-based conditional runtime.

::: pyscarcopula.vine.cvine.CVineCopula
    options:
      members:
        - fit
        - log_likelihood
        - sample
        - predict
        - summary

## RVineCopula compatibility name

`RVineCopula is VineCopula`. The old import remains supported, but it does not
define a second runtime or a distinct R-vine model type.

Its methods and signatures are therefore the same as the canonical
[`VineCopula`](#vinecopula) API above.

## PredictConfig

::: pyscarcopula.PredictConfig

## RVineMatrix

::: pyscarcopula.vine._structure.RVineMatrix
    options:
      members:
        - from_model
        - from_natural_order
        - from_trees
        - edge
        - edges_at_tree
        - n_trees
        - n_edges

## PairCopula

::: pyscarcopula.vine._pair_copula.PairCopula
    options:
      members: false

## select_best_copula

::: pyscarcopula.vine._selection.select_best_copula

::: pyscarcopula.vine._selection.SelectedCopula
