# Vine Copulas

## Overview

Vine copulas decompose a $d$-dimensional copula into $d(d-1)/2$ bivariate
copulas arranged in a tree structure. `VineCopula` is the primary runtime and
supports three structural modes:

- **auto R-vine**: data-driven structure selected by Dissmann's algorithm;
- **fixed C-vine**: successive-root star trees;
- **fixed D-vine**: path trees.

Any other valid regular vine can be supplied as an `RVineMatrix`. C/D/R are
structure modes of one model, not separate generic runtime classes.

Each edge copula is selected automatically from the configured candidate
families via AIC, and can use constant (MLE) or time-varying (SCAR, GAS)
parameters.

The shared `transform_type` option is passed to candidate constructors.
Archimedean edge families use it to select `softplus` or `xtanh`.
`BivariateGaussianCopula` accepts the same argument for constructor
uniformity, but Gaussian edges always use their bounded `GaussianTanh`
correlation mapping.

Use `candidates=` to configure the automatic family-selection pool. For
Gaussian pair-copula edges, use `BivariateGaussianCopula`; the multivariate
`GaussianCopula` is not a valid vine edge family. The lower-level `copulas=`
fit argument has a different purpose: it fixes every edge family and rotation
as `(CopulaClass, rotation)` specs in a list-of-lists matching the tree
layout.

## C-vine

A C-vine uses a star structure where the first variable is the root of tree 0,
the second variable is the root of tree 1, and so on.

```python
from pyscarcopula import VineCopula

vine = VineCopula.cvine(d=u.shape[1])
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         to_pobs=True)
vine.summary()
```

Static MLE models fitted to the same C-vine structure and fixed edge families
have equivalent edge semantics and likelihood. Internal edge ordering and
seeded sampling trajectories are not an API parity guarantee.

Conditional sampling uses matrix-based suffix conditioning and a DAG+MCMC
  fallback for arbitrary conditioning sets;
prediction supports `return_diagnostics`, MCMC controls, and
`dynamic_conditioning`.

## Auto-selected R-vine

An R-vine selects the tree structure from data using Dissmann's algorithm: at
each tree level, a maximum spanning tree is built on $|\text{Kendall's tau}|$,
subject to the proximity condition.

```python
from pyscarcopula import VineCopula

vine = VineCopula()
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         to_pobs=True)
vine.summary()
```

An auto-selected R-vine can place the strongest available pairwise
dependencies at each tree level instead of being constrained to a C-vine star
structure. Compare fitted structures with the same likelihood and information
criterion.

If the conditioning set is known in advance, you can bias structure selection
toward an R-vine that supports the fast exact conditional sampler for that
set, with the fixed variables placed at the end of the R-vine variable order:

```python
vine = VineCopula()
vine.fit(
    u,
    method='scar-tm-ou',
    truncation_level=2,
    min_edge_logL=10,
    to_pobs=True,
    given_vars=[0, 2],
)
```

`given_vars` is a fit-time structure-selection target. With the default
`conditional_strict=True`, `fit` raises `ValueError` if no suffix-compatible
exact structure is constructed. With `conditional_strict=False`, prediction can
use the approximate fallback when the exact path is not available.

## Fixed D-vine

Use a D-vine factory when the first-tree path order is part of the model:

```python
from pyscarcopula import VineCopula

vine = VineCopula.dvine(
    d=u.shape[1],
    order=[0, 2, 1, 3],
).fit(u, method="mle")
```

`VineCopula.cvine(...)` and `.dvine(...)` both configure a fixed
`RVineMatrix`; fitting their pair copulas never runs structure selection.

## Arbitrary fixed regular vine

Prefer decoded tree edges over manually writing either matrix convention:

```python
from pyscarcopula import VineCopula
from pyscarcopula.vine import RVineMatrix

structure = RVineMatrix.from_trees(
    d=4,
    trees=[
        [
            (frozenset({0, 1}), frozenset()),
            (frozenset({1, 2}), frozenset()),
            (frozenset({1, 3}), frozenset()),
        ],
        [
            (frozenset({0, 2}), frozenset({1})),
            (frozenset({0, 3}), frozenset({1})),
        ],
        [
            (frozenset({2, 3}), frozenset({0, 1})),
        ],
    ],
)
vine = VineCopula(structure=structure).fit(u, method="mle")
```

`RVineMatrix.from_trees` validates tree sizes, indices, and the proximity
condition before any optimizer is called.

## Candidate families versus fixed families

Constructor `candidates=` defines the pool searched independently for every
edge:

```python
from pyscarcopula import BivariateGaussianCopula, FrankCopula

vine = VineCopula.dvine(
    d=u.shape[1],
    candidates=[BivariateGaussianCopula, FrankCopula],
).fit(u, method="mle")
```

Fit argument `copulas=` instead fixes every `(family, rotation)` in the
structure's decoded tree order:

```python
trees = vine.structure.to_trees()
fixed_specs = [
    [(BivariateGaussianCopula, 0) for _ in level]
    for level in trees
]
fixed = VineCopula(structure=vine.structure).fit(
    u,
    method="mle",
    copulas=fixed_specs,
)
```

`copulas=` contains classes and rotations, not fitted copula instances.

## Inspecting structure

```python
print(vine.structure)               # canonical RVineMatrix
print(vine.structure.to_trees())    # decoded semantic edges
print(vine.natural_order_matrix)    # numerical/integration convention
```

`vine.matrix` returns the same natural-order representation for compatibility
with older code. Prefer `structure` for modelling and `natural_order_matrix`
when an external integration explicitly needs that convention.

`RVineCopula` is a compatibility name for the same `VineCopula` type.

## Truncation

For large d, not all edges benefit from dynamic parameters:

```python
# Trees 0-1: SCAR, trees 2+: MLE
vine.fit(u, method='scar-tm-ou', truncation_level=2)

# Edges with weak MLE dependence stay MLE
vine.fit(u, method='scar-tm-ou', min_edge_logL=10)

# Both
vine.fit(u, method='scar-tm-ou',
         truncation_level=2, min_edge_logL=10)
```

Edges where no parametric copula beats independence by AIC are set to `IndependentCopula` automatically.

## Goodness of fit

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(vine, u, to_pobs=False)
```

The `gof_test` function dispatches to the correct Rosenblatt transform for
both C-vine and R-vine models, including mixed SCAR/MLE edges.

## Sampling and prediction

```python
import numpy as np

# Predict: next-step conditional sampling (for VaR/CVaR)
predictions = vine.predict(
    n=10000,
    u=u,
    horizon='next',
    rng=np.random.default_rng(2025),
)

# Sample: reproduce fitted model (for parameter recovery)
samples = vine.sample(n=10000, rng=np.random.default_rng(2024))
```

Conditional generation is supported via `given={var_index: u_value}` in
pseudo-observation space:

```python
# Fit with given_vars=[0] when this exact conditioning target is known.
pred_cond = vine.predict(
    n=5000,
    u=u,
    given={0: 0.6},
    horizon='current',
    rng=np.random.default_rng(2026),
)
```

Use a fresh `np.random.default_rng(seed)` for each call when exact
reproducibility is required. Reusing the same generator object advances its
random stream.

`VineCopula` automatically selects exact suffix conditioning when the fitted
structure supports it and otherwise uses an approximate DAG+MCMC fallback.
`VineCopula.predict` does not accept `conditional_method`; that name is
reported only in diagnostics. See
[Prediction Semantics](prediction-semantics.md) for the shared
meaning of `predict`, `sample`, and `horizon`, and
[R-vine Conditioning](rvine-conditioning.md) for exact-path requirements,
MCMC controls, and diagnostic fields.

For SCAR-TM edges, `predict(..., horizon='current')` uses the posterior latent
state after the fitted history and `predict(..., horizon='next')` uses the
one-step-ahead latent state. For SCAR-TM-OU edges, `sample` simulates
independent OU trajectories.

For SCAR-TM predictive parameter sampling, `predictive_r_mode` may be `None`,
`"grid"`, or `"histogram"`.
