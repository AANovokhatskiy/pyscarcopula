# Vine Copulas

## Overview

Vine copulas decompose a $d$-dimensional copula into $d(d-1)/2$ bivariate
copulas arranged in a tree structure. Two vine types are supported:

- **C-vine**: fixed star structure (one "root" variable per tree)
- **R-vine**: data-driven structure selected via Dissmann's MST algorithm

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
from pyscarcopula import CVineCopula

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         to_pobs=True)
vine.summary()
```

## R-vine

An R-vine selects the tree structure from data using Dissmann's algorithm: at
each tree level, a maximum spanning tree is built on $|\text{Kendall's tau}|$,
subject to the proximity condition.

```python
from pyscarcopula import RVineCopula

vine = RVineCopula()
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         to_pobs=True)
vine.summary()
```

The R-vine typically achieves higher log-likelihood than C-vine because it can
capture the strongest pairwise dependencies at each tree level, rather than
being constrained to a star structure.

If the conditioning set is known in advance, you can bias structure selection
toward an R-vine that supports the fast exact conditional sampler for that
set, with the fixed variables placed at the end of the R-vine variable order:

```python
vine = RVineCopula()
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
variable_order = [
    int(vine.matrix[vine.d - 1 - col, col])
    for col in range(vine.d)
]

# Fast exact path: fix the last variables in the R-vine order.
pred_cond = vine.predict(
    n=5000,
    u=u,
    given={variable_order[-1]: 0.6},
    horizon='current',
    rng=np.random.default_rng(2026),
)
pred_cond2 = vine.predict(
    n=5000,
    u=u,
    given={variable_order[-2]: 0.35, variable_order[-1]: 0.75},
    rng=np.random.default_rng(2027),
)
```

For `RVineCopula`, the fast exact conditional sampler requires the fixed
variables to be at the end of the fitted R-vine variable order. The fixed
variables must already be last in the fitted structure, or the fitted structure
must be rebuildable into an equivalent order where they are last.

To inspect the fitted structure as an `RVineMatrix`, use
`vine.to_rvine_matrix()` or `RVineMatrix.from_model(vine)`.

If exact conditioning is not possible, `predict` uses the approximate fallback.
This is more general, but more expensive than exact suffix sampling.

You can inspect the fitted variable order before choosing `given`:

```python
print(variable_order)

samples, diagnostics = vine.predict(
    n=5000,
    u=u,
    given={variable_order[0]: 0.45},
    mcmc_steps=300,
    mcmc_burnin=100,
    return_diagnostics=True,
    rng=np.random.default_rng(2028),
)
print(diagnostics["conditional_method"])  # "suffix" or "dag_mcmc"
if diagnostics["conditional_method"] == "dag_mcmc":
    print(diagnostics["mcmc"]["acceptance_mean"])
```

Use a fresh `np.random.default_rng(seed)` for each call when exact
reproducibility is required. Reusing the same generator object advances its
random stream.

Supported behavior and limits:

- R-vine `fit(..., conditional_mode=...)` accepts
  `conditional_mode='suffix'`.
- R-vine `predict` does not accept `conditional_method` or `quad_order`.
- Arbitrary R-vine conditioning uses an approximate fallback, not an exact
  closed-form posterior sampler.
- C-vine conditional sampling is separate and supports its own prefix/general
  paths.

For a focused description of prediction semantics, see
[Prediction Semantics](prediction-semantics.md). For R-vine-specific details,
see [R-vine Conditioning](rvine-conditioning.md).

For SCAR-TM edges, `predict(..., horizon='current')` uses the posterior latent
state after the fitted history and `predict(..., horizon='next')` uses the
one-step-ahead latent state. For SCAR-TM-OU edges, `sample` simulates
independent OU trajectories.

For SCAR-TM predictive parameter sampling, `predictive_r_mode` may be `None`,
`"grid"`, or `"histogram"`.
