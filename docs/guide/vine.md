# Vine Copulas

## Overview

Vine copulas decompose a d-dimensional copula into d(d-1)/2 bivariate copulas arranged in a tree structure. Two vine types are supported:

- **C-vine**: fixed star structure (one "root" variable per tree)
- **R-vine**: data-driven structure selected via Dissmann's MST algorithm

Each edge copula is selected automatically from all available families via AIC, and can use constant (MLE) or time-varying (SCAR, GAS) parameters.

## C-vine

A C-vine uses a star structure where variable 1 is the root of tree 0, variable 2 is the root of tree 1, etc.

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

An R-vine selects the tree structure from data using Dissmann's algorithm: at each tree level, a maximum spanning tree is built on |Kendall's tau|, subject to the proximity condition.

```python
from pyscarcopula import RVineCopula

vine = RVineCopula()
vine.fit(u, method='scar-tm-ou',
         truncation_level=2,
         min_edge_logL=10,
         to_pobs=True)
vine.summary()
```

The R-vine typically achieves higher log-likelihood than C-vine because it can capture the strongest pairwise dependencies at each tree level, rather than being constrained to a star structure.

Some Dissmann-selected regular-vine tree sets are not encodable by the current
`RVineMatrix` sampler. In that case `RVineCopula.fit` emits a
`RuntimeWarning` and refits a matrix-encodable C-vine fallback instead of
failing. This keeps `sample`, `predict`, and rolling risk workflows usable,
but the fitted fallback structure can have a lower log-likelihood than the
original selected tree set.

You can also provide a custom structure:

```python
from pyscarcopula.vine._structure import RVineMatrix

M = np.array([[0, 0, 0, 0],
              [1, 1, 0, 0],
              [2, 2, 2, 0],
              [3, 3, 3, 3]])
vine = RVineCopula(structure=RVineMatrix(M))
vine.fit(u, method='mle')
```

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

The `gof_test` function auto-dispatches to the correct Rosenblatt transform for both C-vine and R-vine models, handling mixed SCAR/MLE edges correctly.

## Sampling and prediction

```python
# Predict: next-step conditional sampling (for VaR/CVaR)
predictions = vine.predict(n=10000)

# Sample: reproduce fitted model (for parameter recovery)
samples = vine.sample(n=10000)
```

Conditional generation is also supported via `given={var_index: u_value}`:

```python
pred_cond = vine.predict(n=5000, given={2: 0.6})
pred_cond2 = vine.predict(n=5000, given={0: 0.2, 3: 0.8})
```

For arbitrary R-vine conditioning, `RVineCopula.predict` provides these
sampling modes:

- `conditional_method='auto'` uses the fastest available valid path.
- `conditional_method='graph'` uses the computational-graph sampler when no
  posterior grid is needed, or when the flexible graph executor can handle the
  single-given pattern.
- `conditional_method='grid'` uses a joint posterior grid for non-prefix
  conditioning patterns.
- `conditional_method='exact'` uses recursive quadrature for small reference
  runs.

Use `conditional_plan` to inspect the workload before sampling:

```python
given = {0: 0.2, 3: 0.8}
plan = vine.conditional_plan(given, quad_order=4)

print(plan['posterior_dim'])
print(plan['joint_grid_points'])
print(plan['graph_feasible'])

samples = vine.predict(
    n=5000,
    u=u,
    given=given,
    horizon='next',
    quad_order=4,
    conditional_method='auto',
)
```

When the future conditioning variables are known before fitting, fit an
R-vine structure optimized for that pattern:

```python
vine_cond = RVineCopula(
    structure_mode='conditional',
    conditional_vars={0, 3},
    conditional_structure_policy='min_posterior',
)
vine_cond.fit(u, method='scar-tm-ou', truncation_level=2)

plan = vine_cond.conditional_plan({0: 0.2, 3: 0.8}, quad_order=4)
samples = vine_cond.predict(
    n=5000,
    u=u,
    given={0: 0.2, 3: 0.8},
    horizon='next',
    conditional_method='graph' if plan['graph_feasible'] else 'grid',
    quad_order=4,
)
```

The conditional structure policy prioritizes lower sampling cost. With
`conditional_structure_policy='min_posterior'`, the fitter can fall back to a
conditional C-vine if the priority R-vine is not matrix-encodable or would
need a larger posterior grid.

Current limitations:

- `grid` is approximate; accuracy depends on `quad_order`.
- Grid cost grows as `quad_order ** posterior_dim` and is capped internally.
- The flexible `graph` executor currently supports graph-sampleable
  single-given patterns. Multi-given patterns with posterior variables should
  use `auto`, `grid`, or `exact`.
- Conditional structure selection optimizes the known conditioning pattern,
  not a full likelihood/sampling-cost objective over all possible structures.

For SCAR-TM edges, `predict(..., horizon='current')` uses `p(x_T | data)` and `predict(..., horizon='next')` uses `p(x_{T+1} | data)`. `sample` still simulates independent OU trajectories.

## Results on 6-crypto data (T=250)

| Model | logL | GoF p-value |
|-------|------|-------------|
| **C-vine SCAR-TM** | **921.9** | **0.89** |
| R-vine SCAR-TM | 919.1 | 0.62 |
| R-vine MLE | 873.0 | 0.19 |
| C-vine MLE | 869.2 | 0.21 |
| Student-t | 764.4 | 0.0001 |
| Gaussian | 761.0 | 0.0000 |
