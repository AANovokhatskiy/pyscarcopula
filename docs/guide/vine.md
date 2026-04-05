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

For SCAR edges, `predict` uses the posterior distribution p(x_T | data) via transfer matrix, while `sample` simulates independent OU trajectories.

## Results on 6-crypto data (T=250)

| Model | logL | GoF p-value |
|-------|------|-------------|
| **C-vine SCAR-TM** | **921.9** | **0.89** |
| R-vine SCAR-TM | 919.1 | 0.62 |
| R-vine MLE | 873.0 | 0.19 |
| C-vine MLE | 869.2 | 0.21 |
| Student-t | 764.4 | 0.0001 |
| Gaussian | 761.0 | 0.0000 |
