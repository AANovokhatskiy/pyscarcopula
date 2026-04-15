# Transform Functions

## Overview

The transform function `Psi(x)` maps the latent OU process `x(t)` to the copula parameter domain. For example, Gumbel requires `theta >= 1`, so `Psi: R -> [1, inf)`.

pyscarcopula provides two transforms:

| Name | Formula | Properties |
|------|---------|------------|
| `xtanh` | `x * tanh(x) + offset` | Symmetric, linear growth at large `|x|` |
| `softplus` | `log(1 + exp(x)) + offset` | Asymmetric, floor at `offset` |

## Choosing a transform

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

# Default
copula = GumbelCopula(rotate=180)

# Softplus (often better on financial data)
copula = GumbelCopula(rotate=180, transform_type='softplus')
result = fit(copula, u, method='scar-tm-ou')
```

### softplus advantages

The softplus transform has a natural floor - the copula parameter cannot go below a minimum value, e.g. `theta = 1` for Gumbel, which corresponds to independence. This matches financial data where correlations have a lower bound but can spike arbitrarily high.

### xtanh advantages

Symmetric - treats upward and downward movements equally. It may be preferable when the copula parameter can meaningfully decrease below the long-run mean.

## Using with vine

The `transform_type` parameter propagates to all edge copulas in a vine:

```python
from pyscarcopula import CVineCopula

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou', transform_type='softplus')
```
