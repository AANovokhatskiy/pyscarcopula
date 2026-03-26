# Transform Functions

## Overview

The transform function Ψ(x) maps the latent OU process x(t) to the copula parameter domain. For example, Gumbel requires θ ≥ 1, so Ψ: R → [1, ∞).

pyscarcopula provides two transforms:

| Name | Formula | Properties |
|------|---------|------------|
| `xtanh` | x·tanh(x) + offset | Symmetric, linear growth at ±∞ |
| `softplus` | log(1+eˣ) + offset | Asymmetric, floor at offset |

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

The softplus transform has a natural floor — the copula parameter cannot go below a minimum value (e.g., θ=1 for Gumbel, which corresponds to independence). This matches financial data where correlations have a lower bound but can spike arbitrarily high.

### xtanh advantages

Symmetric — treats upward and downward movements equally. May be preferable when the copula parameter can meaningfully decrease below the long-run mean.

## Using with vine

The `transform_type` parameter propagates to all edge copulas in a vine:

```python
from pyscarcopula import CVineCopula

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou', transform_type='softplus')
```
