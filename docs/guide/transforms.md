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
# Default (backward compatible)
cop = GumbelCopula(rotate=180)

# Softplus
cop = GumbelCopula(rotate=180, transform_type='softplus')
```

### softplus advantages

The softplus transform has a natural "floor" — the copula parameter cannot go below a minimum value (e.g., θ=1 for Gumbel, which corresponds to independence). This matches financial data where correlations have a lower bound but can spike arbitrarily high.

On BTC-ETH daily data (Gumbel-180, SCAR-TM-OU):

| Transform | logL | OU parameters (θ, μ, ν) | GoF p-value |
|-----------|------|-------------------------|-------------|
| xtanh | 1042.47 | (49.98, 2.43, 10.66) | 0.643 |
| **softplus** | **1045.78** | **(59.03, 2.17, 15.80)** | **0.698** |

### xtanh advantages

Symmetric — treats upward and downward movements equally. May be preferable when the copula parameter can meaningfully decrease below the long-run mean.

## Using with vine

The `transform_type` parameter propagates to all edge copulas in a vine:

```python
vine = CVineCopula()
vine.fit(u, method='scar-tm-ou', transform_type='softplus')
```

## Technical note

The analytical gradient of the transfer matrix log-likelihood uses the chain rule:

$$\frac{\partial f_i}{\partial x} = f_i \cdot \frac{\partial \log c}{\partial \theta} \cdot \Psi'(x)$$

For softplus, Ψ'(x) = sigmoid(x) = 1/(1+e⁻ˣ). This is automatically handled when `transform_type='softplus'` is set.
