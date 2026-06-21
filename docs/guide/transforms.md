# Transform Functions

## Overview

The transform function $\Psi(x)$ maps the latent OU process $x(t)$ to the
copula parameter domain. For example, Gumbel requires $\theta \ge 1$, so
$\Psi: \mathbb{R} \to [1, \infty)$.

For Archimedean copulas, pyscarcopula provides two selectable transforms. The
default is `softplus`.

| Name | Formula | Properties |
|------|---------|------------|
| `softplus` | $\log(1 + \exp(x)) + \texttt{offset}$ | Default; asymmetric, floor at `offset` |
| `xtanh` | $x \tanh(x) + \texttt{offset}$ | Symmetric, linear growth at large $|x|$ |

Gaussian copulas are different: their correlation parameter always uses the
bounded Gaussian tanh mapping. Although `BivariateGaussianCopula` accepts
`transform_type='softplus'` or `'xtanh'`, that argument exists only so common
copula and vine configuration can be passed to every candidate constructor.
It does not change Gaussian mathematics and emits no warning.

## Inverse-transform semantics

For `softplus` and the fixed Gaussian tanh transform, `inv_transform` is a numerical
inverse of the forward transform.

`xtanh` is deliberately different. The forward function
$x\tanh(x)+\texttt{offset}$ is even, so positive and negative latent values
produce the same copula parameter and no globally unique inverse exists.
For initialization, pyscarcopula uses the established positive-branch
modulus approximation

$$
\operatorname{inv\_transform}(r) = |r| + \texttt{offset}.
$$

This is an initialization convention rather than a mathematical inverse.
Consequently, `transform(inv_transform(r)) == r` is not guaranteed for
`xtanh`.

## Choosing a transform

This choice applies to Archimedean families such as Gumbel, Clayton, Frank,
and Joe. It does not select the transform for Gaussian copulas.

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

# Default: softplus
copula = GumbelCopula(rotate=180)

# Explicit softplus
copula = GumbelCopula(rotate=180, transform_type='softplus')
result = fit(copula, u, method='scar-tm-ou')
```

### softplus advantages

The softplus transform has a natural floor: the copula parameter cannot go
below a minimum value. For Gumbel, $\theta = 1$ corresponds to independence.
This is useful for financial data where correlations may have a lower bound
but can spike sharply.

### xtanh advantages

This transform is symmetric and treats upward and downward movements equally.
It may be preferable when the copula parameter can meaningfully decrease below
the long-run mean. Its `inv_transform` follows the approximation described
above and should not be used where an exact latent round trip is required.

## Using with vine

The `transform_type` parameter propagates through the common constructor flow
to all edge copulas in a vine:

```python
from pyscarcopula import CVineCopula

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou', transform_type='softplus')
```

Archimedean edges use the selected transform. Gaussian edges accept and retain
the value for configuration consistency but always use `GaussianTanh`.
