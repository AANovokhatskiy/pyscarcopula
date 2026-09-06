# Bivariate Copulas

Bivariate models describe dependence between two uniform margins. Start
with a constant MLE fit, then consider GAS or SCAR when time variation is
part of the modelling question. The code blocks on this page run in order.

## Families and data

| Family | Dependence / tail pattern |
|---|---|
| Gumbel, Joe | Positive dependence, upper-tail dependence before rotation |
| Clayton | Positive dependence, lower-tail dependence before rotation |
| Frank | Symmetric dependence, no asymptotic tail dependence |
| Bivariate Gaussian | Positive or negative correlation, no asymptotic tail dependence |
| Independent | No dependence and no fitted parameter |

```python
import numpy as np
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit, sample, predict, predictive_mean

source = GumbelCopula(rotate=180)
u = source.sample_at_parameter(
    120, np.full(120, 1.8), rng=np.random.default_rng(9),
)
```

Application inputs must have shape `(T, 2)`. Pass existing pseudo-observations
directly, or set `to_pobs=True` for raw continuous observations.

## Rotations

| Rotation | Tail pattern for rotatable Archimedean families |
|---|---|
| 0 degrees | Upper tail for Gumbel/Joe, lower tail for Clayton |
| 90 or 270 degrees | Mixed tails and negative association |
| 180 degrees | Opposite tail from the unrotated family |

For lower-tail dependence, compare `GumbelCopula(rotate=180)` and
`ClaytonCopula()` using MLE and goodness of fit. Frank and bivariate Gaussian
do not expose these rotations. See [Parameter Transforms](transforms.md) for
the parameter links used by dynamic fits.

## Fitting

```python
copula = GumbelCopula(rotate=180)
mle_result = fit(copula, u, method="mle")
print(mle_result.copula_param, mle_result.log_likelihood)
if not mle_result.success:
    raise RuntimeError(mle_result.message)
```

`fit` returns a result and stores it on the model. A later fit replaces the
model's fitted state; explicit results passed to the functional API select
the desired fit. See [Configuration and Results](../api/configuration.md#fit-results).

## Sampling and prediction

```python
v = sample(copula, u, mle_result, n=500, rng=np.random.default_rng(2024))
u_pred = predict(copula, u, mle_result, n=500, rng=np.random.default_rng(2025))
u_cond = predict(
    copula, u, mle_result, n=500, given={0: 0.4},
    rng=np.random.default_rng(2026),
)
```

`sample` reproduces a fitted model. `predict` conditions on the observed
history; for MLE both use the same constant parameter. `given` fixes either
column or both columns. The meanings diverge for dynamic models:

| Method | `sample` | `predict` |
|---|---|---|
| MLE | Constant parameter | Constant parameter |
| GAS | Recursive score-driven trajectory | Last filtered or next score state |
| SCAR-TM-OU | New OU trajectory | Posterior or one-step predictive mixture |
| SCAR-TM-JACOBI | `tm_grid` trajectory; experimental `lamperti_euler` opt-in | Posterior or one-step predictive mixture |

## The GAS model

GAS is observation-driven. Its parameter is $r_t=\Psi(g_t)$, with recursion

$$
g_{t+1}=\omega+\beta g_t+\gamma s_t.
$$

Here $s_t$ is the scaled log-density score with respect to $g_t$.

```python
gas_result = fit(copula, u, method="gas", scaling="unit")
print(gas_result.success, gas_result.params)
```

GAS returns `GASResult`. See [Estimation Methods](estimation-methods.md#gas)
for the score contract and [Optimizer Controls](../reference/optimizers.md#gas)
for numerical options.

## The SCAR model

SCAR-TM-OU maps a latent Ornstein-Uhlenbeck process to the copula parameter:

$$
r_t=\Psi(x_t),\qquad dx_t=\kappa(\mu-x_t)\,dt+\nu\,dW_t.
$$

The parameters are mean-reversion speed $\kappa$, long-run latent mean $\mu$,
and latent diffusion coefficient $\nu$. SCAR-TM-JACOBI instead evolves
family-scale Kendall tau with a bounded diffusion.

```python
scar_result = fit(copula, u, method="scar-tm-ou")
result_jacobi = fit(copula, u, method="scar-tm-jacobi")
print(scar_result.success, scar_result.params)
print(result_jacobi.success, result_jacobi.params)
```

Both SCAR methods return `LatentResult`. Inspect optimizer success and
backend diagnostics before interpreting either result. Supported combinations
are listed in [Estimation Methods](estimation-methods.md#model-and-method-compatibility).

`horizon="current"` uses the posterior after the observed sample, while
`horizon="next"` uses its one-step-ahead transition:

```python
u_current = predict(
    copula, u, scar_result, n=500, horizon="current",
    rng=np.random.default_rng(2027),
)
```

See [Prediction Semantics](prediction-semantics.md) for horizons and
[Mathematical Contracts](mathematical-contracts.md) for likelihood formulas.

## Diagnostics

### Predictive mean parameter

```python
r_t = predictive_mean(copula, u, scar_result)
assert r_t.shape == (len(u),)
```

This is the predictive mean copula parameter before each current observation
is absorbed, rather than the posterior smoothed mean.

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=mle_result, to_pobs=False)
```

GAS uses its filtered point state for the Rosenblatt transform. SCAR uses
a predictive mixture of h-functions. Parametric bootstrap supports bivariate,
static and dynamic multivariate, and fitted vine models. See
[Diagnostics](../api/diagnostics.md) for calibration, random streams,
process parallelism, and unsuccessful-refit handling.
