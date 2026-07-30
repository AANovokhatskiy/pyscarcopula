# Bivariate Copulas

## The SCAR model

In the SCAR (Stochastic Copula Autoregressive) model, the copula parameter follows a latent Ornstein-Uhlenbeck process:

$$r_t = \Psi(x_t), \qquad dx_t = \kappa(\mu - x_t)\,dt + \nu\,dW_t$$

The three OU parameters control:

- $\kappa$ - mean-reversion speed
- $\mu$ - long-run mean of the latent process
- $\nu$ - volatility of the latent process

For the likelihood recursion, gradient identity, transition backends, and
predictive Rosenblatt transform used by SCAR and GAS, see
[Mathematical Contracts](mathematical-contracts.md).

## Fitting

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

copula = GumbelCopula(rotate=180)
scar_result = fit(copula, u, method='scar-tm-ou')

print(
    scar_result.params.kappa,
    scar_result.params.mu,
    scar_result.params.nu,
)
print(scar_result.log_likelihood)
```

For Kendall-tau dynamics, use `method='scar-tm-jacobi'`:

```python
result_jacobi = fit(copula, u, method='scar-tm-jacobi')
print(result_jacobi.params.kappa, result_jacobi.params.m, result_jacobi.params.xi)
```

SCAR-TM-JACOBI is available for copulas with a Kendall-tau parameter mapping
such as Gumbel, Clayton, Frank, Joe, and bivariate Gaussian. It models tau
directly with a bounded Jacobi diffusion and maps tau back to the copula
parameter.

The default `transition_method='auto'` selects among the supported numerical
backends and records the selected path in fit diagnostics. See
[Estimation Methods](estimation-methods.md) for model semantics and
[Numerical Backends](numerical-backends.md) for backend selection, fallback
conditions, and configuration.

## Rotations

Rotations capture different tail dependence patterns:

| Rotation | Tail dependence |
|----------|-----------------|
| 0 deg | Upper tail (Gumbel, Joe) or lower tail (Clayton) |
| 90 deg | Mixed |
| 180 deg | Opposite tail |
| 270 deg | Mixed |

For lower-tail dependence, compare `GumbelCopula(rotate=180)` and
`ClaytonCopula(rotate=0)` with an MLE baseline and goodness-of-fit test.

## Sampling and prediction

Two functions serve different purposes:

**`sample`** generates synthetic data from the fitted model. SCAR-TM-OU
simulates an OU trajectory. SCAR-TM-JACOBI simulates Kendall's tau with the
likelihood-consistent `tm_grid` sampler by default, or with the experimental
opt-in `lamperti_euler` sampler, and maps tau to the time-varying copula
parameter. This is useful for model validation:
`fit(copula, sample(...))` should recover similar parameters.

```python
import numpy as np
from pyscarcopula.api import sample, predict

v = sample(
    copula,
    u,
    scar_result,
    n=2000,
    rng=np.random.default_rng(2024),
)
result_refit = fit(copula, v, method='scar-tm-ou')
```

**`predict`** generates samples for next-step forecasting. It also supports
conditional generation via `given={idx: u_value}`. For SCAR-TM,
`horizon='current'` uses $p(x_T \mid data)$, while `horizon='next'` uses the
one-step-ahead predictive distribution $p(x_{T+1} \mid data)$.

For the shared prediction terminology used by bivariate and vine models, see
[Prediction Semantics](prediction-semantics.md).

## The GAS model

The GAS model is observation-driven. The copula parameter is

$$r_t = \Psi(g_t),$$

where the unbounded recursion state follows

$$g_{t+1} = \omega + \beta g_t + \gamma s_t.$$

Here $\omega$ is the intercept, $\gamma$ controls sensitivity to the scaled
score, $\beta$ controls persistence, and $s_t$ is the scaled score of the
current copula log-density with respect to $g_t$.

GAS uses the compiled numerical evaluator:

```python
from pyscarcopula.api import fit

gas_result = fit(copula, u, method='gas', scaling='unit')
```

The example uses `scaling='unit'`. See
[Estimation Methods](estimation-methods.md#gas) for the scaling contract and
[Numerical Backends](numerical-backends.md#gas) for optimizer controls.

```python
u_pred = predict(copula, u, gas_result, n=100_000,
                  rng=np.random.default_rng(2025))

# Conditional forecast: sample U2 | U1 = 0.4
u_cond = predict(copula, u, gas_result, n=20_000, given={0: 0.4},
                  rng=np.random.default_rng(2026))

# SCAR-TM: choose current-step or one-step-ahead latent mixture
u_current = predict(copula, u, scar_result, n=20_000, horizon='current',
                     rng=np.random.default_rng(2027))
```

| Method | `sample` | `predict` |
|--------|----------|-----------|
| MLE | constant r | constant r |
| SCAR-TM-OU | OU trajectory | current/posterior or one-step-ahead mixture |
| SCAR-TM-JACOBI | `tm_grid` trajectory by default; experimental Lamperti--Euler opt-in | current/posterior or one-step-ahead mixture |
| GAS | recursive score-driven simulation | `current`: last filtered state; `next`: one score-recursion step |

## Diagnostics

### Predictive mean parameter

```python
from pyscarcopula.api import predictive_mean

r_t = predictive_mean(copula, u, scar_result)
```

Returns the predictive mean copula parameter at each time step, before the
current observation is absorbed.

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=scar_result, to_pobs=False)
```

The GoF test uses the Rosenblatt transform with the Cramer-von Mises
statistic. GAS evaluates the transform at the filtered point state. SCAR
models integrate the h-function over the predictive latent-state distribution
before the current observation is absorbed, which is the mixture Rosenblatt
contract described in [Mathematical Contracts](mathematical-contracts.md).

Use parametric bootstrap calibration when an asymptotic p-value is
insufficient:

```python
gof = gof_test(
    copula,
    u,
    fit_result=scar_result,
    to_pobs=False,
    bootstrap=True,
    n_bootstrap=499,
    n_jobs=-1,
    rng=20260730,
)
```

Bootstrap replicas run in independent worker processes. Each replica owns its
model and random stream, so a fixed seed produces the same bootstrap
statistics for `n_jobs=1` and `n_jobs>1`. Native computations use one thread
per worker to avoid nested process/thread oversubscription. Bootstrap
calibration is also available for static Gaussian and Student copulas and for
dynamic equicorrelation Gaussian and stochastic Student copulas. Vine
bootstrap remains out of scope.
