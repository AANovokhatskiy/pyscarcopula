# Bivariate Copulas

## The SCAR model

In the SCAR (Stochastic Copula Autoregressive) model, the copula parameter follows a latent Ornstein-Uhlenbeck process:

$$r_t = \Psi(x_t), \qquad dx_t = \kappa(\mu - x_t)\,dt + \nu\,dW_t$$

The three OU parameters control:

- $\kappa$ - mean-reversion speed
- $\mu$ - long-run mean of the latent process
- $\nu$ - volatility of the latent process

## Fitting

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

copula = GumbelCopula(rotate=180)
result = fit(copula, u, method='scar-tm-ou')

print(result.params.kappa, result.params.mu, result.params.nu)
print(result.log_likelihood)
```

By default, SCAR-TM uses `transition_method='auto'`. This selects between the
grid matrix method, local Gauss-Hermite, and the Hermite spectral likelihood
depending on the OU transition scale. Use `transition_method='spectral'` to
force the spectral likelihood, or `transition_method='matrix'` / `'gh'` for
grid-only comparisons. See [Performance](performance.md#spectral-hermite-likelihood)
for the numerical details.

## Rotations

Rotations capture different tail dependence patterns:

| Rotation | Tail dependence |
|----------|-----------------|
| 0 deg | Upper tail (Gumbel, Joe) or lower tail (Clayton) |
| 90 deg | Mixed |
| 180 deg | Opposite tail |
| 270 deg | Mixed |

For financial data (joint crashes), `GumbelCopula(rotate=180)` or `ClaytonCopula(rotate=0)` are common choices.

## Sampling and prediction

Two functions serve different purposes:

**`sample`** generates synthetic data from the fitted model. For SCAR, it
simulates an OU trajectory with the fitted parameters and samples from the
copula with the time-varying parameter. This is useful for model validation:
`fit(copula, sample(...))` should recover similar parameters.

```python
import numpy as np
from pyscarcopula.api import sample, predict

v = sample(copula, u, result, n=2000, rng=np.random.default_rng(2024))
result_refit = fit(copula, pobs(v), method='scar-tm-ou')
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

```python
u_pred = predict(copula, u, result, n=100_000,
                 rng=np.random.default_rng(2025))

# Conditional forecast: sample U2 | U1 = 0.4
u_cond = predict(copula, u, result, n=20_000, given={0: 0.4},
                 rng=np.random.default_rng(2026))

# SCAR-TM: choose current-step or one-step-ahead latent mixture
u_current = predict(copula, u, result, n=20_000, horizon='current',
                    rng=np.random.default_rng(2027))
```

| Method | `sample` | `predict` |
|--------|----------|-----------|
| MLE | constant r | constant r |
| SCAR-TM | OU trajectory | current/posterior or one-step-ahead mixture |
| GAS | recursive score-driven simulation | last filtered value $g_T$ |

## Diagnostics

### Predictive mean parameter

```python
from pyscarcopula.api import predictive_mean

r_t = predictive_mean(copula, u, result)
```

Returns the predictive mean copula parameter at each time step, before the
current observation is absorbed. `smoothed_params` is still available as a
backward-compatible alias.

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=result, to_pobs=False)
```

The GoF test uses the Rosenblatt transform with the Cramer-von Mises
statistic. For SCAR models, it integrates the h-function over the predictive
distribution (mixture Rosenblatt).
