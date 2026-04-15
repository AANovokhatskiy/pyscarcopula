# Bivariate Copulas

## The SCAR model

In the SCAR (Stochastic Copula Autoregressive) model, the copula parameter follows a latent Ornstein-Uhlenbeck process:

$$\theta_t = \Psi(x_t), \qquad dx_t = \theta_\text{OU}(\mu - x_t)\,dt + \nu\,dW_t$$

The three OU parameters control:

- `theta_OU` - mean-reversion speed
- `mu` - long-run mean of the latent process
- `nu` - volatility of the latent process

## Fitting

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

copula = GumbelCopula(rotate=180)
result = fit(copula, u, method='scar-tm-ou')

print(result.params.theta, result.params.mu, result.params.nu)
print(result.log_likelihood)
```

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

**`sample`** generates synthetic data reproducing the fitted model. For SCAR, it simulates an OU trajectory with the fitted parameters and samples from the copula with the time-varying parameter. Useful for model validation: `fit(copula, sample(...))` should recover similar parameters.

```python
from pyscarcopula.api import sample, predict

v = sample(copula, u, result, n=2000)
result_refit = fit(copula, pobs(v), method='scar-tm-ou')
```

**`predict`** generates samples for next-step forecasting. It also supports conditional generation via `given={idx: u_value}`. For SCAR-TM, `horizon='current'` uses `p(x_T | data)`, while `horizon='next'` uses the one-step-ahead predictive distribution `p(x_{T+1} | data)`.

```python
u_pred = predict(copula, u, result, n=100_000)

# Conditional forecast: sample U2 | U1 = 0.4
u_cond = predict(copula, u, result, n=20_000, given={0: 0.4})

# SCAR-TM: choose current-step or one-step-ahead latent mixture
u_current = predict(copula, u, result, n=20_000, horizon='current')
```

| Method | `sample` | `predict` |
|--------|----------|-----------|
| MLE | constant r | constant r |
| SCAR-TM | OU trajectory | current/posterior or one-step-ahead mixture |
| GAS | recursive score-driven simulation | last filtered value `f_T` |

## Diagnostics

### Smoothed parameter

```python
from pyscarcopula.api import smoothed_params

r_t = smoothed_params(copula, u, result)
```

Returns the filtered copula parameter at each time step.

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=result, to_pobs=False)
```

Uses the Rosenblatt transform with the Cramer-von Mises statistic. For SCAR models, it integrates the h-function over the predictive distribution (mixture Rosenblatt).
