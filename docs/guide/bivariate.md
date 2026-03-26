# Bivariate Copulas

## The SCAR model

In the SCAR (Stochastic Copula Autoregressive) model, the copula parameter follows a latent Ornstein-Uhlenbeck process:

$$\theta_t = \Psi(x_t), \qquad dx_t = \theta_\text{OU}(\mu - x_t)\,dt + \nu\,dW_t$$

The three OU parameters control:

- θ_OU — mean-reversion speed
- μ — long-run mean of the latent process
- ν — volatility of the latent process

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
| 0° | Upper tail (Gumbel, Joe) or lower tail (Clayton) |
| 90° | Mixed |
| 180° | Opposite tail |
| 270° | Mixed |

For financial data (joint crashes), `GumbelCopula(rotate=180)` or `ClaytonCopula(rotate=0)` are common choices.

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

Uses the Rosenblatt transform with Cramér-von Mises statistic. For SCAR models, integrates the h-function over the predictive distribution (mixture Rosenblatt).
