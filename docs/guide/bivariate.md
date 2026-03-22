# Bivariate Copulas

## The SCAR model

In the SCAR (Stochastic Copula Autoregressive) model, the copula parameter follows a latent Ornstein-Uhlenbeck process:

$$\theta_t = \Psi(x_t), \qquad dx_t = \theta_\text{OU}(\mu - x_t)\,dt + \nu\,dW_t$$

The three OU parameters control:

- $\theta_\text{OU}$ — mean-reversion speed
- $\mu$ — long-run mean of the latent process
- $\nu$ — volatility of the latent process

## Fitting

```python
from pyscarcopula import GumbelCopula

copula = GumbelCopula(rotate=180, transform_type='softplus')
result = copula.fit(u, method='scar-tm-ou')

print(result.alpha)            # (theta_OU, mu, nu)
print(result.log_likelihood)
print(result.nfev)             # function evaluations
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
theta_t = copula.smoothed_params(u)
```

Returns the filtered copula parameter at each time step — the expectation under the predictive distribution.

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test
gof = gof_test(copula, u, to_pobs=False)
```

Uses the Rosenblatt transform with Cramér-von Mises statistic. For SCAR models, integrates the h-function over the predictive distribution of the latent state (mixture Rosenblatt).

### Predictive distribution

```python
z_grid, prob = copula.xT_distribution(u)
```

Returns the grid and probability weights for the latent state at time T.
