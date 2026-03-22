# Bivariate Copulas

## The SCAR model

In the stochastic copula autoregressive (SCAR) model, the copula parameter follows a latent Ornstein-Uhlenbeck process:

$$\theta_t = \Psi(x_t), \qquad dx_t = \theta_\text{OU}(\mu - x_t)\,dt + \nu\,dW_t$$

The three parameters $\alpha = (\theta_\text{OU}, \mu, \nu)$ control:

- $\theta_\text{OU}$ — mean-reversion speed
- $\mu$ — long-run mean of the latent process ($\Psi(\mu) \approx$ MLE parameter)
- $\nu$ — volatility of the latent process (higher = more dynamic)

## Transfer matrix method

The likelihood integrates over all latent paths. The transfer matrix method discretizes the latent state on a grid of $K$ points, reducing the path integral to a sequence of matrix-vector products:

$$\mathbf{m}_t = \widetilde{\mathbf{T}}(\mathbf{f}_t \odot \mathbf{m}_{t+1})$$

Complexity: $O(TKb)$ where $b$ is the transition kernel bandwidth (sparse regime) or $O(TK^2)$ (dense).

## Fitting

```python
from pyscarcopula import GumbelCopula

copula = GumbelCopula(rotate=180)
result = copula.fit(u, method='scar-tm-ou')

print(result.alpha)            # (theta_OU, mu, nu)
print(result.log_likelihood)   # log-likelihood
print(result.nfev)             # number of function evaluations
```

## Rotations

Rotations transform $(u_1, u_2)$ before evaluation, allowing the same copula family to capture different tail dependence patterns:

| Rotation | Transform | Tail dependence |
|----------|-----------|-----------------|
| 0° | $(u_1, u_2)$ | Upper tail (Gumbel, Joe) or lower tail (Clayton) |
| 90° | $(1-u_1, u_2)$ | Mixed |
| 180° | $(1-u_1, 1-u_2)$ | Opposite tail |
| 270° | $(u_1, 1-u_2)$ | Mixed |

For financial data (joint crashes), `GumbelCopula(rotate=180)` or `ClaytonCopula(rotate=0)` are common choices.

## Diagnostics

### Smoothed parameter

```python
theta_t = copula.smoothed_params(u)
```

Returns the filtered copula parameter $\bar{\theta}_k = \mathbb{E}[\Psi(x_k) \mid u_{1:k-1}]$ at each time step — the expectation under the predictive distribution from the transfer matrix forward pass.

### Goodness of fit

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, to_pobs=False)
```

Uses the mixture Rosenblatt transform, which integrates the h-function over the full predictive distribution of the latent state rather than plugging in a point estimate.

### Predictive distribution

```python
z_grid, prob = copula.xT_distribution(u)
```

Returns the grid and probability weights for the latent state at time $T$, useful for scenario generation.
