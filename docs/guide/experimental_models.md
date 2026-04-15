# Experimental Models

!!! warning "Experimental API"
    The models in `pyscarcopula.copula.experimental` are under active development. Their API may change in future releases.

## Overview

The `experimental` module contains d-dimensional copula models that extend the SCAR framework beyond bivariate Archimedean families. All models use a **single scalar latent OU process**, so the existing transfer matrix infrastructure works unchanged.

| Model | Class | Latent parameter | Description |
|-------|-------|-----------------|-------------|
| Equicorrelation Gaussian | `EquicorrGaussianCopula` | $\rho(t)$ | Single dynamic correlation for d assets |
| Stochastic Student-t | `StochasticStudentCopula` | $\nu(t)$ | Fixed correlation, OU-driven degrees of freedom |
| Stochastic Student-t DCC | `StochasticStudentDCCCopula` | $\nu(t)$ | DCC-driven correlation + OU-driven df |

## Equicorrelation Gaussian Copula

For d assets, the standard Gaussian copula has `d(d-1)/2` correlation parameters, all static. The equicorrelation model uses a single dynamic correlation:

$$R(t) = (1-\rho(t)) \cdot I + \rho(t) \cdot \mathbf{1}\mathbf{1}^\top$$

All pairwise correlations equal $\rho(t)$, which follows an OU process via SCAR. This gives 3 parameters instead of `d(d-1)/2`.

The density is analytical and `O(d)` per evaluation, with no matrix inversion required.

### Usage

```python
from pyscarcopula.copula.experimental import EquicorrGaussianCopula

cop = EquicorrGaussianCopula(d=6)

# MLE (constant rho)
cop.fit(u, method='mle')

# SCAR (time-varying rho)
cop.fit(u, method='scar-tm-ou')
```

### When to use

Equicorrelation SCAR is a good fit when:

- All pairwise correlations move together, common in equity and crypto markets
- You need fast estimation for large `d`, with `O(d)` density evaluation
- You want a simple interpretable model with 3 parameters

For heterogeneous dependence, use C-vine or R-vine instead.

## Stochastic Student-t Copula

A d-dimensional t-copula where the correlation matrix $R$ is estimated once (via Kendall's $\tau$) and fixed, while the degrees-of-freedom parameter $\nu(t)$ follows a latent OU process:

$$\nu(t) = 2 + \mathrm{softplus}(x(t)), \qquad x(t) \sim \text{OU}(\theta, \mu, \sigma)$$

The transform maps $\mathbb{R} \to (2, \infty)$, ensuring finite variance.

### Usage

```python
from pyscarcopula.copula.experimental import StochasticStudentCopula

cop = StochasticStudentCopula(d=6)

# Fit (R estimated automatically via Kendall tau)
result = cop.fit(returns, method='scar-tm-ou', to_pobs=True)

# Predictive df(t) path
df_t = cop.smoothed_params()

# Predict with time-varying df
pred = cop.predict(10000)

# GoF
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, returns, to_pobs=True)
```

### When to use

- When tail dependence varies over time
- When the correlation structure is relatively stable but tail thickness changes
- As an alternative to vine copulas for moderate dimensions

## Stochastic Student-t DCC Copula

Extends the Stochastic Student-t copula with a time-varying correlation matrix driven by a DCC(1,1) filter:

$$U_t \mid x_t, R_t \sim \text{t-copula}(R_t, \nu_t)$$

where $\nu_t = 2 + \mathrm{softplus}(x_t)$ is OU-driven and $R_t$ is a deterministic DCC path.

The latent state remains scalar, so all SCAR-TM-OU machinery works unchanged. The DCC estimation is a separate step.

### Usage

```python
from pyscarcopula.copula.experimental import StochasticStudentDCCCopula

cop = StochasticStudentDCCCopula(d=6)

# Fit (DCC + SCAR jointly)
result = cop.fit(returns, method='scar-tm-ou', to_pobs=True)

# Predict (uses last R_T or DCC forecast)
pred = cop.predict(10000)
```

### When to use

- When both correlation structure and tail heaviness change over time
- For capturing joint dynamics in volatile markets
- When a static correlation matrix is too restrictive but a full vine is too complex

## Common API

All experimental models support the same core operations:

```python
# Goodness of fit
from pyscarcopula.stattests import gof_test
gof = gof_test(cop, u, to_pobs=False)

# Sampling (fixed parameter)
samples = cop.sample(n=10000)

# Prediction (conditional on data)
pred = cop.predict(n=10000)

# Smoothed parameter path
params_t = cop.smoothed_params()
```
