# Quick Start

## Prepare data

pyscarcopula works with pseudo-observations — uniform marginals obtained from ranked data.

```python
import pandas as pd
import numpy as np
from pyscarcopula.utils import pobs

prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=';')
returns = np.log(prices[['BTC-USD', 'ETH-USD']] /
                 prices[['BTC-USD', 'ETH-USD']].shift(1))[1:].values
u = pobs(returns)
```

## Fit a bivariate copula

```python
from pyscarcopula import GumbelCopula

# Constant parameter (MLE)
copula_mle = GumbelCopula(rotate=180)
copula_mle.fit(u, method='mle')

# Time-varying parameter (SCAR)
copula = GumbelCopula(rotate=180, transform_type='softplus')
copula.fit(u, method='scar-tm-ou')

print(f"MLE:  logL = {copula_mle.fit_result.log_likelihood:.2f}")
print(f"SCAR: logL = {copula.fit_result.log_likelihood:.2f}")
```

## Goodness-of-fit test

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, to_pobs=False)
print(f"p-value = {gof.pvalue:.4f}")
```

## Smoothed copula parameter

```python
theta_t = copula.smoothed_params(u)
# theta_t[k] = E[Psi(x_k) | u_{1:k-1}]
```

## Fit a multivariate C-vine

```python
from pyscarcopula import CVineCopula

tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD']
returns_6d = np.log(prices[tickers] / prices[tickers].shift(1))[1:251].values
u6 = pobs(returns_6d)

vine = CVineCopula()
vine.fit(u6, method='scar-tm-ou',
         truncation_level=2, min_edge_logL=10,
         transform_type='softplus')
vine.summary()
```

## Available copula families

| Family | Class | Rotations | SCAR support |
|--------|-------|-----------|--------------|
| Gumbel | `GumbelCopula` | 0, 90, 180, 270 | Yes |
| Clayton | `ClaytonCopula` | 0, 90, 180, 270 | Yes |
| Frank | `FrankCopula` | 0 | Yes |
| Joe | `JoeCopula` | 0, 90, 180, 270 | Yes |
| Independence | `IndependentCopula` | — | — |
| Gaussian | `BivariateGaussianCopula` | — | Yes |
| Equicorrelation | `EquicorrGaussianCopula` | — | Yes |
| Gaussian (d-dim) | `GaussianCopula` | — | MLE only |
| Student-t (d-dim) | `StudentCopula` | — | MLE only |

## Available estimation methods

| Method | Key | Description |
|--------|-----|-------------|
| MLE | `'mle'` | Constant copula parameter |
| SCAR-TM-OU | `'scar-tm-ou'` | Transfer matrix with OU latent process |
| GAS | `'gas'` | Observation-driven score model |
