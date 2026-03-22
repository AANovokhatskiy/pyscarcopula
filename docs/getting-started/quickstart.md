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
result_mle = copula_mle.fit(u, method='mle')

# Time-varying parameter (SCAR transfer matrix)
copula_tm = GumbelCopula(rotate=180)
result_tm = copula_tm.fit(u, method='scar-tm-ou')

print(f"MLE:     logL = {result_mle.log_likelihood:.2f}")
print(f"SCAR-TM: logL = {result_tm.log_likelihood:.2f}")
```

## Goodness-of-fit test

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula_tm, u, to_pobs=False)
print(f"p-value = {gof.pvalue:.4f}")
```

Under correct model specification, the Rosenblatt-transformed observations are iid Uniform. The test uses the Cramér-von Mises statistic.

## Smoothed copula parameter

```python
theta_t = copula_tm.smoothed_params(u)
# theta_t[k] = E[Ψ(x_k) | u_{1:k-1}]
```

## Fit a multivariate C-vine

```python
from pyscarcopula import CVineCopula

tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD']
returns_6d = np.log(prices[tickers] / prices[tickers].shift(1))[1:251].values
u6 = pobs(returns_6d)

vine = CVineCopula()
vine.fit(u6, method='scar-tm-ou', 
         truncation_level=2, min_edge_logL=10)
vine.summary()
```

## Sample and predict

```python
# Sample from fitted model
samples = copula_tm.predict(n=10000)

# Vine prediction
vine_samples = vine.predict(n=10000)
```

## Available copula families

| Family | Class | Rotations | SCAR support |
|--------|-------|-----------|--------------|
| Gumbel | `GumbelCopula` | 0, 90, 180, 270 | Yes |
| Clayton | `ClaytonCopula` | 0, 90, 180, 270 | Yes |
| Frank | `FrankCopula` | 0 | Yes |
| Joe | `JoeCopula` | 0, 90, 180, 270 | Yes |
| Independence | `IndependentCopula` | — | — |
| Gaussian | `GaussianCopula` | — | MLE only |
| Student-t | `StudentCopula` | — | MLE only |

## Available estimation methods

| Method | Key | Description |
|--------|-----|-------------|
| MLE | `'mle'` | Constant copula parameter |
| SCAR-TM-OU | `'scar-tm-ou'` | Transfer matrix (recommended) |
| GAS | `'gas'` | Observation-driven score model |
| SCAR-P-OU | `'scar-p-ou'` | Monte Carlo p-sampler |
| SCAR-M-OU | `'scar-m-ou'` | Monte Carlo m-sampler with EIS |
