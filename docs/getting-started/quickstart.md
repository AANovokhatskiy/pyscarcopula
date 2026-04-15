# Quick Start

## Prepare data

pyscarcopula works with pseudo-observations - uniform marginals obtained from ranked data.

```python
import pandas as pd
import numpy as np
from pyscarcopula._utils import pobs

prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=';')
returns = np.log(prices[['BTC-USD', 'ETH-USD']] /
                 prices[['BTC-USD', 'ETH-USD']].shift(1))[1:].values
u = pobs(returns)
```

## Fit a bivariate copula

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit, smoothed_params

copula = GumbelCopula(rotate=180)

# Constant parameter (MLE)
result_mle = fit(copula, u, method='mle')

# Time-varying parameter (SCAR)
result_tm = fit(copula, u, method='scar-tm-ou')

print(f"MLE:  logL = {result_mle.log_likelihood:.2f}")
print(f"SCAR: logL = {result_tm.log_likelihood:.2f}")
```

## Goodness-of-fit test

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=result_tm, to_pobs=False)
print(f"p-value = {gof.pvalue:.4f}")
```

## Smoothed copula parameter

```python
r_t = smoothed_params(copula, u, result_tm)
# r_t[k] = E[Psi(x_k) | u_{1:k-1}]
```

## Sample and predict

```python
from pyscarcopula.api import sample, predict

# sample: reproduce the fitted model (for validation)
v = sample(copula, u, result_tm, n=2000)
result_refit = fit(copula, pobs(v), method='scar-tm-ou')
gof_v = gof_test(copula, pobs(v), fit_result=result_refit, to_pobs=False)
print(f"GoF on sample: p={gof_v.pvalue:.4f}")  # should pass

# predict: next-step forecast (for risk metrics)
u_pred = predict(copula, u, result_tm, n=100_000)

# conditional forecast in pseudo-observation space
u_cond = predict(copula, u, result_tm, n=20_000, given={0: 0.35})
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

# Vine sampling and prediction
v6 = vine.sample(2000)
u_pred_6d = vine.predict(100_000, u=u6)

# Conditional vine forecast: fix one variable
u_pred_6d_cond = vine.predict(20_000, u=u6, given={2: 0.6})
```

## Available copula families

| Family | Class | Rotations | SCAR support |
|--------|-------|-----------|--------------|
| Gumbel | `GumbelCopula` | 0, 90, 180, 270 | Yes |
| Clayton | `ClaytonCopula` | 0, 90, 180, 270 | Yes |
| Frank | `FrankCopula` | 0 | Yes |
| Joe | `JoeCopula` | 0, 90, 180, 270 | Yes |
| Independence | `IndependentCopula` | - | - |
| Gaussian | `BivariateGaussianCopula` | - | Yes |
| Equicorrelation | `EquicorrGaussianCopula` | - | Yes |
| Stochastic Student-t | `StochasticStudentCopula` | - | Yes |
| Stochastic Student-t DCC | `StochasticStudentDCCCopula` | - | Yes |
| Gaussian (d-dim) | `GaussianCopula` | - | MLE only |
| Student-t (d-dim) | `StudentCopula` | - | MLE only |

## Available estimation methods

| Method | Key | Description |
|--------|-----|-------------|
| MLE | `'mle'` | Constant copula parameter |
| SCAR-TM-OU | `'scar-tm-ou'` | Transfer matrix with OU latent process |
| GAS | `'gas'` | Observation-driven score model |
