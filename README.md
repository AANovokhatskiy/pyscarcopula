# pyscarcopula

Stochastic copula models with Ornstein-Uhlenbeck latent process in Python.

- [About](#about)
- [Install](#install)
- [Features](#features)
- [Mathematical background](#mathematical-background)
  * [Copula models](#copula-models)
  * [Stochastic copula (SCAR)](#stochastic-copula-scar)
  * [Transfer matrix method](#transfer-matrix-method)
  * [Goodness of fit](#goodness-of-fit)
- [Examples](#examples)
  * [1. Read dataset](#read-dataset)
  * [2. Initialize and fit a bivariate copula](#fit-bivariate)
  * [3. Sample from copula](#sample-copula)
  * [4. Fit a multivariate C-vine copula](#fit-vine)
  * [5. Risk metrics (VaR / CVaR)](#risk-metrics)
<!-- - [Citation](#citation) -->
- [License](#license)

## About

**pyscarcopula** fits multivariate distributions using the copula approach with time-varying dependence. The classical constant-parameter model is extended to a stochastic model where the copula parameter follows an Ornstein-Uhlenbeck process. The idea is based on the discrete SCAR model of Liesenfeld and Richard (2003) and Hafner and Manner (2012); here we develop it in the continuous-time setting.

For parameter estimation we provide three methods:
| Method | Key | Description |
|---|---|---|
| Maximum likelihood | `mle` | Classical fit with constant copula parameter |
| MC p-sampler | `scar-p-ou` | Monte Carlo without importance sampling |
| MC m-sampler | `scar-m-ou` | Monte Carlo with efficient importance sampling (EIS) |
| **Transfer matrix** | **`scar-tm-ou`** | **Deterministic quadrature on a grid — numerically exact, no MC bias** |

The transfer matrix method exploits the Markov structure and known Gaussian transition density of the OU process to evaluate the likelihood function as a sequence of matrix-vector products with complexity $O(TK^2)$. The implementation automatically selects between dense and sparse transfer matrices depending on the kernel bandwidth, and adaptively refines the grid to resolve the transition kernel.

<!-- For details see our paper:

> A. A. Novokhatskiy, M. E. Semenov, *Robust numerical scheme for stochastic copula models* (2026). -->

## Install

```bash
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install .
```

**Dependencies:** numpy, numba, scipy, sympy, joblib.

## Features

**Copula families**
- Archimedean: Gumbel, Frank, Clayton, Joe (with rotations 0°/90°/180°/270°)
- Elliptical: Gaussian, Student-t (MLE only)
- C-vine copula (MLE and SCAR-TM-OU)

**Estimation methods**
- MLE with constant parameter
- SCAR-P-OU (Monte Carlo p-sampler)
- SCAR-M-OU (Monte Carlo m-sampler with EIS)
- SCAR-TM-OU (transfer matrix — recommended)

**Diagnostics**
- Goodness-of-fit test via Rosenblatt transform + Cramér–von Mises statistic
- Mixture Rosenblatt transform for stochastic models (accounts for latent state uncertainty)
- Smoothed copula parameter $\bar{\theta}_k = \mathbb{E}[\Psi(x_k) | u_{1:k-1}]$

**Risk metrics**
- VaR and CVaR calculation via Monte Carlo sampling from fitted copula
- CVaR-optimized portfolio weights (Rockafellar & Uryasev, 2000)
- Rolling window computation

## Mathematical background

### Copula models

By Sklar's theorem, any joint distribution can be decomposed as

```math
F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))
```

We focus on single-parameter Archimedean copulas defined via a generator $\phi(t; \theta)$:

```math
C(u_1, \ldots, u_d) = \phi^{-1}(\phi(u_1; \theta) + \cdots + \phi(u_d; \theta))
```

| Copula  | $\phi(t; \theta)$ | $\phi^{-1}(t; \theta)$ | $\theta \in$ |
|---------|-------------------|------------------------|--------------|
| Gumbel  | $(-\log t)^\theta$ | $\exp(-t^{1/\theta})$ | $[1, \infty)$ |
| Frank   | $-\log\left(\frac{e^{-\theta t} - 1}{e^{-\theta} - 1}\right)$ | $-\frac{1}{\theta}\log(1 + e^{-t}(e^{-\theta} - 1))$ | $(0, \infty)$ |
| Joe     | $-\log(1 - (1-t)^\theta)$ | $1 - (1 - e^{-t})^{1/\theta}$ | $[1, \infty)$ |
| Clayton | $\frac{1}{\theta}(t^{-\theta} - 1)$ | $(1 + t\theta)^{-1/\theta}$ | $(0, \infty)$ |

### Stochastic copula (SCAR)

In the stochastic model the copula parameter is driven by a latent Ornstein-Uhlenbeck process:

```math
\theta_t = \Psi(x_t), \qquad dx_t = \theta_{\text{OU}}(\mu - x_t)\,dt + \nu\,dW_t
```

where $\Psi$ maps the OU state to the copula parameter domain. The likelihood function is an integral over all latent paths:

```math
L = \int \prod_{t} c(u_{1t}, u_{2t}; \Psi(x_t))\;p(x_t | x_{t-1})\;dx_0 \cdots dx_T
```

### Transfer matrix method

The Markov property allows this high-dimensional integral to be factored into a chain of one-dimensional integrals, each computed as a matrix-vector product on a discretized grid. The backward recursion is:

```math
\mathbf{m}_t = \widetilde{\mathbf{T}}(\mathbf{f}_t \odot \mathbf{m}_{t+1}), \qquad t = T-1, \ldots, 1
```

where $\widetilde{\mathbf{T}}$ is the transition kernel with trapezoidal weights baked in, $\mathbf{f}_t$ is the copula density evaluated at observation $t$, and $\odot$ is the element-wise product. Total complexity: $O(TK^2)$ (dense) or $O(TKb)$ (sparse, where $b$ is the kernel bandwidth).

### Goodness of fit

Model quality is assessed via the Rosenblatt transform. For the stochastic model we use the **mixture Rosenblatt transform**:

```math
u'_{2,t} = \mathbb{E}\left[h(u_{2t}, u_{1t}; \Psi(x_t)) \mid u_{1:t-1}\right]
```

which integrates the h-function over the predictive distribution of the latent state, avoiding the Jensen bias from plugging in a point estimate. Uniformity of the transformed sample is tested with the Cramér–von Mises statistic.


## Examples

<a name="read-dataset"></a>
### 1. Read dataset

```python
import pandas as pd
import numpy as np
from pyscarcopula.utils import pobs
from pyscarcopula import (GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula,
                          GaussianCopula, StudentCopula, CVineCopula)
from pyscarcopula.stattests import gof_test

crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=';')
tickers = ['BTC-USD', 'ETH-USD']

returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:]
returns = returns_pd.values
u = pobs(returns)
```

<a name="fit-bivariate"></a>
### 2. Initialize and fit a bivariate copula

```python
copula_mle = GumbelCopula(rotate=180)
copula_tm = GumbelCopula(rotate=180)

# Static model (constant parameter)
fit_result_mle = copula_mle.fit(data=returns, method='mle', to_pobs=True)
gof_result_mle = gof_test(copula_mle, returns, to_pobs=True)

print(f"MLE: logL = {fit_result_mle.log_likelihood:.4f}, "
      f"theta = {fit_result_mle.copula_param:.4f}")
print(f"GoF: statistic={gof_result_mle.statistic:.4f}, "
      f"p-value={gof_result_mle.pvalue:.4f}")

# Stochastic model (transfer matrix)
fit_result_tm = copula_tm.fit(data=returns, method='scar-tm-ou', to_pobs=True)
gof_result_tm = gof_test(copula_tm, returns, to_pobs=True)

print(f"SCAR-TM: logL = {fit_result_tm.log_likelihood:.4f}, "
      f"alpha = {fit_result_tm.alpha}")
print(f"GoF: statistic={gof_result_tm.statistic:.4f}, "
      f"p-value={gof_result_tm.pvalue:.4f}")
```

```
MLE: logL = 955.6275, theta = 2.8318
GoF: statistic=0.7536, p-value=0.0094

SCAR-TM: logL = 1045.4997, alpha = [58.99857924  1.48765678  4.53265176]
GoF: statistic=0.0718, p-value=0.7404
```

Available rotations: `[0, 90, 180, 270]`. Available methods: `['mle', 'scar-p-ou', 'scar-m-ou', 'scar-tm-ou']`.

<a name="sample-copula"></a>
### 3. Sample from copula

```python
# Sample from constant-parameter copula
samples_mle = copula_mle.sample(n=1000, r=fit_result_mle.copula_param)

# Sample next state from stochastic copula
samples_tm = copula_tm.predict(n=1000)
```

<a name="fit-vine"></a>
### 4. Fit a multivariate C-vine copula

```python
tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD']
returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:251]
returns = returns_pd.values
u = pobs(returns)

vine = CVineCopula()
vine.fit(u, method='scar-tm-ou')
vine.summary()

copula_s = StudentCopula()
copula_g = GaussianCopula()
copula_s.fit(u)
copula_g.fit(u)

gof_result_vine = gof_test(vine, u, to_pobs=False)
gof_result_s = gof_test(copula_s, u, to_pobs=False)
gof_result_g = gof_test(copula_g, u, to_pobs=False)

print(f"C-vine (SCAR):  logL={vine.log_likelihood(u):.1f}, p={gof_result_vine.pvalue:.4f}")
print(f"Student:        logL={copula_s.log_likelihood(u):.1f}, p={gof_result_s.pvalue:.4f}")
print(f"Gaussian:       logL={copula_g.log_likelihood(u):.1f}, p={gof_result_g.pvalue:.4f}")
```

```
C-vine (SCAR):  logL=890.3, p=0.7609
Student:        logL=764.4, p=0.0001
Gaussian:       logL=761.0, p=0.0000
```

<a name="risk-metrics"></a>
### 5. Risk metrics (VaR / CVaR)

```python
from pyscarcopula.metrics import risk_metrics

d = len(tickers)
result = risk_metrics(
    vine, returns, 
    window_len=100,
    gamma=[0.95],
    MC_iterations=[100_000],
    marginals_method='johnsonsu',
    method='mle',
    optimize_portfolio=False,
    portfolio_weight=np.ones(d) / d,
)

var = result[0.95][100_000]['var']
cvar = result[0.95][100_000]['cvar']
```

See `example.ipynb` for a complete walkthrough with plots.

<!-- ## Citation

If you use this package in your research, please cite:

```bibtex
@article{novokhatskiy2026scar,
  title={Robust numerical scheme for stochastic copula models},
  author={Novokhatskiy, A. A. and Semenov, M. E.},
  year={2026}
}
``` -->

## License

MIT License. See [LICENSE.txt](LICENSE.txt).