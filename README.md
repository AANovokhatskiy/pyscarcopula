# pyscarcopula

Stochastic copula models with Ornstein-Uhlenbeck latent process in Python.

* [About](#about)
* [Install](#install)
* [Features](#features)
* [Mathematical background](#mathematical-background)
* [Examples](#examples)
* [Performance tuning](#performance-tuning)
* [Architecture](#architecture)
* [License](#license)

## About

**pyscarcopula** fits multivariate distributions using the copula approach with time-varying dependence. The classical constant-parameter model is extended to a stochastic model where the copula parameter follows an Ornstein-Uhlenbeck process.

For parameter estimation we provide five methods:

| Method              | Key              | Description                                                                                                                   |
| ------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Maximum likelihood  | `mle`            | Classical fit with constant copula parameter                                                                                  |
| MC p-sampler        | `scar-p-ou`      | Monte Carlo without importance sampling                                                                                       |
| MC m-sampler        | `scar-m-ou`      | Monte Carlo with efficient importance sampling (EIS)                                                                          |
| **Transfer matrix** | **`scar-tm-ou`** | **Deterministic quadrature on a grid — no Monte Carlo variance; accuracy depends on grid resolution and truncation settings** |
| GAS                 | `gas`            | Generalized autoregressive score (observation-driven, deterministic)                                                          |

The transfer matrix method exploits the Markov structure and known Gaussian transition density of the OU process to evaluate the likelihood function as a sequence of matrix-vector products. The implementation automatically selects between dense and sparse transfer matrices depending on the kernel bandwidth, and adaptively refines the grid to resolve the transition kernel.

## Install

```bash
pip install pyscarcopula
```

For development (includes data files and tests):

```bash
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install -e ".[test]"
pytest tests/
```

**Dependencies:** numpy, numba, scipy, joblib, tqdm.

## Features

**Copula families**

* Archimedean: Gumbel, Frank, Clayton, Joe (with rotations 0°/90°/180°/270°)
* Elliptical: Gaussian, Student-t (MLE only)
* Stochastic Student-t: d-dimensional t-copula with OU-driven degrees of freedom
* Independence copula (null model for automatic vine pruning)
* Equicorrelation Gaussian (single dynamic correlation for d assets)

**Vine copulas**

* C-vine pair copula construction (fixed star structure)
* R-vine pair copula construction (data-driven structure via Dissmann's MST algorithm)
* Automatic copula family and rotation selection per edge (AIC/BIC)
* Automatic pruning of weak edges via independence copula baseline
* Tree-level and edge-level truncation for scalability
* Mixed SCAR/MLE edges within a single vine

**Estimation methods**

* MLE — constant copula parameter
* SCAR-TM-OU — transfer matrix with analytical gradient (recommended)
* GAS — observation-driven score model
* SCAR-P-OU / SCAR-M-OU — Monte Carlo alternatives

**Sampling and prediction**

* `sample` — generate synthetic data reproducing the fitted model. For SCAR models an OU trajectory is simulated with the correct time discretization; for GAS a recursive score-driven simulation is used.
* `predict` — generate samples for next-step forecasting, conditional on the observed data. For SCAR-TM this uses mixture sampling from the posterior distribution `p(x_T | data)`, accounting for latent-state uncertainty. For GAS it uses the last filtered value.

**Diagnostics and risk**

* Goodness-of-fit via Rosenblatt transform + Cramér–von Mises test
* Mixture Rosenblatt transform for stochastic models
* Predictive time-varying copula parameter path
* VaR / CVaR via `pyscarcopula.contrib` (rolling window, marginals, portfolio optimization)

## Mathematical background

### Copula models

By Sklar's theorem, any joint distribution can be decomposed as

$$F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))$$

We focus on single-parameter Archimedean copulas defined via a generator φ(t; θ):

$$C(u_1, \ldots, u_d) = \phi^{-1}(\phi(u_1; \theta) + \cdots + \phi(u_d; \theta))$$

| Copula  | Generator                        | Inverse generator                 | Domain     |
| ------- | -------------------------------- | --------------------------------- | ---------- |
| Gumbel  | (-log t)^θ                       | exp(-t^(1/θ))                     | θ ∈ [1, ∞) |
| Frank   | -log((e^(-θt) - 1)/(e^(-θ) - 1)) | -(1/θ)log(1 + e^(-t)(e^(-θ) - 1)) | θ ∈ (0, ∞) |
| Joe     | -log(1 - (1-t)^θ)                | 1 - (1 - e^(-t))^(1/θ)            | θ ∈ [1, ∞) |
| Clayton | (1/θ)(t^(-θ) - 1)                | (1 + tθ)^(-1/θ)                   | θ ∈ (0, ∞) |

### Stochastic copula (SCAR)

In the stochastic model the copula parameter is driven by a latent Ornstein-Uhlenbeck process:

$$\theta_t = \Psi(x_t), \qquad dx_t = \theta_{\text{OU}}(\mu - x_t),dt + \nu,dW_t$$

where Ψ maps the OU state to the copula parameter domain. The likelihood function is an integral over latent paths:

$$L = \int p(x_0),\prod_t c(u_{1t}, u_{2t}; \Psi(x_t)),\prod_{t\ge 1} p(x_t \mid x_{t-1}),dx_{0:T}$$

In the current implementation, the latent process is initialized from the stationary OU distribution.

### Transfer matrix method

The Markov property allows this high-dimensional integral to be factored into a chain of one-dimensional integrals, each computed as a matrix-vector product on a discretized grid. Total complexity is `O(TK²)` in the dense case and `O(TKb)` in the sparse case, where `b` is the effective kernel bandwidth.

The method is deterministic, but it is still a numerical approximation: accuracy depends on the grid range, adaptive grid size, and transition-kernel truncation.

### Vine copulas

Vine copulas decompose a d-dimensional dependence structure into `d(d-1)/2` bivariate copulas arranged in a tree hierarchy. Two types are supported:

* **C-vine**: fixed star structure where one variable is the root of each tree.
* **R-vine**: data-driven structure selected by Dissmann's algorithm — at each tree level a maximum spanning tree is built on `|Kendall's τ|`, subject to the proximity condition.

Each edge in the vine can use a different copula family and estimation method (MLE, SCAR-TM, or GAS). Weak edges can be replaced with independence copulas.

### Goodness of fit

Model quality is assessed via the Rosenblatt transform. For stochastic models the implementation uses a **mixture Rosenblatt transform**, which integrates the h-function over the predictive distribution of the latent state, reducing the Jensen bias that appears when plugging in a point estimate.

Uniformity of the transformed sample is tested with the Cramér–von Mises statistic. As with other plug-in goodness-of-fit procedures, finite-sample calibration can vary, especially after refitting on simulated samples.

## Examples

### 1. Read dataset

```python
import pandas as pd
import numpy as np
from pyscarcopula._utils import pobs
from pyscarcopula import (
    GumbelCopula, CVineCopula, RVineCopula,
    GaussianCopula, StudentCopula, StochasticStudentCopula,
)
from pyscarcopula.api import fit, sample, predict, smoothed_params
from pyscarcopula.stattests import gof_test

crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep=';')
tickers = ['BTC-USD', 'ETH-USD']

returns = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:].values
u = pobs(returns)
```

### 2. Fit a bivariate copula

```python
copula = GumbelCopula(rotate=180)

result_mle = fit(copula, u, method='mle')
result_tm  = fit(copula, u, method='scar-tm-ou')
result_gas = fit(copula, u, method='gas')

print(f"MLE:     logL={result_mle.log_likelihood:.2f}, r={result_mle.copula_param:.4f}")
print(f"SCAR-TM: logL={result_tm.log_likelihood:.2f}, theta={result_tm.params.theta:.2f}")
print(f"GAS:     logL={result_gas.log_likelihood:.2f}, beta={result_gas.beta:.4f}")

# GoF test
gof = gof_test(copula, u, fit_result=result_tm, to_pobs=False)
print(f"GoF p-value: {gof.pvalue:.4f}")
```

Results on daily BTC-ETH data (`T = 1460`):

| Model       | logL        | GoF p-value |
| ----------- | ----------- | ----------- |
| MLE         | 955.63      | 0.0087      |
| GAS         | 1031.42     | 0.5282      |
| **SCAR-TM** | **1042.47** | **0.6201**  |

These numbers are dataset-specific and provided as an illustration, not as a benchmark guarantee.

### 3. Predictive copula parameter path

```python
r_t = smoothed_params(copula, u, result_tm)
# r_t[k] = E[Psi(x_k) | u_{1:k-1}]
```

This quantity is predictive: it uses information up to time `k-1`, not a full smoother using all observations.

### 4. Sampling and prediction

The API provides two distinct sampling functions:

* `sample` generates synthetic data that reproduces the fitted model. Useful for validation: `fit(copula, sample(...))` should recover similar parameters on average.
* `predict` generates samples for next-step forecasting, conditional on the observed data. Used for risk metrics.

```python
from pyscarcopula.api import sample, predict

# Model validation: sample -> refit -> compare parameters
v = sample(copula, u, result_tm, n=2000)
result_refit = fit(copula, pobs(v), method='scar-tm-ou')

# GoF on sampled data: should not show systematic rejection across repeated runs
gof_v = gof_test(copula, pobs(v), fit_result=result_refit, to_pobs=False)
print(f"GoF on sample: p={gof_v.pvalue:.4f}")

# Prediction: conditional on current market state
u_pred = predict(copula, u, result_tm, n=100_000)
```

How `sample` and `predict` work for each method:

| Method  | `sample(n)`                                     | `predict(n)`                                     |
| ------- | ----------------------------------------------- | ------------------------------------------------ |
| MLE     | n observations with constant r                  | same as sample                                   |
| SCAR-TM | OU trajectory r(t) → copula.sample(n, r)        | mixture sampling from posterior `p(x_T \| data)` |
| GAS     | recursive simulation: sample → score → update f | copula.sample(n, r=Ψ(f_T))                       |
| SCAR-MC | OU trajectory (same as SCAR-TM)                 | sample from stationary OU                        |

For SCAR-TM, `predict` accounts for latent-state uncertainty by sampling the copula parameter from the posterior distribution over the latent state rather than using a point estimate.

### 5. Fit a vine copula

C-vine (fixed star structure):

```python
tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD']
returns_6d = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:251].values
u6 = pobs(returns_6d)

vine = CVineCopula()
vine.fit(u6, method='scar-tm-ou',
         truncation_level=2, min_edge_logL=10)
vine.summary()
```

R-vine (data-driven structure via Dissmann's MST algorithm):

```python
vine = RVineCopula()
vine.fit(u6, method='scar-tm-ou',
         truncation_level=2, min_edge_logL=10)
vine.summary()
```

Results on 6-dimensional crypto data (`T = 250`):

| Model              | logL      | GoF p-value |
| ------------------ | --------- | ----------- |
| **C-vine SCAR-TM** | **924.8** | **0.63**    |
| R-vine SCAR-TM     | 919.1     | 0.62        |
| R-vine MLE         | 873.0     | 0.19        |
| C-vine MLE         | 869.2     | 0.21        |
| Student-t          | 764.4     | 0.00        |

These numbers are dataset-specific and seed-dependent.

Sampling and prediction are implemented for both vine types:

```python
# Model validation
v6 = vine.sample(2000)
gof_v6 = gof_test(vine, pobs(v6), to_pobs=False)

# Prediction (for risk metrics)
u_pred_6d = vine.predict(100_000, u=u6)
```

### 6. Stochastic Student-t copula

A d-dimensional t-copula where the degrees-of-freedom parameter follows an OU process:

```python
cop = StochasticStudentCopula(d=6)
cop.fit(returns_6d, method='scar-tm-ou', to_pobs=True)

# Predictive df(t) path
df_t = cop.smoothed_params()

# Predict with time-varying df
pred = cop.predict(10000)

# GoF
gof = gof_test(cop, returns_6d, to_pobs=True)
```

### 7. Risk metrics (VaR / CVaR)

Rolling VaR and CVaR estimation with copula models. Supports bivariate, vine, and elliptical copulas (Gaussian, Student-t).

```python
from pyscarcopula.contrib.risk_metrics import risk_metrics

result = risk_metrics(
    GumbelCopula(rotate=180),
    returns_6d, window_len=100,
    gamma=[0.95], N_mc=[100_000],
    marginals_method='johnsonsu',
    method='mle',
    optimize_portfolio=False,
    portfolio_weight=np.ones(6) / 6,
    n_jobs=-1,
)

var = result[0.95][100_000]['var']
cvar = result[0.95][100_000]['cvar']
```

For elliptical copulas:

```python
result = risk_metrics(
    GaussianCopula(), returns_6d, window_len=100,
    gamma=[0.95], N_mc=[100_000],
    marginals_method='johnsonsu',
    method='mle',  # ignored for Gaussian/Student (uses its own MLE)
)
```

See `example_new_api.ipynb` for a complete walkthrough with plots.

## Performance tuning

### Bivariate copula

```python
copula = GumbelCopula(rotate=180)
result = fit(copula, u, method='scar-tm-ou')

# Relaxed tolerance (faster, slight logL loss)
result = fit(copula, u, method='scar-tm-ou', tol=5e-2)
```

| Parameter         | Default | Effect                                                                                                                      |
| ----------------- | ------- | --------------------------------------------------------------------------------------------------------------------------- |
| `analytical_grad` | `True`  | Analytical gradient. ~3–4x fewer function evaluations.                                                                      |
| `smart_init`      | `True`  | Heuristic initial point. Up to 5x speedup on long series.                                                                   |
| `tol`             | `1e-2`  | Gradient tolerance. `5e-2` is often faster with small logL loss.                                                            |
| `K`               | `300`   | Minimum grid size. The adaptive rule may increase it.                                                                       |
| `pts_per_sigma`   | `2`     | Grid density: points per conditional standard deviation. Higher values improve quadrature accuracy at the cost of larger K. |

### Vine copula

```python
vine = RVineCopula()
vine.fit(u, method='scar-tm-ou', truncation_level=2, min_edge_logL=10)
```

| Parameter          | Default | Effect                                                                                        |
| ------------------ | ------- | --------------------------------------------------------------------------------------------- |
| `truncation_level` | `None`  | Trees at or above this level are not fitted with dynamic SCAR updates.                        |
| `truncation_fill`  | `'mle'` | Policy for edges above `truncation_level`: static MLE selection or forced independence.       |
| `min_edge_logL`    | `None`  | Edges with low MLE log-likelihood can be kept static rather than upgraded to a dynamic model. |

### Sampling performance

For vine copulas with GAS edges, `sample` uses step-by-step simulation, which is slower than the more vectorized paths used for MLE and SCAR models.

## Architecture

The codebase is organized in layers with top-down dependencies:

| Layer     | Directory                | Responsibility                                                                     |
| --------- | ------------------------ | ---------------------------------------------------------------------------------- |
| API       | `api.py`                 | Entry points: `fit()`, `sample()`, `predict()`, `smoothed_params()`, `mixture_h()` |
| Strategy  | `strategy/`              | Estimation methods: MLE, SCAR-TM, GAS. Each implements `fit`, `sample`, `predict`. |
| Copula    | `copula/`                | Pure math: PDF, h-functions, transforms, base sampling                             |
| Vine      | `vine/`                  | Vine copula models: C-vine, R-vine, edge/selection/structure utilities             |
| Numerical | `numerical/`             | TM grid, gradient, MC samplers, GAS filter, OU kernels                             |
| Types     | `_types.py`, `_utils.py` | Typed results, config, shared utilities                                            |
| Contrib   | `contrib/`               | Risk metrics, marginal distributions                                               |

All functions in `api.py` are stateless: they accept a copula object, data, and a result, and return new values without mutation. Convenience methods on copula classes (`copula.fit()`, `copula.predict()`) delegate to the API internally and store state for chained calls.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map.

## License

MIT License. See [LICENSE.txt](LICENSE.txt).
