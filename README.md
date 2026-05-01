# pyscarcopula

Stochastic copula models with Ornstein-Uhlenbeck latent processes in Python.

* [About](#about)
* [Install](#install)
* [Features](#features)
* [Mathematical background](#mathematical-background)
* [Examples and docs](#examples-and-docs)
* [License](#license)

## About

**pyscarcopula** fits bivariate and multivariate dependence models using copulas.
Alongside classical constant-parameter copulas, it supports stochastic copula
autoregressive (SCAR) models where the copula parameter is driven by a latent
Ornstein-Uhlenbeck process.

The package is aimed at financial time series, risk modelling, and experiments
with dynamic dependence. It provides bivariate copulas, C-vines, R-vines,
conditional sampling, prediction, goodness-of-fit diagnostics, and risk metrics.

Supported estimation methods:

| Method | Key | Description |
| --- | --- | --- |
| Maximum likelihood | `mle` | Constant copula parameter |
| SCAR transfer matrix | `scar-tm-ou` | Deterministic OU latent-state likelihood |
| SCAR Monte Carlo | `scar-p-ou`, `scar-m-ou` | Monte Carlo alternatives |
| GAS | `gas` | Observation-driven score model |

## Install

```bash
pip install pyscarcopula
```

For local development:

```bash
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install -e ".[test]"
pytest
```

Core dependencies: `numpy`, `numba`, `scipy`, `joblib`, `tqdm`.

## Features

**Copula families**

* Archimedean: Gumbel, Frank, Clayton, Joe, including rotations where supported
* Elliptical: Gaussian and Student-t
* Independence copula for null models and vine pruning
* Experimental models in `pyscarcopula.copula.experimental`

**Vine copulas**

* C-vine pair-copula construction with fixed star structure
* R-vine pair-copula construction with Dissmann-style structure selection
* Automatic family and rotation selection per edge using AIC/BIC
* Tree-level and edge-level truncation
* Mixed MLE, SCAR, GAS, and independence edges within one vine

**Sampling and prediction**

* Unconditional sampling from fitted bivariate and vine models
* Conditional sampling for R-vines, including exact suffix/rebuild paths and
  runtime-DAG plus MCMC fallback for arbitrary conditioning sets
* `PredictConfig` for explicit prediction options
* Reproducible random generation via `rng`
* JSON persistence through `model.save()` and `ModelClass.load()`

**Diagnostics and risk**

* Rosenblatt-transform based goodness-of-fit tests
* Mixture Rosenblatt transform for stochastic models
* Smoothed and predictive time-varying copula parameter paths
* VaR and CVaR utilities in `pyscarcopula.contrib`

## Mathematical background

By Sklar's theorem, a joint distribution can be represented as

```math
F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d)),
```

where `C` is a copula and `F_i` are marginal distributions. This separates
marginal modelling from dependence modelling.

For a one-parameter Archimedean copula with generator `phi`,

```math
C(u_1, \ldots, u_d; \theta)
  = \phi^{-1}(\phi(u_1; \theta) + \cdots + \phi(u_d; \theta)).
```

In SCAR models the copula parameter is time-varying:

```math
\theta_t = \Psi(x_t),
\qquad
dx_t = \kappa(\mu - x_t)dt + \nu dW_t,
```

where `x_t` is a latent Ornstein-Uhlenbeck process and `Psi` maps the latent
state to the valid parameter domain.

The transfer matrix method evaluates the latent-state likelihood by exploiting
the Markov structure of the OU process. The path integral is computed as a
sequence of matrix-vector products on a discretized grid, avoiding Monte Carlo
variance at the cost of numerical grid approximation.

Vine copulas decompose a `d`-dimensional dependence model into bivariate copulas
arranged in a sequence of trees. R-vines choose the tree structure from data
subject to the proximity condition; C-vines use a fixed star structure.

## Examples and docs

Worked notebooks are available in [`examples/`](examples/):

* [`01_basic_api.ipynb`](examples/01_basic_api.ipynb)
* [`02_bivariate.ipynb`](examples/02_bivariate.ipynb)
* [`03_vine.ipynb`](examples/03_vine.ipynb)
* [`04_risk_metrics.ipynb`](examples/04_risk_metrics.ipynb)
* [`05_pyvinecopulib_comparison.ipynb`](examples/05_pyvinecopulib_comparison.ipynb)

Additional documentation is in [`docs/`](docs/). Performance-related details are
kept in [`docs/guide/performance.md`](docs/guide/performance.md).

## License

MIT License. See [`LICENSE.txt`](LICENSE.txt).
