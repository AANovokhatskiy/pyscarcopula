# pyscarcopula

A Python library for dynamic copula modelling: bivariate, multivariate, vine, and
stochastic copula models for financial time series and risk analytics.

* [About](#about)
* [Install](#install)
* [Quick start](#quick-start)
* [Features](#features)
* [Mathematical background](#mathematical-background)
* [Examples and docs](#examples-and-docs)
* [License](#license)

## About

**pyscarcopula** fits bivariate and multivariate dependence models using
copulas in Python. Alongside classical constant-parameter copulas, it supports
stochastic copula autoregressive (SCAR) models where the copula parameter is
driven by a latent Ornstein-Uhlenbeck process or Kendall's tau follows a
bounded Jacobi diffusion.

The package is aimed at financial time series, risk modelling, and experiments
with dynamic dependence. It provides bivariate copulas, C-vines, R-vines,
conditional sampling, prediction, goodness-of-fit diagnostics, and risk metrics.

Supported estimation methods:

| Method | Key | Description |
| --- | --- | --- |
| Maximum likelihood | `mle` | Static/constant model parameters |
| SCAR transfer matrix | `scar-tm-ou` | Deterministic OU latent-state likelihood |
| SCAR Jacobi transfer matrix | `scar-tm-jacobi` | Deterministic Kendall-tau diffusion likelihood |
| GAS | `gas` | Observation-driven score model |

## Install

```bash
pip install pyscarcopula
```

For source and editable builds, compiler selection, testing, benchmarks, and
documentation builds, see the [installation guide](docs/getting-started/installation.md).

## Quick start

pyscarcopula expects a two-dimensional array with observations in rows and
variables in columns. Copula fitting uses pseudo-observations in the open unit
interval. Pass existing pseudo-observations directly, as below, or use
`to_pobs=True` to rank-transform raw continuous observations during fitting.

```python
import numpy as np

from pyscarcopula import GumbelCopula

rng = np.random.default_rng(2026)
source = GumbelCopula(rotate=180)
u = source.sample_at_parameter(
    400,
    r=np.full(400, 1.8),
    rng=rng,
)

model = GumbelCopula(rotate=180)
result = model.fit(u, method="mle")
forecast = model.predict(1_000, rng=np.random.default_rng(2027))

print(result.log_likelihood)
print(forecast.shape)  # (1000, 2)
```

The fitted result is returned by `fit` and also stored as
`model.fit_result`. `sample(...)` reproduces the fitted model, while
`predict(...)` draws from its predictive distribution; this distinction
matters for dynamic models. See the complete
[Quick Start](docs/getting-started/quickstart.md) and
[Prediction Semantics](docs/guide/prediction-semantics.md).

Choose the model surface before tuning an estimation method:

| Task | Start with |
| --- | --- |
| Two-variable dependence | `GumbelCopula`, `ClaytonCopula`, `FrankCopula`, `JoeCopula`, or `BivariateGaussianCopula` |
| Static unrestricted multivariate dependence | `GaussianCopula` or `StudentCopula` |
| Scalar dynamic multivariate dependence | `EquicorrGaussianCopula` or `StochasticStudentCopula` |
| Flexible high-dimensional pair decomposition | `VineCopula` |

See [Choosing a Model](docs/getting-started/choosing-a-model.md) for the
corresponding correlation modes, estimation methods, and limitations.

## Features

The library covers [bivariate](docs/guide/bivariate.md),
[multivariate](docs/guide/multivariate_models.md), [factor](docs/guide/factor-models.md),
and [vine](docs/guide/vine.md) copulas. See the documentation for
[estimation methods](docs/guide/estimation-methods.md),
[sampling and prediction](docs/guide/prediction-semantics.md),
[diagnostics](docs/api/diagnostics.md), and
[CPU parallelism](docs/guide/parallelism.md).

## Mathematical background

By Sklar's theorem, a joint distribution can be represented as

```math
F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d)),
```

where `C` is a copula and `F_i` are marginal distributions. This separates
marginal and dependence modelling.

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

where `x_t` is a latent Ornstein-Uhlenbeck process and `Psi` maps it to the
valid parameter domain. The Jacobi variant models Kendall's tau with a bounded
diffusion. Deterministic likelihood evaluation uses grid, spectral, or local
quadrature backends.

For derivations and parameter mappings, see
[Mathematical Contracts](docs/guide/mathematical-contracts.md). Implementation
and grid details are covered by [Numerical Backends](docs/guide/numerical-backends.md),
with practical choices summarized in
[Performance Tuning](docs/guide/performance.md).

## Examples and docs

Worked notebooks are available in [`examples/`](examples/):

* [`01_basic_api.ipynb`](examples/01_basic_api.ipynb)
* [`02_bivariate.ipynb`](examples/02_bivariate.ipynb)
* [`03_multivariate.ipynb`](examples/03_multivariate.ipynb)
* [`04_vine.ipynb`](examples/04_vine.ipynb)
* [`05_risk_metrics.ipynb`](examples/05_risk_metrics.ipynb)
* [`06_pyvinecopulib_comparison.ipynb`](examples/06_pyvinecopulib_comparison.ipynb)

Additional documentation is in [`docs/`](docs/). Estimation methods are
described in [`docs/guide/estimation-methods.md`](docs/guide/estimation-methods.md),
and performance-related details are kept in
[`docs/guide/performance.md`](docs/guide/performance.md). CPU threading,
process workers, thread safety, and scaling limits are documented in
[`docs/guide/parallelism.md`](docs/guide/parallelism.md). Release history is in
[`CHANGELOG.md`](CHANGELOG.md).

## License

MIT License. See [`LICENSE.txt`](LICENSE.txt).

## Contacts

Contact me for any questions or discussion aanovokhatskiy@gmail.com
