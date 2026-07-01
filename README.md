# pyscarcopula

A Python library for dynamic copula modelling: bivariate, multivariate, vine, and
stochastic copula models for financial time series and risk analytics.

* [About](#about)
* [Install](#install)
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
| Maximum likelihood | `mle` | Constant copula parameter |
| SCAR transfer matrix | `scar-tm-ou` | Deterministic OU latent-state likelihood |
| SCAR Jacobi transfer matrix | `scar-tm-jacobi` | Deterministic Kendall-tau diffusion likelihood |
| SCAR Monte Carlo | `scar-p-ou`, `scar-m-ou` | Monte Carlo alternatives |
| GAS | `gas` | Observation-driven score model |

## Install

```bash
pip install pyscarcopula
```

The package includes a required pybind11 C++ extension. It provides built-in
copula kernels, static likelihoods, GAS, and SCAR-TM-OU numerical evaluation.
Official wheels bundle the extension and do not need a local compiler. Source
and editable installs require a C++17 compiler:

* Windows: Microsoft C++ Build Tools / Visual Studio Build Tools
* Linux: GCC or Clang with the usual Python development headers
* macOS: Xcode Command Line Tools

For local development:

```bash
git clone https://github.com/AANovokhatskiy/pyscarcopula
cd pyscarcopula
pip install -e ".[test]"
```

To run the full test suite from the source tree, build the C++ extension in
place first:

```bash
python setup.py build_ext --inplace
pytest --run-validation
```

`pytest --run-validation` enables optional validation tests. A source checkout
without a successfully built extension is incomplete for the default
bivariate GAS workflow.

Optional benchmark and large validation checks are disabled by default. Enable
them explicitly:

```powershell
$env:PYSCA_RUN_BENCHMARKS="1"; $env:PYSCA_RUN_LARGE_BENCHMARKS="1"; $env:PYSCA_RUN_VINE_BENCHMARKS="1"; pytest tests --run-validation
```

Core dependencies: `numpy`, `numba`, `scipy`, `joblib`, `tqdm`.

Verify a native installation with:

```bash
python -m pyscarcopula._native_smoke
```

## Features

**Copula families**

* Archimedean: Gumbel, Frank, Clayton, Joe, including rotations where supported
* Elliptical: Gaussian and Student-t
* Independence copula for null models and vine pruning
* Multivariate Gaussian, Student-t, equicorrelation, and stochastic Student models
* Explicit `CopulaBase` / `BivariateCopula` / `MultivariateCopula` hierarchy
  with capability-based strategy validation

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
* Predictive time-varying copula parameter paths
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
state to the valid parameter domain. `scar-tm-jacobi` instead evolves
Kendall's tau directly with a bounded Jacobi diffusion and maps tau back to the
copula parameter for families that implement `tau_to_param`.

The transfer matrix method evaluates the latent-state likelihood by exploiting
the Markov structure of the latent process. The path integral is computed as a
sequence of matrix-vector products on a discretized grid or spectral basis,
avoiding Monte Carlo variance at the cost of numerical approximation.

For SCAR-TM-OU, `transition_method='auto'` uses a hybrid deterministic strategy:
Hermite spectral evaluation where it is reliable, matrix-based transition
evaluation for regimes better handled on a grid, and local Gauss-Hermite in
narrow-kernel OU cases. In broad terms, this keeps the latent path integral as
repeated deterministic linear-algebra updates while choosing the most suitable
transition representation automatically. See
[`docs/guide/performance.md`](docs/guide/performance.md) for the details and
the available `transition_method` values.

SCAR-TM-OU uses the bundled C++ extension as its only production numerical
engine.

```python
result = fit(copula, u, method="scar-tm-ou")
```

GAS uses the compiled numerical evaluator for likelihood, score recursion,
state updates, prediction, and Rosenblatt paths:

```python
result = fit(copula, u, method="gas")
```

Use the default `scaling="unit"` for production. `scaling="fisher"` remains an
experimental, numerically sensitive mode.

See [`docs/guide/performance.md`](docs/guide/performance.md) for supported C++
families and numerical options.

Custom Python copulas remain useful with custom Python strategies, sampling,
and diagnostics. Built-in native production strategies do not execute
arbitrary Python copula kernels: unsupported classes fail before optimization.

Vine copulas decompose a `d`-dimensional dependence model into bivariate copulas
arranged in a sequence of trees. R-vines choose the tree structure from data
subject to the proximity condition; C-vines use a fixed star structure.

## Examples and docs

Worked notebooks are available in [`examples/`](examples/):

* [`01_basic_api.ipynb`](examples/01_basic_api.ipynb)
* [`02_bivariate.ipynb`](examples/02_bivariate.ipynb)
* [`03_multivariate.ipynb`](examples/03_multivariate.ipynb)
* [`04_vine.ipynb`](examples/04_vine.ipynb)
* [`05_risk_metrics.ipynb`](examples/05_risk_metrics.ipynb)
* [`06_pyvinecopulib_comparison.ipynb`](examples/06_pyvinecopulib_comparison.ipynb)

Additional documentation is in [`docs/`](docs/). Method semantics are described
in [`docs/guide/estimation-methods.md`](docs/guide/estimation-methods.md), and
performance-related details are kept in
[`docs/guide/performance.md`](docs/guide/performance.md).
Migration notes for the native-core and multivariate namespace changes are in
[`docs/release-notes/native-core-migration.md`](docs/release-notes/native-core-migration.md).

## License

MIT License. See [`LICENSE.txt`](LICENSE.txt).
