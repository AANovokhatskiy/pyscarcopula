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
| SCAR Monte Carlo | `scar-p-ou`, `scar-m-ou` | Monte Carlo alternatives |
| GAS | `gas` | Observation-driven score model |

## Install

```bash
pip install pyscarcopula
```

Official wheels include the compiled numerical extension and do not need a
local compiler. Source and editable installs require a C++17 compiler:

* Windows: Microsoft C++ Build Tools / Visual Studio Build Tools, or
  MinGW-w64 GCC (see below)
* Linux: GCC or Clang with the usual Python development headers
* macOS: Xcode Command Line Tools

On Windows, a MinGW-w64 GCC toolchain (for example the MSYS2 `ucrt64` GCC) can
be used instead of the Microsoft compiler by opting in explicitly:

```bash
PYSCA_CPP_COMPILER=mingw32 pip install .
# or, from the source tree:
python setup.py build_ext --compiler=mingw32 --inplace
```

The GCC runtime is linked statically, so the resulting extension does not need
MSYS2 DLLs at runtime. MSVC remains the default Windows toolchain.

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

```bash
PYSCA_RUN_BENCHMARKS=1 \
PYSCA_RUN_LARGE_BENCHMARKS=1 \
PYSCA_RUN_VINE_BENCHMARKS=1 \
pytest tests --run-validation
```

On Windows PowerShell:

```powershell
$env:PYSCA_RUN_BENCHMARKS = "1"
$env:PYSCA_RUN_LARGE_BENCHMARKS = "1"
$env:PYSCA_RUN_VINE_BENCHMARKS = "1"
pytest tests --run-validation
```

Conditional sampling has separate PR, distributional, nightly external/d=50,
and manual benchmark layers.  See the
[conditional sampling validation guide](docs/guide/conditional-sampling-validation.md)
for exact marker selections, oracle-only calibration, artifacts, and failure
triage.

Core dependencies: `numpy`, `numba`, `scipy`, `joblib`, `tqdm`.

Verify the compiled extension with:

```bash
python -m pyscarcopula._native_smoke
```

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

### Vine copulas

`VineCopula` is the primary API for regular vines. Omitting `structure`
selects an R-vine from data; C-vines and D-vines are fixed
[`RVineMatrix`](docs/api/vine.md) structures:

```python
import numpy as np

from pyscarcopula import VineCopula

rng = np.random.default_rng(2028)
vine_data = rng.random((400, 4))

# Data-driven regular vine
auto = VineCopula().fit(vine_data, method="mle")

# Fixed standard structures
c_vine = VineCopula.cvine(d=4).fit(vine_data, method="mle")
d_vine = VineCopula.dvine(d=4).fit(vine_data, method="mle")
```

For an arbitrary valid regular-vine tree sequence, construct
`RVineMatrix.from_trees(...)` and pass it as `VineCopula(structure=...)`.
The [Vine guide](docs/guide/vine.md) defines the decoded tree format and
explains family selection, truncation, and conditional sampling.

Use `vine.structure` for the `RVineMatrix` and
`vine.natural_order_matrix` when an integration specifically needs the
natural-order runtime matrix. `vine.matrix` is a compatibility property.
The raw `pyvinecopulib` matrix uses one-based labels and the opposite
tree-level order above each anti-diagonal entry, so it is not obtained by
simply adding one. See the
[matrix-layout conversion](docs/api/vine.md#matrix-layout-and-pyvinecopulib).

**Copula families**

* Archimedean: Gumbel, Frank, Clayton, Joe, including rotations where supported
* Elliptical: Gaussian and Student-t
* Independence copula for null models and vine pruning
* Multivariate Gaussian, Student-t, equicorrelation, and stochastic Student models
* Shared APIs for bivariate, multivariate, and vine models

**Vine copulas**

* One `VineCopula` runtime for auto-selected and fixed regular vines
* Fixed C-vine and D-vine factories backed by `RVineMatrix`
* Arbitrary valid structures built from decoded tree edges
* Automatic family and rotation selection per edge using AIC/BIC
* Tree-level and edge-level truncation
* Mixed MLE, SCAR, GAS, and independence edges within one vine

**Sampling and prediction**

* Unconditional sampling from fitted bivariate, multivariate, and vine models
* Conditional sampling for static and dynamic multivariate models and R-vines
* Exact and approximate conditional modes for R-vines
* `PredictConfig` for explicit prediction options
* Reproducible random generation via `rng`
* JSON persistence through `model.save()` and `ModelClass.load()`
  (`include_data=False` can omit stored training data)

**Diagnostics and risk**

* Rosenblatt-transform based goodness-of-fit tests
* Mixture Rosenblatt transform for stochastic models
* Predictive time-varying copula parameter paths
* VaR and CVaR utilities in `pyscarcopula.contrib`

**CPU parallelism**

* Explicit native threading for eligible multivariate row, emission,
  conditional-sampling, static-likelihood, and Monte Carlo kernels
* Process-level `fit_independent` and rolling `risk_metrics` execution
* Absolute one-thread default: omitted `n_threads` always means `1`, regardless
  of environment variables
* Dependency-free C++17 linear algebra without hidden BLAS or OpenMP pools

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

```python
result = fit(copula, u, method="scar-tm-ou")
```

```python
result = fit(copula, u, method="gas")
```

Use the default `scaling="unit"` for production. `scaling="fisher"` remains an
experimental, numerically sensitive mode.

See [`docs/guide/performance.md`](docs/guide/performance.md) for supported
families and numerical options.

Vine copulas decompose a `d`-dimensional dependence model into bivariate
copulas arranged in a sequence of trees. `VineCopula()` selects a regular-vine
structure from data subject to the proximity condition.
`VineCopula.cvine(...)` and `VineCopula.dvine(...)` use fixed standard
structures, while `VineCopula(structure=RVineMatrix.from_trees(...))` accepts
an arbitrary valid decoded tree sequence.

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
