# Architecture

## Module Map

```text
pyscarcopula/
|-- __init__.py              # Public re-exports and BLAS thread policy
|-- api.py                   # Top-level fit/predict/sample helpers
|-- _types.py                # Results and numerical configuration
|-- io.py                    # Versioned JSON persistence and migrations
|-- stattests.py             # Goodness-of-fit orchestration
|-- copula/
|   |-- _protocol.py         # Common, bivariate, multivariate protocols
|   |-- base.py              # CopulaBase, BivariateCopula, capabilities
|   |-- gumbel.py, frank.py, joe.py, clayton.py
|   |-- elliptical.py        # Bivariate Gaussian copula
|   `-- multivariate/
|       |-- base.py          # MultivariateCopula
|       |-- gaussian.py, student.py
|       `-- equicorr.py, stochastic_student.py
|-- strategy/
|   |-- _base.py             # Strategy registry and capability validation
|   |-- mle.py, gas.py, scar_tm.py
|   `-- scar_jacobi.py, scar_mc.py
|-- numerical/
|   |-- copula_native.py, multivariate_native.py
|   |-- static_likelihood.py, gas_filter.py
|   |-- _cpp_scar_ou.py, _cpp_gas.py, _cpp_gas_rvine.py
|   |-- jacobi_tm.py         # Retained Python Jacobi orchestration
|   `-- mc_samplers.py       # Retained Python SCAR-MC/EIS orchestration
|-- vine/
|   |-- vine.py               # Canonical generic VineCopula runtime
|   |-- rvine.py              # RVineCopula compatibility module alias
|   |-- cvine.py              # Legacy CVineCopula implementation
|   |-- _structure.py         # RVineMatrix and C/D structure factories
|   |-- _vine_fit.py          # Shared fixed-structure edge fitting
|   `-- _rvine_*.py, _selection.py
`-- contrib/                 # Marginals and risk analytics
```

## Copula Hierarchy

All built-in copulas derive from `CopulaBase`.

```text
CopulaBase
|-- BivariateCopula
|   |-- ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula
|   |-- BivariateGaussianCopula
|   `-- IndependentCopula
`-- MultivariateCopula
    |-- GaussianCopula, StudentCopula
    |-- EquicorrGaussianCopula
    `-- StochasticStudentCopula
```

`BivariateCopula` supplies pair operations used by vines: density, `h`,
inverse-`h`, rotation handling, and scalar-parameter transforms.
`MultivariateCopula` supplies row-density and sampling contracts without
pretending to be a vine pair copula.

The runtime-checkable protocols in `copula/_protocol.py` describe structural
typing. They do not grant native support by themselves.

## Capabilities And Strategies

Class hierarchy answers what a model is. `CopulaCapabilities` answers which
built-in strategies and numerical operations it supports:

- `supports_pair_ops`
- `supports_native_mle`
- `supports_gas`
- `supports_scar_ou`
- `supports_latent_grid`
- `supports_conditional_sampling`
- `has_dynamic_scalar_parameter`

The strategy registry in `strategy/_base.py` validates these capabilities
before fitting. Strategy classes own optimization and result construction;
copula classes own model metadata, parameter transforms, and sampling.

The main dependency flow is:

```text
api.py -> strategy/ -> numerical native adapters -> C++ extension
                    -> copula model metadata
vine/vine.py -> structure selection or fixed RVineMatrix
             -> _vine_fit.py + bivariate copula contract
stattests/ -> fitted strategy outputs + retained GoF orchestration
```

## Native Boundary

The pybind11 C++ extension is mandatory. Built-in point operations, static
likelihoods, GAS filtering, multivariate conditional linear algebra,
sequential GAS R-vine sampling, and SCAR-TM-OU likelihood/gradient/forward
operations have one production implementation in C++.

Python remains responsible for:

- optimizer orchestration and result construction;
- correlation parameterization and chain rules around native evaluators;
- RNG and generation of fixed draws used by native conditional sampling;
- Jacobi filtering orchestration;
- SCAR-MC/EIS orchestration;
- goodness-of-fit and contribution analytics.

There is no GAS or SCAR-TM-OU backend selector and no Python likelihood
fallback.

SCAR-TM-OU joint Stochastic Student fits can hold a prepared native evaluator
for one optimizer loop. That object owns the copied observations, native
copula specification, Student PPF cache, and reusable gradient workspaces.
Python still owns the raw correlation parameterization and updates only the
native Student factor between objective calls. Direct functional adapters
remain available for one-off evaluations.

Gaussian and Student multivariate conditional kernels accept read-only native
views. C-contiguous NumPy `float64` inputs remain alive for the complete
synchronous binding call, including the section executed without the GIL, and
are not copied into C++ vectors. Non-contiguous arrays and other dtypes retain
pybind11's `forcecast` fallback. Every numeric input still receives one finite
validation pass before the GIL is released.

Static and prepared SCAR-OU equicorrelation evaluators cache per-row `sum(z)`
and `sum(z^2)` statistics for their owned observation snapshots. Repeated
objective, gradient, and forward calls therefore reuse both the per-row normal
scores and the statistics across every latent-grid node.

## Custom Python Extensions

User-defined Python copulas may implement the public protocols for their own
sampling, diagnostics, custom strategies, or other Python workflows. This
does not make them executable by native production strategies.

Built-in GAS and SCAR-TM-OU accept only copula families explicitly represented
by the native support matrix. Unknown classes fail before optimization instead
of calling arbitrary Python density methods from the native evaluator.

New estimation methods can still be registered in Python:

```python
from pyscarcopula.strategy._base import register_strategy

@register_strategy("MY-METHOD")
class MyStrategy:
    def __init__(self, config=None, **kwargs):
        self.config = config

    def fit(self, copula, u, **kwargs):
        ...
```

## State And Persistence

The top-level API can be used directly:

```python
from pyscarcopula.api import fit, predict

result = fit(copula, u, method="scar-tm-ou")
samples = predict(copula, u, result, n=1000)
```

Model methods are convenience wrappers that store `fit_result` and the last
fitting data.

Persistence uses versioned JSON. New files use v3; the loader retains v2
migrations, including the historical
`pyscarcopula.vine.rvine.RVineCopula` path and pre-`VineCopula` state layout.
Removing v2 support requires a separate file-format migration policy.

For generic vines, `RVineMatrix` is the canonical public structure. The model
stores a separate natural-order matrix for numerical traversal:

```text
VineCopula(structure=None)          -> Dissmann auto selection
VineCopula(structure=RVineMatrix)   -> fixed structure, no MST selection
VineCopula.cvine(...) / .dvine(...) -> fixed RVineMatrix factories
```

`RVineCopula` is the same runtime type as `VineCopula`.
`CVineCopula` remains a separate legacy implementation.

## BLAS Thread Policy

Package import does not mutate BLAS thread environment variables. Applications
that need a specific BLAS thread policy should configure their execution
environment before importing NumPy/SciPy, or use a runtime thread limiter such
as `threadpoolctl`.
