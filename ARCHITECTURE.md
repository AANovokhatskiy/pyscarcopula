# Architecture

## Module Map

```text
pyscarcopula/
|-- __init__.py              # Public re-exports and BLAS thread policy
|-- api.py                   # Stateless fit/predict/sample API
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
|   |-- _cpp_scar_ou.py, _cpp_gas.py
|   |-- jacobi_tm.py         # Retained Python Jacobi orchestration
|   `-- mc_samplers.py       # Retained Python SCAR-MC/EIS orchestration
|-- vine/
|   |-- cvine.py, rvine.py
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
vine/ -> strategy/ + bivariate copula contract
stattests/ -> fitted strategy outputs + retained GoF orchestration
```

## Native Boundary

The pybind11 C++ extension is mandatory. Built-in point operations, static
likelihoods, GAS filtering, and SCAR-TM-OU likelihood/gradient/forward
operations have one production implementation in C++.

Python remains responsible for:

- optimizer orchestration and immutable result construction;
- correlation parameterization and chain rules around native evaluators;
- RNG and conditional sampling;
- Jacobi filtering orchestration;
- SCAR-MC/EIS orchestration;
- goodness-of-fit and contribution analytics.

There is no GAS or SCAR-TM-OU backend selector and no Python likelihood
fallback.

SCAR-TM-OU joint Stochastic Student fits can hold a prepared native evaluator
for one optimizer loop. That object owns the copied observations, native
copula specification, Student PPF cache, and reusable gradient workspaces.
Python still owns the raw correlation parameterization and updates only the
native Student factor between objective calls. The direct functional adapters
remain stateless entry points for one-off evaluations.

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

The functional API is stateless:

```python
from pyscarcopula.api import fit, predict

result = fit(copula, u, method="scar-tm-ou")
samples = predict(copula, u, result, n=1000)
```

Model methods are convenience wrappers that store `fit_result` and the last
fitting data. Results themselves are immutable dataclasses.

Persistence uses versioned JSON. The loader migrates historical experimental
class paths to the multivariate namespace and ignores removed legacy backend
fields.

## BLAS Thread Policy

Package import sets common BLAS thread variables to one by default. This
avoids oversubscription during outer-level parallel work. Set
`PYSCA_BLAS_THREADS` before importing `pyscarcopula` to override the policy.
