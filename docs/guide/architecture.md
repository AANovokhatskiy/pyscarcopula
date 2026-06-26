# Model And Strategy Architecture

## Class Hierarchy

Every built-in copula derives from `CopulaBase`:

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

`BivariateCopula` exposes pair operations required by vines, including `h`
and inverse-`h`. `MultivariateCopula` exposes row-density and sampling
contracts without presenting a pair-copula API.

The corresponding runtime protocols are
`CommonCopulaProtocol`, `BivariateCopulaProtocol`, and
`MultivariateCopulaProtocol`.

## Capabilities

Inheritance describes model shape. `CopulaCapabilities` describes which
strategies and native operations a built-in model supports.

```python
from pyscarcopula import EquicorrGaussianCopula

copula = EquicorrGaussianCopula(d=6)
print(copula.capabilities.supports_gas)
print(copula.capabilities.supports_scar_ou)
```

The strategy layer validates capabilities before optimization. A multivariate
model is therefore not accepted by a pair-only strategy merely because it has
similarly named methods.

## Strategy Ownership

Strategies own optimization, filtering orchestration, and result
construction. Copulas own model metadata, transforms, and sampling behavior.
Native adapters own calls into the mandatory C++ extension.

| Layer | Main responsibility |
|-------|---------------------|
| Copula class | Model identity, parameter domain, sampling |
| Capability descriptor | Explicit strategy support |
| Strategy | Optimization and immutable fit results |
| Native evaluator | Density, likelihood, gradient, filtering |
| Python orchestration | RNG, Jacobi, MC/EIS, GoF, persistence |

## Numerical Safety Boundaries

Numerical boundaries are named by purpose rather than represented by a
generic `eps`. Python values live in `pyscarcopula._constants`; native
pair-copula values live in `scar/numerical_constants.hpp`.

- `PSEUDO_OBS_EPS` protects pseudo-observations passed to quantiles and
  h-functions.
- `H_FUNCTION_EPS` bounds numerical h/inverse-h outputs.
- `ROSENBLATT_OUTPUT_EPS` protects final GoF normal quantiles.
- `CONDITIONAL_SAMPLE_EPS` applies only to newly sampled free coordinates.
- `PDF_FLOOR` protects density and logarithm arguments.

The h-function and Rosenblatt boundaries use the same numeric value, but they
remain separate contracts. Vine runtime code uses the shared
pseudo-observation helper; it does not define local `_EPS` constants.

## Custom Python Copulas

Custom Python copulas can implement public protocols for custom strategies,
sampling, diagnostics, and other Python workflows. Protocol conformance alone
does not add a family to the native support matrix.

Built-in GAS and SCAR-TM-OU reject unknown classes before optimization. They do
not silently call arbitrary Python density methods as a fallback.

Custom estimation methods remain a Python extension point through
`register_strategy`.

## Public Imports

Base classes and capabilities are available at the package top level:

```python
from pyscarcopula import (
    BivariateCopula,
    CopulaBase,
    CopulaCapabilities,
    MultivariateCopula,
)
```

Multivariate models can be imported either from `pyscarcopula` or from
`pyscarcopula.copula.multivariate`.
