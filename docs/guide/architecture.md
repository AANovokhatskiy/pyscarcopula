# Developer Architecture

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
behavior without presenting a pair-copula API.

## Capabilities

Inheritance describes model shape. `CopulaCapabilities` describes which
strategies and compiled operations a built-in model supports.

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

Strategies own optimization, filtering coordination, and result
construction. Copulas own model metadata, transforms, and sampling behavior.
Native adapters own calls into the mandatory C++ extension.

| Layer | Main responsibility |
|-------|---------------------|
| Copula class | Model identity, parameter domain, sampling |
| Capability metadata | Explicit strategy support |
| Strategy | Optimization and fit-result construction |
| Native evaluator | Density, likelihood, gradient, filtering, multivariate conditional linear algebra |
| Python coordination | RNG and fixed draws, Jacobi, MC/EIS, GoF, persistence |

Sequential unconditional sampling for R-vines with fitted GAS edges is also a
native operation. Python builds the flat edge/topology plan and generates the
same fixed draws as the generic path; the C++ kernel owns the row recursion and
causal GAS state updates. Unsupported custom stateful strategies remain on the
generic Python path.

## Native Array Boundary

Multivariate Gaussian and Student conditional kernels consume read-only views
of their numeric inputs. Already C-contiguous NumPy `float64` arrays are passed
without an additional C++ vector copy and stay alive throughout the synchronous
native call, including work performed with the GIL released. Non-contiguous
arrays and other dtypes are converted by pybind11's `forcecast` path.

Zero-copy does not remove validation: correlations, latent conditioning values,
degrees of freedom, normal draws, and chi-square draws are checked for finite
values before native computation starts. The small integer `given_indices`
array is still copied into an owning vector.

## Numerical Safety Boundaries

Numerical boundaries are named by purpose rather than represented by a
generic `eps`. Python values live in `pyscarcopula._constants`; compiled
pair-copula values live in `scar/numerical_constants.hpp`.

- `PSEUDO_OBS_EPS` protects pseudo-observations passed to quantiles and
  h-functions.
- `H_FUNCTION_EPS` bounds numerical h/inverse-h outputs.
- `ROSENBLATT_OUTPUT_EPS` protects final GoF normal quantiles.
- `CONDITIONAL_SAMPLE_EPS` applies only to newly sampled free coordinates.
- `PDF_FLOOR` protects density and logarithm arguments.

The h-function and Rosenblatt boundaries use the same numeric value, but they
are named separately because they protect different calculations. Vine code
uses the shared pseudo-observation helper; it does not define local `_EPS`
constants.

## Custom Python Copulas

Custom Python copulas can be paired with custom strategies, sampling,
diagnostics, and other Python workflows. This does not add a family to the
compiled support matrix.

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
