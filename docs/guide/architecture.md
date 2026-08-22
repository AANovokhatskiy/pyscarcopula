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

The retained `pyscarcopula.numerical.TMGrid` class is a manual low-level
NumPy/SciPy reference implementation. Production OU likelihood, prediction,
smoothing, and GoF paths do not call it. Keeping the reference grid independent
from the native evaluator provides an implementation oracle for parity tests;
it is not a compatibility wrapper or a deprecated alias.

## Native Thread Runtime

Eligible multivariate kernels use one lazily created C++17 thread pool per
process. Calls divide independent rows or trajectories into stable blocks;
GAS and SCAR time recursions remain sequential. `n_threads=1` takes a direct
fast path without creating or querying the pool.

The pool records its owning process ID. A spawned or forked child creates its
own runtime when it first performs explicitly parallel work and never reuses
the parent's workers. Nested native dispatch from a worker falls back to a
local sequential call, preventing starvation and deadlock.

Model mutation is protected by a per-instance re-entrant lock. Prepared SCAR
evaluators protect mutable workspace with a native mutex. Concurrent work
should normally use independent models/evaluators; sharing one prepared
evaluator is safe but serializes its objective calls.

The default thread count is an absolute `1`. Environment variables are not
consulted. See [CPU Parallelism](parallelism.md) for the public contract.

R-vine execution uses one shared flat plan and family-operation layer. Native
entry points cover supported static unconditional and conditional sampling,
row log-density, coordinate-update MCMC, and static Rosenblatt transforms.
Sequential unconditional sampling with fitted GAS edges reuses the same
topology and family semantics while retaining its causal state-update driver.
Python owns topology construction, capability dispatch, random draws,
parameter trajectories, bootstrap orchestration, and the preserved reference
executor. Unsupported custom copulas and stateful operations remain on that
Python path.

Compiled R-vine plans and immutable edge specifications are transient fitted
model state. Their cache keys include the structure, exact copula type,
rotation, transform metadata, fitted-result identity, parameter storage kind,
and complete plan signature. Fitting or semantic mutation invalidates the
cache, and persistence always reconstructs an empty cache. Request-owned
NumPy buffers are never retained.

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

## Native Linear Algebra

Dense matrix-vector, Cholesky, SPD-solve, and triangular kernels are provided
by an internal C++17 layer. The extension does not depend on Eigen, BLAS, or
OpenMP and therefore does not introduce a second, hidden thread pool. Scalar
and portable compiler-vectorizable reduction backends are retained for
cross-backend correctness tests. Runtime paths use the portable reduction
only for kernels whose end-to-end benchmarks pass the recorded performance
gate.

## Numerical Safety Boundaries

Numerical boundaries are named by purpose rather than represented by a
generic `eps`. Python values live in `pyscarcopula._constants`; compiled
pair-copula values live in `scar/numerical_constants.hpp`.

- `PSEUDO_OBS_EPS` protects pseudo-observations passed to quantiles and
  h-functions.
- `H_FUNCTION_EPS` bounds internal numerical h/inverse-h outputs without
  removing tail probabilities needed by deeper vine trees.
- `ROSENBLATT_OUTPUT_EPS` protects final GoF normal quantiles.
- `CONDITIONAL_SAMPLE_EPS` applies only to newly sampled free coordinates.
- `PDF_FLOOR` protects density and logarithm arguments.

The internal h-function and pseudo-observation boundaries are `1e-10`.
The final Rosenblatt boundary remains `1e-6` because it protects GoF normal
quantiles and must not be reused inside vine recursion. Vine code uses the
shared pseudo-observation helper; it does not define local `_EPS` constants.

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
