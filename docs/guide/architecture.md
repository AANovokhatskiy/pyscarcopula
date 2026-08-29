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

## Native capabilities

For exact built-in model types, the opaque C++ descriptor and operation-level
capability query are the only support contract. The strategy layer validates
named native requirements before optimization. A multivariate
model is therefore not accepted by a pair-only strategy merely because it has
similarly named methods.

## Strategy Ownership

Strategies own optimization, filtering coordination, and result
construction. Copulas own model metadata, transforms, and sampling behavior.
The `pyscarcopula._native` facade owns loading the mandatory C++ extension,
typed model descriptors, capability decisions, status translation, and thread
validation. Production callers use this facade directly; retired numerical
adapter names are not dispatch surfaces.

Unused numerical implementations are not shipped as importable references.
The former `_utils.linear_least_squares` kernel and
`numerical.gof_blocks` streaming state helpers are retired; maintained GoF
paths call native evaluators directly. Reachability is established from
production imports, not inferred from an absence of observed runtime calls.

The binary implementation lives at `pyscarcopula._native._scar_cpp`. Only the
facade loader imports that raw module in production code. The former
`pyscarcopula._scar_cpp` import path is removed and has no compatibility alias.

| Layer | Main responsibility |
|-------|---------------------|
| Copula class | Model identity, parameter domain, sampling |
| Native descriptor registry | Explicit strategy support |
| Strategy | Optimization and fit-result construction |
| Native evaluator | Density, likelihood, gradient, filtering, multivariate conditional linear algebra |
| Python coordination | Fit/RNG orchestration, fixed draws, GoF reporting, persistence |

## Native Thread Runtime

Eligible multivariate kernels use one lazily created C++17 thread pool per
process. Calls divide independent rows or trajectories into stable blocks;
GAS and SCAR time recursions remain sequential. `n_threads=1` takes a direct
fast path without creating or querying the pool.

The pool records its owning process ID. A spawned or forked child creates its
own runtime when it first performs explicitly parallel work and never reuses
the parent's workers. Nested native dispatch from a worker falls back to a
local sequential call, preventing starvation and deadlock.

Each submitted batch captures the calling thread's C floating-point
environment and applies it before numerical work starts on a pool worker.
This prevents operating-system or runtime-specific worker defaults from
changing the last bits of otherwise identical serial and parallel results.

Model mutation is protected by a per-instance re-entrant lock. Prepared SCAR
evaluators protect mutable workspace with a native mutex. Concurrent work
should normally use independent models/evaluators; sharing one prepared
evaluator is safe but serializes its objective calls.

The default thread count is an absolute `1`. Environment variables are not
consulted. See [CPU Parallelism](parallelism.md) for the public contract.

R-vine execution uses one shared flat plan and family-operation layer. Native
entry points cover supported static unconditional and conditional sampling,
row log-density, coordinate-update MCMC, and static or dynamic Rosenblatt
traversals. Python owns topology construction, exact-type capability dispatch,
random draws, request assembly, and bootstrap orchestration. Unsupported
custom subclasses fail before any Python model formula can run; test-only
candidate traversal harnesses are never part of production dispatch.

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

Repository validation compiles and links the complete computational source
manifest without Python headers or libraries, compiles every public C++ header
as a self-contained unit, and runs focused C++ model suites. The architecture
gate validates the complete logical target dependency graph and rejects
domain cycles. It additionally rejects private copies of foundation CDF,
incomplete beta/gamma, softplus, inverse-softplus, and logistic formulas, and
exact Python function-body clones inside or across shipped modules. ASan/UBSan
and TSan instrument the standalone computational
executable separately from the Python extension; cross-platform wheel,
accuracy, configuration, and pinned-runner performance workflows use the same
canonical computational source manifest. Every supported wheel executes the
frozen numerical golden comparison after installation, and the pinned-runner
performance workflow is triggered automatically as well as on demand.

Optimizer adapters translate only structured native numerical failures into a
penalty through the C++ model-policy API. Invalid or unsupported statuses and
unexpected exceptions propagate through the centralized exception policy. A
status-OK non-finite objective raises `FloatingPointError`; Python never
constructs a replacement objective or zero gradient. Jacobi domain rejection
uses native validation and the same native optimizer policy.

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

## Public Imports

Base classes are available at the package top level:

```python
from pyscarcopula import (
    BivariateCopula,
    CopulaBase,
    MultivariateCopula,
)
```

Multivariate models can be imported either from `pyscarcopula` or from
`pyscarcopula.copula.multivariate`.
