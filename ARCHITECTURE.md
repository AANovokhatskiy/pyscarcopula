# Architecture

## Module Map

```text
pyscarcopula/
|-- __init__.py              # Public re-exports and native extension loading
|-- api.py                   # Top-level fit/predict/sample helpers
|-- _types.py                # Results and numerical configuration
|-- _native/                 # Mandatory extension facade and support policy
|   |-- pair.py, multivariate.py, static.py
|   |-- gas.py, scar_ou.py, jacobi.py, model_policy.py, statistics.py, vine.py
|   `-- registry.py, errors.py, threads.py, _extension.py
|-- io.py                    # JSON model persistence
|-- stattests.py             # Goodness-of-fit orchestration
|-- copula/
|   |-- base.py              # CopulaBase and BivariateCopula
|   |-- gumbel.py, frank.py, joe.py, clayton.py
|   |-- elliptical.py        # Bivariate Gaussian copula
|   `-- multivariate/
|       |-- base.py          # MultivariateCopula
|       |-- gaussian.py, student.py
|       `-- equicorr.py, stochastic_student.py
|-- strategy/
|   |-- _base.py             # Strategy registry and capability validation
|   |-- mle.py, gas.py, scar_tm.py
|   `-- scar_jacobi.py
|-- numerical/               # Public numerical configuration helpers
|   `-- _arrays.py, _scar_ou_config.py, _transition_methods.py
|-- vine/
|   |-- vine.py               # Canonical generic VineCopula runtime
|   |-- rvine.py              # RVineCopula compatibility module alias
|   |-- _structure.py         # RVineMatrix and C/D structure factories
|   |-- _vine_fit.py          # Shared fixed-structure edge fitting
|   |-- _rvine_sampling_plan.py # Canonical unconditional traversal plan
|   `-- _rvine_*.py, _selection.py
`-- contrib/                 # Marginals and risk analytics
```

This file is the repository source map and dependency contract. The
published [Developer Architecture](docs/guide/architecture.md) explains
runtime ownership and the development workflow. Numerical algorithms are
documented in the [numerical reference](docs/guide/numerical-backends.md).

## Copula Hierarchy

`CopulaBase` branches into pair-oriented `BivariateCopula` and row-oriented
`MultivariateCopula`. See the [class hierarchy](docs/guide/architecture.md#class-hierarchy)
for concrete models. Vine topology belongs to `vine/`, not to the pair classes.

## Capabilities And Strategies

The dependency flow is:

```text
api.py -> strategy/ -> _native facade -> C++ computational APIs
                   -> copula metadata and state
vine/vine.py -> structure selection -> _vine_fit.py -> pair strategy
stattests.py -> native Rosenblatt/filter operations -> GoF reporting
```

Exact-type native descriptors and operation-level capability queries decide
support. A custom subclass cannot opt into numerical execution merely by
inheriting methods. Strategies own optimization and result construction;
models own their metadata and fitted state. Numerical policy values, domain
bounds, penalties, and transforms have native owners.

## Static Multivariate Correlation Policy

`CorrelationPolicy` separates the public MLE method label from `corr_mode`
and `corr_estimator`. See [correlation estimation](docs/guide/mathematical-contracts.md#static-elliptical-correlation-estimation)
for fixed, shrinkage, cholesky, factor, and parameter-count contracts.
The Python adapter carries raw optimizer coordinates; native evaluators own
correlation parameterization, identifiability, and gradient pullbacks.

## Native Boundary

`pyscarcopula/_cpp/build_support/sources.py` is the canonical source manifest:
`SCAR_COMPUTE_SOURCES` contains Python-free computation and
`PYTHON_BINDING_SOURCES` contains pybind adapters. `setup.py` combines them
into `pyscarcopula._native._scar_cpp`. Only `_native/_extension.py` imports
that binary in production; all other callers use `_native` domain facades.

Native source ownership (relative to `pyscarcopula/_cpp`):

```text
include/scar/copula/spec.hpp                 generic metadata boundary
include/scar/copula/model_storage.hpp        typed model-storage variant
src/copula/pair/                             pair families and dispatch
src/copula/multivariate/correlation/         dense and factor operators
src/copula/multivariate/gaussian/            Gaussian density/conditional
src/copula/multivariate/equicorrelation/     equicorrelation model/kernel
src/copula/multivariate/student/             Student distribution, quantile,
                                             PPF cache, density, conditional,
                                             factor grid, and Rosenblatt code
```

The foundation under `include/scar/core` and `include/scar/math` owns views,
checked sizes, thread validation, transforms, common probability functions,
and typed status vocabulary. It depends only on foundation headers. Model
state belongs to typed model storage; universal metadata does not own
model-specific mutable workspaces.

The dependency rules enforced by `tools/check_cpp_architecture.py` are:

- Every C++ source and public header belongs to one logical target; includes
  follow the declared downward target graph. Header and domain cycles fail.
- Pair headers do not depend on Student, factor, or SCAR-OU implementations;
  Gaussian headers do not depend on Student implementation headers.
- Dense and factor correlation have separate contracts. Factor kernels must
  not acquire an implicit dense `d*d` workspace.
- Each binder includes its own computational API. Shared `bindings/array.*`
  owns model-neutral NumPy conversion and lifetime helpers;
  `bindings/module.hpp` declares registration entry points. Domain DTO
  conversion remains in its owning binder.
- Binders obtain buffer metadata before releasing the GIL, keep owning arrays
  alive through the synchronous call, then reacquire the GIL before Python
  serialization. They do not implement filtering or result-dependent policy.
- `_native/errors.py` translates typed statuses centrally. Only structured
  numerical failure can select a native optimizer penalty. Unsupported
  operations and unexpected failures propagate; a status-OK non-finite
  objective raises `FloatingPointError`.
- Foundation CDF, beta/gamma, and transform formulas have one owner.
  Production Python modules do not retain duplicated numerical function bodies
  or unused numerical alternatives as importable reference implementations.

Python owns optimization coordination, raw RNG draws, request assembly,
GoF reporting, and persistence. C++ owns numerical trajectories, quantiles,
densities, conditionals, filtering, gradients, and vine plan execution.
Test oracles are independent test implementations and never production fallbacks.

### Build and validation entry points

```bash
python tools/check_cpp_architecture.py
python tools/build_cpp_tests.py
```

The standalone build compiles every computational translation unit and every
public header in isolation, then links and runs the model suites without
Python, NumPy, or pybind11 headers/libraries. `--sanitize address-undefined`
and `--sanitize thread` instrument that executable separately from the extension.

`build_support/build_parallel.py` owns compilation parallelism. Both builds
default to one job; `PYSCA_CPP_BUILD_JOBS=N` opts in. CLI overrides are
`build_ext --parallel N` and `build_cpp_tests.py --build-jobs N`. Linking is
sequential. Compiler provisioning belongs to the environment; builds do not
modify `PATH`. Runtime thread policy is a separate contract.

## State And Persistence

Model methods store fit results and owned training snapshots. Fit transactions
restore previous state on exceptions; callers inspect returned candidates'
success flags. Constructor policy survives refits. Native plans/evaluators are
transient and rebuilt after loading.

`io.py` uses an explicit model/configuration/result registry for JSON model
persistence; payload class paths cannot select arbitrary imports. Standalone
factor correlation additionally supports NPZ and memory-mapped storage. See
[Persistence](docs/api/persistence.md) for supported formats and examples.

`RVineMatrix` owns public vine structure, with a separate natural-order
traversal representation. `RVineCopula` is the same runtime type as `VineCopula`.

## BLAS Thread Policy

Import does not mutate BLAS environment variables. Native runtime scheduling,
process ownership, locks, and reproducibility are documented centrally in
[CPU Parallelism](docs/guide/parallelism.md).
