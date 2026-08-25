# Architecture

## Module Map

```text
pyscarcopula/
|-- __init__.py              # Public re-exports and BLAS thread policy
|-- api.py                   # Top-level fit/predict/sample helpers
|-- _types.py                # Results and numerical configuration
|-- _native/                 # Stable extension facade and support policy
|-- io.py                    # JSON model persistence
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
|   `-- scar_jacobi.py
|-- numerical/
|   |-- copula_native.py, multivariate_native.py
|   |-- static_likelihood.py, gas_filter.py
|   |-- _cpp_scar_ou.py, _cpp_gas.py, _cpp_gas_rvine.py
|   `-- jacobi_tm.py         # Retained Python Jacobi orchestration
|-- vine/
|   |-- vine.py               # Canonical generic VineCopula runtime
|   |-- rvine.py              # RVineCopula compatibility module alias
|   |-- cvine.py              # Legacy CVineCopula implementation
|   |-- _structure.py         # RVineMatrix and C/D structure factories
|   |-- _vine_fit.py          # Shared fixed-structure edge fitting
|   |-- _rvine_sampling_plan.py # Canonical unconditional traversal plan
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

Class hierarchy answers what a model is. For exact built-in model types, the
opaque C++ `TypedModelDescriptor` and `query_capability` operation matrix are
the authoritative support contract. `CopulaCapabilities` is a temporary public
projection of that contract during the Stage 8 migration:

- `supports_pair_ops`
- `supports_native_mle`
- `supports_gas`
- `supports_scar_ou`
- `supports_latent_grid`
- `supports_conditional_sampling`
- `has_dynamic_scalar_parameter`

The strategy registry in `strategy/_base.py` checks named
`StrategyRequirements` through the native query before fitting. Custom Python
subclasses remain on the temporary legacy adapter and do not acquire native
support through inheritance. Strategy classes own optimization and result
construction; copula classes own model metadata, parameter transforms, and
sampling.

The main dependency flow is:

```text
api.py -> strategy/ -> _native facade -> C++ extension
                    -> transitional numerical adapters
                    -> copula model metadata
vine/vine.py -> structure selection or fixed RVineMatrix
             -> _vine_fit.py + bivariate copula contract
stattests/ -> fitted strategy outputs + retained GoF orchestration
```

## Static Multivariate Correlation Policy

`GaussianCopula` and `StudentCopula` use `method="mle"` as the public label
for a static fit. The label does not imply that every correlation parameter
is part of one joint optimizer vector. `CorrelationPolicy` records the actual
procedure independently through canonical `corr_mode` and `corr_estimator`
values.

- `fixed` retains the fast compatibility path. A constructor-supplied `R` is
  held fixed; without `R`, Gaussian uses normal-score correlation and Student
  uses a Kendall plug-in correlation.
- `shrinkage` jointly optimizes one correlation weight.
- `cholesky` jointly optimizes all `d*(d-1)/2` dense correlation parameters
  and is guarded for small dimensions.
- `factor` stores compact loadings. Gaussian and two-stage Student fits use a
  plug-in loading estimate; static Student additionally supports identified
  joint loading optimization.

`corr_estimator` distinguishes `supplied`, `gaussian_score`,
`kendall_plugin`, `joint_mle`, `factor_two_stage`, and `factor_joint`.
`corr_plugin_n_params` and `corr_n_params` remain separate, while
`corr_effective_n_params` is the count consumed by AIC/BIC. Worker
reconstruction copies constructor policy rather than fitted mutable state;
JSON persistence retains fitted raw parameters and compact factor state.

## Native Boundary

The pybind11 C++ extension is mandatory. `_native/_extension.py` is the sole
owner of importing its current binary location; `_native` domain modules expose
the stable facade while legacy numerical adapters remain as Stage 8.1
compatibility shims. Built-in point operations, static
likelihoods, GAS filtering, multivariate conditional linear algebra,
sequential GAS R-vine sampling, dense Student Rosenblatt transforms, and
SCAR-TM-OU likelihood/gradient/forward operations have one production
implementation in C++.

Pair-copula unconditional sampling keeps RNG ownership in Python but passes
the complete fixed-uniform draw matrix through the native facade. C++ applies
the transposed conditional orientation, including rotated-family semantics;
built-in pair subclasses contain no frailty or conditional-inversion sampling
formulae. Point evaluation and sampling use the same family `h_inverse`
implementation and accuracy contract; there is no sampling-specific inverse.

Static dense/factor Gaussian and Student runtime sampling/GoF paths likewise
keep only RNG draw generation and output assembly in Python. Fixed normal and
chi-square draws, unconditional and conditional latent transforms, correlation
algebra, marginal CDFs, dense/factor Rosenblatt transforms, and the radial GoF
summary are owned by `copula::multivariate`. SciPy remains at the facade only
for the final one-sample Cramér-von Mises statistic and p-value.
Equicorrelation unconditional sampling, conditional sampling, and Rosenblatt
transforms use specialized scalar-or-row C++ kernels without materializing a
dense correlation matrix. Only dynamic state/filter ownership remains in the
Stage 8.3 migration boundary.

Static dense-correlation preprocessing uses the dependency-free C++17 Jacobi
eigensolver as its canonical arithmetic path. Python performs boundary shape
and finiteness checks and constructs the public diagnostics DTO, but does not
run an eigendecomposition or reconstruct the projected matrix. The numerical
baseline was explicitly refrozen when this owner changed; LAPACK, OpenBLAS,
and other external linear-algebra runtimes are not build dependencies.

The extension build has one canonical source manifest at
`pyscarcopula/_cpp/build_support/sources.py`. `SCAR_COMPUTE_SOURCES` contains
only Python-free computational translation units;
`PYTHON_BINDING_SOURCES` contains the pybind adapter. `setup.py` combines both
lists for `_scar_cpp` without creating or shipping a separate C++ library.

The adapter has no umbrella binding header. `bindings/module.hpp` declares only
the registration entry points, while `bindings/array.hpp` and `array.cpp` own
the shared, model-neutral NumPy array/view conversions and their synchronous
lifetime helper. Module metadata, constants, and thrown-exception translation
remain in `bindings/common.cpp`. Enums, model-specific result serialization,
and other domain DTO conversion live in the binder that owns that domain. Each
binder imports its own computational API explicitly; in particular, Student
headers are absent from pair-only, GAS, R-vine, parallel, and SCAR-OU binders.
The architecture checker enforces this include matrix and rejects a recreated
`bindings/common.hpp` or model-result conversion in a shared header. It also
rejects result-dependent status policy, NumPy buffer access while the GIL is
released, and SCAR-OU grid/filter orchestration inside a binder.

Native results are serialized mechanically, including `Status` and failure
context. `_native/errors.py` centrally decides whether a non-OK status becomes
`ValueError`, `NativeUnsupported`, `FloatingPointError`, or `NativeError`;
the legacy `Cpp*` names are aliases during migration. Adapters contribute only
operation and location labels. Binders do not throw based on returned failure
indices. Raw OU emission filtering is a public computational
operation, `filter_ou_grid_emissions`, returning `OuGridFilterResult`; the
SCAR-OU binder only validates the buffer shape/lifetime, invokes that API
without the GIL, and serializes its result.

Native copula code is organized by model ownership rather than by a shared
horizontal implementation file:

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

`CopulaSpec` owns only universal metadata plus a `TypedModelStorage` whose
variant alternative is synchronized with family and correlation kind. Pair,
dense Gaussian, factor Gaussian, equicorrelation, dense Student, and factor
Student state therefore have distinct storage types. Compatibility accessors
remain at binding and migration boundaries, but computational kernels resolve
the typed alternative once before entering a hot loop.

Dense and factor correlation are separate native contracts. Dense kernels may
own inverse-Cholesky and log-determinant state; factor kernels own compact
loadings/operators and must not acquire an implicit dense `d*d` workspace.
Pair headers depend on pair contracts and `spec.hpp`, not on Student, factor,
or SCAR-OU headers. Gaussian multivariate headers likewise do not depend on
Student implementation headers. `tools/check_cpp_architecture.py` enforces
these dependency and placement rules.

The computational boundary is independently buildable as a C++17 executable:

```bash
python tools/check_cpp_architecture.py
python tools/build_cpp_tests.py
```

The second command uses setuptools' configured compiler abstraction. It
compiles every computational source and every `scar/*.hpp` header in
isolation, links `tests/cpp/compute_smoke.cpp`, and runs the executable without
Python, NumPy, or pybind11 include paths or libraries. Compiler provisioning is
the responsibility of the local environment or CI runner; `setup.py` does not
modify `PATH`.

The native foundation has explicit, model-independent owners under
`include/scar/core`, `include/scar/math`, and the common `include/scar/copula`
wrappers. It owns C++17 `Span`/`DoubleView`/`MatrixView`, checked shape and byte
arithmetic, thread-count validation and worker limiting, normal CDF/quantile,
parameter transforms, rotations, and the shared `Status`/`Result`/
`FailureContext` vocabulary. Foundation math includes only foundation headers;
the architecture checker rejects reverse dependencies on model or workflow
headers. `CopulaSpec` remains a temporary compatibility DTO, but converts to a
`TypedModelDescriptor`; `expected_dimension()` belongs to that typed
descriptor rather than to foundation or universal metadata. Model-specific
mutable state belongs to `TypedModelStorage`, not to common fields. Production
Rosenblatt shape validation uses `Result<std::size_t>` and preserves typed
failure context. Migration of the remaining domain result DTOs is deferred to
the error-model cleanup stage. Kernel-specific work thresholds and
parallel-axis policies stay in the owning kernels.

Python remains responsible for:

- optimizer orchestration and result construction;
- correlation parameterization and chain rules around native evaluators;
- RNG and generation of fixed draws used by native conditional sampling;
- Jacobi filtering orchestration;
- goodness-of-fit and contribution analytics.

For dense static and GAS Student GoF, Python owns dispatch and the `df`
trajectory. In the differential-validated native domain (`df >= 0.1`, a
symmetric unit-diagonal SPD correlation with condition number at most `1e4`),
it transfers the complete observation matrix and either a scalar or per-row
`df` path in one call. C++ factors the fixed correlation once, executes all
sequential conditionals without the GIL, and may parallelize independent rows.
Inputs outside that domain are rejected by a pre-call capability gate: `auto`
preserves the SciPy oracle's low-`df`, invalid-input, and ill-conditioned-matrix
semantics, while `native_strict` reports `CppUnsupported`. Native runtime
errors are not retried. Factor-correlation and latent SCAR Rosenblatt paths
keep their specialized native implementations.

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
validation pass before the GIL is released. Binding lambdas take the owning
`py::array_t` objects by value; buffer metadata and native views are created
before `py::gil_scoped_release`, and serialization back to Python starts only
after that scope ends. The same lifetime order is checker-covered for the
shared observation helper and direct SCAR-OU view paths.

Static and prepared SCAR-OU equicorrelation evaluators cache per-row `sum(z)`
and `sum(z^2)` statistics for their owned observation snapshots. Repeated
objective, gradient, and forward calls therefore reuse both the per-row normal
scores and the statistics across every latent-grid node.

Unconditional generic R-vine sampling compiles the natural-order matrix,
semantic trees, and edge map into one model-independent
`RVineTraversalPlan`. The Python reference sampler and the native sequential
GAS sampler execute that same plan. Model-specific parameter generation and
state updates remain in their strategy executors; the plan owns only topology,
node dependencies, edge orientation, and operation order.

Arbitrary-given R-vine MCMC also compiles, once per density plan, the
topologically ordered operations and nodes affected by each original
coordinate. The native incremental executor caches node values and individual
edge log-density contributions per chain, recomputes only that closure for a
proposal, and then sums all edge contributions in their original order. A
rejected proposal never changes the accepted cache. Its accepted/proposal
caches are row-chunked under the same preflight memory budget as states,
log-densities, and replay draws. The adapter selects the incremental path only
for structurally profitable closures that fit the budget. Otherwise it uses
the preserved full-recompute oracle only when that driver's complete state,
proposal, density-workspace, and draw footprint also fits; if neither path
fits, preflight fails before consuming RNG state or allocating MCMC buffers.
Algorithm and workspace measurements are internal native diagnostics, while
the public MCMC diagnostics schema remains unchanged. Single-chain calls
retain full recomputation because the incremental cache setup does not
amortize there.

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

Persistence uses a single JSON representation. The loader restores the same
canonical class paths and state layout written by the current package.

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
