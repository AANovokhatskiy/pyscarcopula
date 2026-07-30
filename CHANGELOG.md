# Changelog

## 0.20.0 - Unreleased

Version: `0.19.1` -> `0.20.0`

- Centralize process-count validation, native-thread policy, independent
  random streams, and worker model reconstruction in a shared internal
  parallel runtime used by independent fitting and rolling risk metrics.
- Add an adapter-based, process-parallel parametric-bootstrap GoF runner for
  bivariate copulas, static `GaussianCopula`/`StudentCopula`, and dynamic
  `EquicorrGaussianCopula`/`StochasticStudentCopula` MLE, GAS, and SCAR-TM-OU
  fits. Dense, shrinkage, Cholesky, and supported factor correlation policies
  are preserved. The `n_jobs` API keeps a sequential default and uses
  per-replication `SeedSequence` streams, deterministic results across worker
  counts, isolated model ownership, and protection against nested
  native-thread oversubscription.
- Reject non-boolean bootstrap switches, preserve fitted SCAR grid settings
  during bootstrap simulation and refitting, and evaluate factor stochastic
  Student Rosenblatt transforms without materializing a dense correlation
  matrix.

## 0.19.1 - 2026-07-28

Version: `0.19.0` -> `0.19.1`

- Route stochastic Student prediction through the fitted strategy so MLE,
  GAS, and SCAR models share the documented `current`/`next` timing,
  prediction configuration, conditioning, and memory-budget behavior.
- Correct the SCAR-TM-JACOBI analytical-gradient contract and persist every
  numerical option that affects likelihood, prediction, sampling, or
  admissibility when reconstructing a strategy from a fit result.
- Add likelihood-consistent unconditional SCAR-TM-JACOBI grid-path sampling,
  including deterministic seeded behavior, bounded allocation checks,
  persistence, and vine integration.
- Add an opt-in sparse moving-grid Jacobi transition backend for likelihood,
  filtering, prediction, state distributions, and exact fixed-seed grid
  sampling equivalence. Add deterministic full-horizon diagnostics and an
  experimental end-to-end MH stationarity correction; dense/auto defaults
  remain unchanged.
- Extend sparse Jacobi transitions to `local_fixed`, including shared sparse
  derivative support, sequential analytical-gradient filtering, and
  fixed-seed sampling equivalence with the dense backend.
- Add an experimental deterministic adaptive-order selector for uncorrected
  sparse local Jacobi transitions. Strategy integration freezes the selected
  order before optimization, persists it, and reports initial and fitted
  full-horizon gates without changing sampling semantics.
- Add an experimental sparse IPFP stationarity correction and deterministic
  none/MH/IPFP/higher-order comparison. Infeasible sparse support is reported
  explicitly without hidden diagonal regularization.
- Extend the Jacobi validation tool with a predefined adaptive-order
  parameter/horizon calibration matrix.
- Add an experimental `lamperti_euler` unconditional sampler for
  SCAR-TM-JACOBI with strict options, substeps, constant-time reflection,
  boundary-intervention diagnostics, memory preflight, persistence, and vine
  adapter support. The likelihood-consistent `tm_grid` sampler remains the
  default.
- Add a strictly sequential Numba Lamperti kernel with chunked NumPy
  innovations, a Python reference engine, pathwise/chunk-size equivalence
  checks, boundary-singular diagnostics, and an opt-in million-update
  performance gate. `parallel=True` is permanently forbidden for trajectory
  kernels.
- Add an ensemble comparison tool for Jacobi stationary Beta error,
  conditional first moments, KS/TV distance, boundary interventions, and
  runtime. Exact Wright--Fisher sampling remains out of scope.
- Prevent a vector Jacobi `alpha0` from being forwarded to the scalar MLE used
  for fixed-family C-/R-vine edge selection.
- Validate GoF inputs consistently across bivariate, Gaussian, Student,
  equicorrelation, stochastic Student, C-vine, and R-vine entry points.
  Reject non-finite, malformed, undersized, or out-of-domain data and invalid
  bootstrap counts before transforms, refits, or large allocations.
- Reorganize the numerical-backend documentation, tighten public API and
  mathematical-contract examples, and validate documented imports,
  signatures, persistence behavior, and model-selection guidance.

## 0.19.0 - 2026-07-26

Version: `0.18.0` -> `0.19.0`

Commit: `966758b`  
Merge PR: #43 (`2d6ebfc`, 2026-07-26)

- Removes temporary VineCopula refactoring guardrails after their permanent
  contracts were absorbed by the regular vine API, structure, fitting,
  persistence, and compatibility suites. Completes public type annotations
  and docstrings for the new prepared equicorrelation, factor-correlation,
  factor-Student, and independent-fit APIs, and aligns
  `RVineTraversalPlan` with the project's R-vine acronym style.
- Introduces `VineCopula` as the generic auto/fixed regular-vine runtime while
  retaining `RVineCopula` as a compatibility name and `CVineCopula` as a
  supported legacy implementation. No runtime deprecation warning is emitted
  for `CVineCopula`; new C-vine code should use `VineCopula.cvine(...)`.
- Adds fixed C-vine/D-vine factories, arbitrary validated `RVineMatrix`
  structures, shared fixed-edge fitting, type-aware GoF, and canonical JSON
  persistence. Documentation and notebooks now use
  decoded tree edges for structure comparison instead of raw matrix
  conventions.
- Adds reusable, copula-independent `FactorCorrelation` and
  `PreparedFactorCorrelation` objects for correlations of the form
  `R = D + B B.T`, with unit diagonal and explicit uniqueness bounds.
- Adds a dependency-free native Woodbury backend for factor-correlation
  matrix products, solves, quadratic forms, log determinants, and normal
  sampling. Row kernels are deterministic across thread counts, default to
  one thread, and a prepared operator is safe for concurrent read-only calls.
- Adds the independent `FactorStudentEvaluator` adapter with static Student
  copula row log densities, analytical degrees-of-freedom derivatives,
  scalar or row-specific `df`, optimizer-ready likelihood reduction, and
  deterministic native row parallelism. Dense and factor paths share the
  same Student normalization and derivative implementation.
- Adds tiled factor-Student emission over a degrees-of-freedom grid without
  an `O(T*K*d)` quantile cache. Fixed dimension tiles are reduced in a
  deterministic order, parallel work selects either independent grid cells
  or dimension tiles, and row-batch APIs enforce a peak-memory budget.
- Adds bounded factor-normal sampling, explicit guarded dense
  materialization, compact `.npz` persistence, and read-only mmap
  persistence. Storage remains `O(d*k + k^2)`.
- Adds the `StochasticStudentCopula(corr_mode="factor")` model
  adapter with supplied or deterministic two-stage randomized-SVD loadings,
  compact diagnostics and model persistence, guarded explicit dense
  materialization, static row likelihood, and tiled latent-grid evaluation.
- Integrates factor Student correlation with static MLE, native GAS,
  SCAR-TM-OU matrix/local/spectral objectives and gradients, and native
  SCAR-MC trajectory likelihood. A shared immutable operator in `CopulaSpec`
  avoids copying loadings or constructing dense Cholesky factors; SCAR
  emission directly reuses the tiled factor grid kernel.
- Adds unconditional, bounded-batch, fitted-model, predictive, and exact
  conditional sampling for factor Student models. Conditional generation
  factorizes only a `k*k` system and never constructs a dense Schur
  complement. Fixed-seed results are exact across native thread counts,
  sampling defaults literally to `n_threads=1`, and memory budgets can reject
  oversized monolithic or batch requests before their main allocations.
- Reuses the independent factor-correlation operator in
  `GaussianCopula(corr_mode="factor")`. Supplied or tiled two-stage loadings
  feed native matrix-free likelihood, compact MLE results, deterministic
  sampling, bounded batches, exact conditional generation, rolling-window
  reconstruction, persistence, and an `O(T*k + k^2)` Rosenblatt transform.
  No Gaussian factor path constructs a dense correlation or Schur complement.
- Adds guarded joint factor estimation for static Student MLE. A native
  matrix-free kernel returns analytical gradients for common `df` and every
  loading, while pivoted lower-triangular positive-diagonal coordinates remove
  rotational degrees of freedom. Fixed reduction blocks make gradients exact
  across thread counts. The optimizer enforces parameter-count, uniqueness,
  Woodbury-condition, regularization, and terminal-gradient convergence
  gates. Dynamic GAS/SCAR joint loading estimation remains explicitly
  unsupported pending derivatives through their sequential filters.
- Adds deterministic, tiled CPU preparation of equicorrelation Gaussian
  sufficient statistics from ndarray, memmap, or streamed observation blocks.
- Adds immutable `EquicorrPreparedData` with compact `.npz` and read-only
  memory-mapped persistence. The prepared representation stores two `O(T)`
  vectors and never retains the original `T * d` observation matrix.
- Lets static MLE, row/grid evaluation, GAS, and SCAR-TM-OU consume prepared
  Equicorr statistics directly. Prepared MLE avoids materializing a dense
  correlation matrix in its result.
- Adds bounded row-batch grid output with a pre-allocation memory budget and
  structural `O(n*d)` Equicorr sampling over the full admissible correlation
  interval, including negative correlation.
- Adds fitted `sample_batches` and `predict_batches` for Equicorr MLE, GAS,
  and SCAR-TM-OU while preserving per-row GAS recursion and SCAR path or
  posterior semantics. Monolithic sampling APIs can reject oversized output
  before allocation.
- Adds opt-in `d=10^4`, `10^5`, and `10^6` preparation benchmark gates with
  exact cross-thread result checks and an enforceable four-thread efficiency
  target.
- Adds opt-in native CPU threading for stochastic Student and
  equicorrelation emission/gradient grids, static multivariate row
  likelihoods and MLE, Gaussian/Student conditional sampling, and historical
  SCAR Monte Carlo trajectory evaluation.
- Introduces a reusable process-local C++17 thread runtime with deterministic
  block partitioning, sequential nested-call fallback, exception propagation,
  process ownership checks, and a no-pool `n_threads=1` fast path.
- Makes parallelism strictly explicit: omitted `n_threads` always means one
  native thread, `NumericalConfig()` defaults to `1`, and environment variables
  cannot enable threads implicitly.
- Preserves fixed-input/fixed-draw results across supported thread counts and
  reports the same smallest failure index in sequential and parallel kernels.
- Serializes mutable operations on one model instance and protects prepared
  evaluator workspace, while allowing independent models to execute
  concurrently and safely in rolling-window workflows.
- Adds prepared static likelihood evaluators, reusable Student workspaces and
  inverse-CDF caches, and equicorrelation sufficient-statistic reuse across
  optimizer calls.
- Adds `fit_independent` for process-level batches and extends rolling
  `risk_metrics` with explicit `n_jobs`, `n_threads`, multiprocessing method,
  per-window seed ownership, and nested-parallelism diagnostics.
- Adds a dependency-free native linear-algebra layer with scalar and portable
  compiler-vectorizable backends; no Eigen, BLAS, OpenMP, or other external
  runtime dependency is introduced.
- Adds the parallel release-gate matrix: strict GCC/Clang/MSVC wheels,
  Linux/macOS/Windows dependency audits, ASan/UBSan, ThreadSanitizer, Unix
  process-lifecycle stress, allocation/RSS instrumentation, explicit
  subinterpreter rejection, and aggregate validation artifacts.
- Documents CPU configuration, thread safety, deterministic sampling,
  oversubscription, diagnostics, and current large-dimension limits. Dense
  correlation modes remain `O(d^2)`; factor Student and Gaussian modes are
  available only when `corr_mode="factor"` is selected explicitly.

## 0.18.0 - 2026-07-21

Version: `0.17.5` -> `0.18.0`

Commit: `9b99d59`<br>
Merge PR: #41 (`cd87208`, 2026-07-21)

- Adds conditional sampling and prediction for static multivariate Gaussian
  and Student copulas, with fitted-correlation and analytical-moment coverage.
- Fixes top-level `fit()` state synchronization so subsequent stateful
  `predict()` and `sample()` calls use the requested strategy result and data.
- Rejects dynamic methods for static Gaussian and Student copulas instead of
  silently ignoring them.
- Fixes stochastic multivariate prediction to draw an independent stationary
  latent parameter per sample, and prevents joint-parameter posterior queries
  from mutating the model correlation state.
- Hardens stochastic Student fitting against false optimizer convergence,
  caches its Cholesky factor, and caps the optional Student inverse-CDF table
  at 256 MiB with an exact-quantile fallback for larger inputs.
- Accelerates Stochastic Student SCAR-TM-OU fitting with internal
  `(log(kappa), mu, log(sigma_x))` optimizer coordinates while preserving the
  public `(kappa, mu, nu)` parameterization and reporting the selected
  parameterization in fit diagnostics.
- Stabilizes Student degrees-of-freedom gradients with native analytical
  quantile derivatives, dense inverse-CDF nodes near the `df > 2` boundary,
  coverage through `df = 1000`, and a controlled large-`df` normal asymptotic.
- Improves vine family screening with exact public Kendall-tau mappings,
  preserving small interior dependence, Gaussian signs, and rotations while
  handling only true parameter boundaries as unsupported starts.
- Reuses each selected edge's static MLE to initialize dynamic fitting,
  computes both h-function directions in one posterior pass, avoids redundant
  pseudo-observation copies, and caches conditional suffix plans.
- Reports requested versus actual R-vine edge methods, dynamic-fit fallbacks,
  discarded optimizer work, failure messages, and edge-level fit timings.
- Reuses R-vine edge statistics and top-candidate static evaluators during
  family selection, and screens candidates with value-only native likelihoods.
- Accelerates long SCAR-TM-OU and SCAR-M/P-OU model-sampling trajectories with
  a sequential cached Numba OU kernel while preserving the NumPy RNG stream.
- Reuses prepared Gaussian normal quantiles and fixed-grid spectral terms in
  SCAR-OU gradient and mixture-h evaluation without changing transition
  backends or basis-order policy.
- Caches fitted or content-fingerprinted R-vine prediction histories and
  terminal dynamic states, reducing repeated 6D SCAR-TM-OU `predict(1000)`
  from about 351 ms cold to a 4.5 ms warm median without parallel execution.
- Replaces row-by-row Python GAS R-vine sampling with a sequential native
  stateful kernel while preserving the exact RNG stream, score-update order,
  rotations, mixed-edge behavior, and generic-strategy fallback.
- Reduces median GAS `RVineCopula.sample(1000)` time on the profiled 6D data
  from about 878 ms to 3.2 ms without parallel execution.
- Optimizes native multivariate hot paths through shared-correlation Cholesky
  reuse, reusable Student workspaces, equicorrelation sufficient statistics,
  prepared SCAR-OU snapshot caching, and zero-copy contiguous inputs while
  preserving jitter, validation, fixed-draw, and compatibility semantics.
- Adds opt-in MinGW-w64 builds on Windows with statically linked GCC runtimes;
  MSVC remains the default toolchain.
- Publishes the PEP 561 `py.typed` marker and expands public annotations,
  docstrings, guides, API references, regression tests, and benchmark reports.

## 0.17.5 - 2026-07-06

Version: `0.17.4` -> `0.17.5`

Commit: `ce8068e`<br>
Merge PR: #40 (`09181f3`, 2026-07-06)

- Adds GAS fitting support for `StochasticStudentCopula`, including fixed-correlation parameter accounting and joint static shrinkage-correlation estimation.
- Fails unsupported stochastic Student GAS correlation modes explicitly before mutating fitted model state.
- Adds `GASResult.parameter_count` and richer correlation diagnostics for stochastic Student GAS results.
- Improves top-level API dispatch for vine models and exposes aggregate `RVineCopula.fit_result` metadata.
- Adds vine contract helpers: `SelectedCopula`, fixed `copulas=` validation, clearer pair-copula candidate errors, and fitted R-vine matrix conversion APIs.
- Refactors shared strategy plumbing for L-BFGS-B option overrides, removed `tol` handling, OU initial-point diagnostics, and SCAR-MC fit flow.
- Updates documentation, source-distribution packaging entries, and regression coverage for stochastic Student GAS, vine APIs, strategy validation, and numerical option checks.

## 0.17.4 - 2026-07-03

Version: `0.17.3` -> `0.17.4`

Commit: `6d0d4b8`  
Merge PR: #39 (`593bb0a`, 2026-07-03)

- Stops `pyscarcopula` package import from changing BLAS thread environment variables.
- Removes the obsolete `pyscarcopula.api.configure` BLAS thread helper.
- Adds top-level `pyscarcopula.__version__` using installed distribution metadata.
- Refreshes documentation for external BLAS thread management and explicit benchmark/validation flags.
- Adds bash and Windows PowerShell examples for optional benchmark and large validation checks.

## 0.17.3 - 2026-06-30

Version: `0.17.2` -> `0.17.3`

Commit: `0d3d243`  
Merge PR: #38 (`0d39093`, 2026-07-01)

- Adds prepared C++ SCAR-OU evaluators, native Student correlation-gradient plumbing, and Python strategy integration for repeated fit/posterior evaluations.
- Expands R-vine dynamic conditioning, prediction diagnostics, performance profiling, and related test coverage.
- Adds stricter invalid-data handling for multivariate Gaussian and StochasticStudent fits without partial state mutation.

## 0.17.2 - 2026-06-26

Version: `0.17.1` -> `0.17.2`

Commit: `5c40246`  
Merge PR: #37 (`b6a1997`, 2026-06-26)

- Improve risk_metrics compatibility across independent, bivariate, R-vine, and stochastic Student copulas.
- Expose NumericalConfig and LBFGSBConfig from the public top-level API.
- Make stochastic Student posterior weights more robust after mixed latent/MLE fit flows.
- Refresh documentation for prediction semantics, GAS/SCAR-TM behavior, R-vine conditioning, and public imports.
- Remove obsolete DCC-oriented tests and add targeted regression coverage.
- Bump package version to 0.17.2.

## 0.17.1 - 2026-06-25

Version: `0.17.0` -> `0.17.1`

Commit: `71f1476`  
Merge PR: #36 (`08ff748`, 2026-06-25)

- Count Kendall-derived fixed correlations in effective parameter totals and diagnostics.
- Reweight SCAR Rosenblatt transforms by observed prefixes for equicorr Gaussian and stochastic Student models.
- Restrict data-estimated static correlation paths to MLE and SCAR-TM-OU.
- Add mathematical-contract documentation and update related guide/API pages.
- Bump package version to 0.17.1.

## 0.17.0 - 2026-06-21

Version: `0.16.1` -> `0.17.0`

Commit: `1fdec85`  
Merge PR: #35 (`2de1939`, 2026-06-21)

- Migrated built-in copula operations, static likelihoods, GAS, and SCAR-TM-OU computations to the required C++ backend.
- Reorganized the C++ core into dedicated copula, GAS, likelihood, SCAR-OU, and binding modules.
- Introduced the `CopulaBase`, `BivariateCopula`, and `MultivariateCopula` hierarchy with explicit capability-based strategy validation.
- Promoted Gaussian, Student, equicorrelation, and stochastic Student models to `copula.multivariate`.
- Added joint correlation estimation and analytical gradients for multivariate Student models.
- Unified fitted-model `sample()` and `predict()` contracts across bivariate, multivariate, and vine copulas.
- Added `MultivariateMLEResult`, improved diagnostics, AIC/BIC reporting, and persistence migrations.
- Removed obsolete Python backends, compatibility aliases, and the experimental DCC model.
- Updated documentation, examples, architecture notes, and regression/contract coverage.

Breaking changes:

- The C++ extension is now mandatory; GAS and SCAR-TM-OU no longer accept `backend=`.
- `copula.experimental` was replaced by `copula.multivariate`.
- `StochasticStudentDCCCopula` was removed.
- `sample_mode`, `LatentResult.alpha`, `u_train`, legacy Jacobi aliases, and deprecated numerical modules were removed.
- Use `sample_at_parameter()` for explicit bivariate parameter sampling.

## 0.16.1 - 2026-06-21

Version: `0.16.0` -> `0.16.1`

Commit: `fc2bfd8`  
Merge PR: #35 (`2de1939`, 2026-06-21)

- Migrated built-in copula operations, static likelihoods, GAS, and SCAR-TM-OU computations to the required C++ backend.
- Reorganized the C++ core into dedicated copula, GAS, likelihood, SCAR-OU, and binding modules.
- Introduced the `CopulaBase`, `BivariateCopula`, and `MultivariateCopula` hierarchy with explicit capability-based strategy validation.
- Promoted Gaussian, Student, equicorrelation, and stochastic Student models to `copula.multivariate`.
- Added joint correlation estimation and analytical gradients for multivariate Student models.
- Unified fitted-model `sample()` and `predict()` contracts across bivariate, multivariate, and vine copulas.
- Added `MultivariateMLEResult`, improved diagnostics, AIC/BIC reporting, and persistence migrations.
- Removed obsolete Python backends, compatibility aliases, and the experimental DCC model.
- Updated documentation, examples, architecture notes, and regression/contract coverage.

Breaking changes:

- The C++ extension is now mandatory; GAS and SCAR-TM-OU no longer accept `backend=`.
- `copula.experimental` was replaced by `copula.multivariate`.
- `StochasticStudentDCCCopula` was removed.
- `sample_mode`, `LatentResult.alpha`, `u_train`, legacy Jacobi aliases, and deprecated numerical modules were removed.
- Use `sample_at_parameter()` for explicit bivariate parameter sampling.

## 0.16.0 - 2026-06-02

Version: `0.15.1` -> `0.16.0`

Commit: `3310a3f`  
Merge PR: #27 (`25115db`, 2026-06-02)

- Added optional pybind11 C++ backend for SCAR-TM-OU likelihoods, gradients, forward passes, and copula h/h_inverse kernels.
- Added backend selection via `backend='auto' | 'python' | 'cpp'`, with Python fallback for unsupported C++ combinations.
- Updated auto TM routing to use `spectral -> matrix -> local` fallback diagnostics.
- Added adaptive spectral basis order policy and fit diagnostics in `LatentResult`.
- Added C++ wheel build workflow and source packaging entries.
- Updated docs, README, examples, and tests for the new backend and fallback behavior.
- Forwarded `n_jobs` to rolling marginal fitting in risk metrics.

## 0.15.1 - 2026-05-29

Version: `0.15.0` -> `0.15.1`

Commit: `78aa0d0`  
Merge PR: #26 (`998f0fd`, 2026-05-29)

- Added full-sample emission caches for fixed-R and DCC Stochastic Student copulas, including block offset support and cache invalidation.
- Wired cached emissions into TM, TM gradients, Hermite-TM, SCAR TM, GAS, and SCAR GOF paths.
- Added Numba fast paths for fixed-R Student GAS unit filtering and DCC recursion/log-likelihood evaluation.
- Added block/vectorized Student GOF helpers and forward-weight block array iteration.
- Updated model serialization to respect `__getstate__` / `__setstate__` and exclude transient caches.
- Added no-copy float64 array normalization helpers across numerical paths.
- Expanded tests for cache lifecycle, persistence, cached block emissions, fast kernels, TM/Hermite/GOF integration, DCC recursion, and benchmark smoke coverage.
- Updated experimental model docs and bumped package version to 0.15.1.

## 0.15.0 - 2026-05-24

Version: `0.14.1` -> `0.15.0`

Commit: `9b1b74e`  
Merge PR: #25 (`2bc4b90`, 2026-05-24)

- Add scar-tm-jacobi with Jacobi tau dynamics, strategy dispatch, prediction, GoF support, and persistence metadata.
- Add Jacobi numerical backends: spectral matrix, local Lamperti transition, fixed-grid gradient path, and coefficient filtering.
- Refactor transition-method handling: replace legacy gh naming with local, centralize normalization, and simplify SCAR-OU auto routing with spectral-to-local fallback.
- Add Kendall tau mappings for supported copulas and improve Archimedean h-function numerical stability.
- Remove deprecated smoothed_params aliases and standardize on predictive_mean.
- Update docs and tests for Jacobi fitting, local transitions, optimizer defaults, and numerical edge cases.

## 0.14.1 - 2026-05-23

Version: `0.14.0` -> `0.14.1`

Commit: `afc50cb`  
Merge PR: #24 (`0964728`, 2026-05-23)

- Fixed SCAR-TM-OU auto/spectral backend:
- Hermite spectral recursion now preserves absolute t_index for time-varying models.
- StochasticStudentDCCCopula.pdf_and_grad_on_grid_batch now evaluates the correct R_t slice.
- Prevents spectral likelihood/gradient evaluation from using misaligned time-varying correlation paths.
- Fixed PPF cache handling in experimental Student models:
- Cache validation no longer relies only on id(u).
- Prevents false cache hits for temporary NumPy views.
- Hardened optimizer failure handling:
- SCARTMStrategy no longer reports success=True when the objective is the failure sentinel.
- Invalid objective values now force success=False.
- Fixed bivariate RVine GoF behavior:
- RVine with d=2 now matches the corresponding CVine/bivariate Rosenblatt transform and GoF result.
- Added regression coverage:
- Spectral gradient vs finite differences.
- Hermite batch t_index propagation.
- Failed objective cannot be reported as optimizer success.
- Experimental PPF cache safety.
- CVine/RVine GoF equality for the bivariate case.

## 0.14.0 - 2026-05-22

Version: `0.13.0` -> `0.14.0`

Commit: `484d1b8`  
Merge PR: #23 (`df76d07`, 2026-05-22)

- Add conditional sampling for experimental multivariate copulas via `given={idx: value}`.
- Add shared conditional samplers for equicorrelation Gaussian and Student-t copula families, including DCC `R_t` paths.
- Route top-level `api.predict(..., given=...)` through multivariate `sample_conditional` when available.
- Improve TM/GAS experimental-model support with row-wise density APIs, block-based GoF helpers, and model-specific GAS optimizer configs.
- Update experimental model docs and bump package version to `0.14.0`.

## 0.13.0 - 2026-05-17

Version: `0.12.1` -> `0.13.0`

Commit: `13d525a`  
Merge PR: #22 (`d053ade`, 2026-05-17)

- Refactors vine runtime around a shared PairCopula container and strategy-generic edge dispatch.
- Extracts R-vine sampling and prediction logic into focused helpers for suffix conditioning, DAG/MCMC fallback, dynamic conditioning, and summary formatting.
- Replaces legacy tol handling with unified LBFGSBConfig optimizer options across MLE, GAS, SCAR, and experimental copulas.
- Improves SCAR-TM initialization with a dependence-aware starting point while keeping legacy and GAS warm-start paths.
- Fixes GAS score_eps propagation after fitting and during vine sampling/prediction.
- Expands public API support for multivariate and experimental copulas.
- Adds generic strategy dispatch coverage for C-vine/R-vine runtime.
- Adds numerical regression tests for OU kernels, TM forward likelihood, optimizer config, dynamic conditioning, and multivariate prediction.
- Updates architecture and R-vine conditioning documentation.

## 0.12.1 - 2026-05-09

Version: `0.12.0` -> `0.12.1`

Commit: `905dde1`  
Merge PR: #21 (`dac6acb`, 2026-05-09)

- Added shared numba-backed parameter transforms for Archimedean copulas and reused them across Clayton, Frank, Gumbel, and Joe implementations.
- Made transform_type handling explicit for GAS filtering, including validation for supported transform modes.
- Added configurable GAS optimizer ftol via NumericalConfig.default_ftol_gas and per-fit ftol, forwarding it to L-BFGS-B.
- Added regression tests for transform derivatives, inverse transforms, near-boundary softplus behavior, and GAS ftol forwarding.
- Updated performance documentation with GAS, SCAR, vine tuning guidance, and removed the standalone equicorrelation API nav page.
- Bumped package version to 0.12.1.

## 0.12.0 - 2026-05-08

Version: `0.11.0` -> `0.12.0`

Commit: `a04cc3f`  
Merge PR: #20 (`234d6ff`, 2026-05-08)

- Rename dynamic-model parameters for clearer public API:
- SCAR OU: `theta` -> `kappa`.
- GAS score sensitivity: `alpha` -> `gamma`.
- Add a numba-accelerated GAS filter path for supported bivariate copulas, including rotations and both `softplus`/`xtanh` transforms.
- Make `softplus` the default transform for supported pair copulas and update docs/examples accordingly.
- Improve SCAR-TM/vine propagation by carrying fitted TM settings through likelihood, prediction, and mixture `h` computations.
- Update prediction terminology from smoothed paths to `predictive_mean` and align notebooks/docs with current APIs.
- Change model persistence default to avoid embedding training data unless `include_data=True`.
- Bump package version to `0.12.0`.

## 0.11.0 - 2026-05-04

Version: `0.10.1` -> `0.11.0`

Commit: `b9646aa`  
Merge PR: #19 (`2bdba36`, 2026-05-04)

- Introduce `predictive_mean` as the explicit API for dynamic copula parameter paths, while keeping `smoothed_params` as a backward-compatible alias.
- Fix TM forward prediction by adding `TMGrid.predict_matvec` and using it in forward weights, mixture h, and predictive state calculations.
- Add parametric bootstrap calibration for bivariate GoF tests, including bootstrap statistics and per-iteration diagnostics.
- Add AR(0/1)-GARCH(1,1) marginal filtering utilities for standardized residuals and pseudo-observations.
- Improve GAS fitting and filtering with reusable score computation, configurable optimizer bounds/options, and updated default initialization.
- Replace Joe copula sampling with conditional inversion to avoid slow heavy-tailed frailty draws in bootstrap workflows.
- Update documentation and tests for the predictive mean terminology and new numerical behavior.

## 0.10.1 - 2026-05-02

Version: `0.10.0` -> `0.10.1`

Commit: `cac8731`  
GitHub PR text: not found

- updated docs

## 0.10.0 - 2026-05-01

Version: `0.9.1` -> `0.10.0`

Commit: `092d797`  
Merge PR: #18 (`708d000`, 2026-05-01)

- Add JSON-based model persistence through `model.save()` / `Model.load()` and top-level `save_model()` / `load_model()` helpers.
- Cover fitted bivariate copulas, elliptical copulas, C-vines, and R-vines, including fitted results and conditional R-vine structure metadata.
- Split the old monolithic example notebook into focused notebooks under `examples/`: basic API, bivariate models, vine models, risk metrics, and a pyvinecopulib comparison notebook.
- Shorten README to a project overview with links to notebooks and docs.
- Improve R-vine MLE performance with faster Kendall tau calculation, less pair-copula selection overhead, prepared Gaussian logpdf / h-function paths, paired h-function kernels for Archimedean copulas, and faster static R-vine sampling/loglik loops.
- Tighten Gumbel h-inverse and cover extreme quantiles with an additional regression test.

## 0.9.1 - 2026-04-26

Version: `0.9.0` -> `0.9.1`

Commit: `f9edd1b`  
GitHub PR text: not found

- updated docstrings rvine; added new conditional sampling tests

## 0.9.0 - 2026-04-25

Version: `0.8.1` -> `0.9.0`

Commit: `ce2fb6d`  
GitHub PR text: not found

- extended r-vine conditional sampling

## 0.8.1 - 2026-04-23

Version: `0.8.0` -> `0.8.1`

Commit: `3bfbab3`  
Merge PR: #16 (`100af7e`, 2026-04-23)

- Fix GAS predictive semantics:
- Use the final score update for one-step-ahead prediction.
- Honor `horizon='current'/'next'`.
- Store/use the correct predictive parameter.
- Refactor vine predictive parameter generation:
- Route `get_r_predict` / predict-time edge params through strategies.
- Remove unsafe GAS fallback behavior.
- Keep GAS vine sampling stepwise instead of collapsing to a constant path.
- Make sampling reproducible:
- Pass rng through copula sampling/predict paths.
- Remove remaining implicit `np.random` usage in predictive code.
- Use independent per-window seeds in `risk_metrics`.
- Improve SCAR-TM predictive sampling:
- Add shared grid-sampling helper.
- Support `predictive_r_mode`.
- Keep default behavior centralized.
- Fix conditional predict API behavior:
- Bivariate `predict(..., given=...)` now actually applies conditioning.
- C-vine conditional guard now allows independent edges correctly.
- Add regression and validation coverage for API/vine/R-vine predictive behavior, GAS/SCAR horizon semantics, reproducibility, conditional sampling vs analytical oracles, Rosenblatt residuals, family sweep, and dynamic recovery cases.

## 0.8.0 - 2026-04-21

Version: `0.7.1` -> `0.8.0`

Commit: `e9ab1a4`  
Merge PR: #15 (`3e119af`, 2026-04-21)

- Simplify `RVineCopula.predict(..., given=...)` so R-vine conditional sampling is supported only when the fixed variables can be placed at the end of the R-vine variable order.
- Support this directly in the fitted natural-order matrix, or after rebuilding an equivalent natural-order matrix from the same fitted tree structure.
- Make arbitrary non-rebuildable conditioning patterns fail explicitly instead of falling back to the old DAG planner.
- Remove the DAG conditional sampling implementation and its dedicated tests.
- Remove old optional validation coverage for graph/grid/exact R-vine conditional methods.
- Reduce `_conditional_rvine.py` to given validation.
- Update RVineCopula.predict docs and error text to use clearer terminology: R-vine variable order, anti-diagonal, and natural-order matrix.
- Update user-facing docs, README, and example.ipynb to describe the current supported path.
- Replace old "peel-order suffix" language with "fixed variables can be placed at the end of the R-vine variable order."
- Unsupported arbitrary given patterns now raise: `RVineCopula.predict: fixed variables cannot be placed last in the R-vine variable order; arbitrary conditional sampling is not supported`.

## 0.7.1 - 2026-04-17

Version: `0.7.0` -> `0.7.1`

Commit: `86c171d`  
Merge PR: #14 (`5cd68bd`, 2026-04-17)

- Add user-facing documentation for arbitrary RVine conditional sampling:
- `conditional_method='auto' | 'graph' | 'grid' | 'exact'`.
- `conditional_plan(given)`.
- `flexible_graph_plan(given)`.
- `structure_mode='conditional'`.
- Current capabilities and limitations.
- Update the Vine API docs to expose RVine conditional helpers.
- Update `example.ipynb` with conditional sampling examples for R-vines, posterior workload diagnostics, conditional-optimized structure selection, and a rerun Risk metrics section with `RVineCopula()`.
- Add MkDocs nav entries for existing Equicorrelation pages and RVine conditional notes.
- Fix MkDocs strict-build autorefs warnings from a docstring formula.
- Add `.gitignore` entry for `*.egg-info/`.
- Make `RVineCopula.fit` handle non-encodable Dissmann-selected regular-vine tree sets by warning and refitting a matrix-encodable C-vine fallback.
- Keep sample, predict, and rolling Risk metrics workflows usable when selected Dissmann structures cannot be represented by `RVineMatrix`.

## 0.7.0 - 2026-04-15

Version: `0.6.1` -> `0.7.0`

Commit: `beda473`  
GitHub PR text: not found

- add conditional predict for copulas and vines

## 0.6.1 - 2026-04-15

Version: `0.6.0` -> `0.6.1`

Commit: `648e679`  
GitHub PR text: not found

- updated cvm_test; updated docs

## 0.6.0 - 2026-04-15

Version: `0.5.2` -> `0.6.0`

Commit: `99e2c20`  
GitHub PR text: not found

- updated RVineMatrix, added new RVine tests, updated log eval for Clayton

## 0.5.2 - 2026-04-09

Version: `0.5.1` -> `0.5.2`

Commit: `6241a76`  
GitHub PR text: not found

- fix rvine structure, added experimental models

## 0.5.1 - 2026-04-05

Version: `0.5.0` -> `0.5.1`

Commit: `21f9d29`  
GitHub PR text: not found

- added r-vine

## 0.5.0 - 2026-04-05

Version: `0.4.2` -> `0.5.0`

Commit: `3a45089`  
GitHub PR text: not found

- fix version

## 0.4.2 - 2026-03-31

Version: `0.4.1` -> `0.4.2`

Commit: `1f885ed`  
GitHub PR text: not found

- optimized vine copula selection, h_inverse perfomance

## 0.4.1 - 2026-03-30

Version: `0.4.0` -> `0.4.1`

Commit: `1c792c6`  
GitHub PR text: not found

- copula mle optimization

## 0.4.0 - 2026-03-30

Version: `0.3.1` -> `0.4.0`

Commit: `a44cc5f`  
GitHub PR text: not found

- sampling fix

## 0.3.1 - 2026-03-28

Version: `0.3.0` -> `0.3.1`

Commit: `63e77ef`  
Merge PR: #12 (`54d31bd`, 2026-03-28)

- refactor: unified FitResult types, GAS dedup, delete latent/

## 0.3.0 - 2026-03-27

Version: `0.2.0` -> `0.3.0`

Commit: `40fd4cf`  
Merge PR: #10 (`170f8a0`, 2026-03-27)

- refactor: remove legacy code

## 0.2.0 - 2026-03-22

Version: `0.1.1` -> `0.2.0`

Commit: `5ebd979`  
GitHub PR text: not found

- update docs, refactor forward pass, add softplus transform

## 0.1.1 - 2026-03-22

Version: `0.1.0` -> `0.1.1`

Commit: `905ba55`  
GitHub PR text: not found

- added docs

## 0.1.0 - 2026-03-22

Version: initial -> `0.1.0`

Commit: `bf52a4f`  
GitHub PR text: not found

- install update
