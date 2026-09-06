# Conditional sampling validation

Conditional sampling is checked in four deliberately separate layers.  The
split keeps pull requests fast, preserves strong distributional evidence, and
prevents external-library or wall-clock noise from being mistaken for a core
correctness failure.

## Support surface

The machine-readable source of truth is
`tests/conditional/support_matrix.json`.  It covers the following canonical
runtimes:

| Runtime group | Conditional paths covered |
|---|---|
| Six bivariate families | both conditioning directions; all 15 supported family/rotation cells |
| Gaussian and Student copulas | dense and factor exact kernels |
| Equicorr Gaussian and Stochastic Student | MLE, GAS, and supported latent predictive paths |
| Generic `VineCopula` | direct suffix, rebuilt suffix, and DAG+MCMC routing for C-, D-, and R-vines |
`RVineCopula` is a compatibility alias for `VineCopula`, not a thirteenth
runtime.  Unsupported method/correlation combinations are explicit negative
contract cases in the registry.

## Test layers

| Layer | Trigger | Selection | Purpose |
|---|---|---|---|
| PR smoke | pull request and push to `master` | non-validation, non-benchmark, non-external | API contracts, deterministic parity, routing, seeds, fixed columns |
| Validation | push to `master` or one-time manual run | `validation` excluding external/benchmark | analytical and distributional gates, including non-external `d=50` cases |
| External/high-dimensional | one-time manual run only | `external or high_dimensional`, excluding benchmark | pinned pyvine parity, full high-dimensional matrix, independent oracle tests |
| Benchmark | manual only | benchmark contracts plus permanent runner | warmed JSON/CSV measurements; never a wall-clock correctness gate |

The workflow is `.github/workflows/conditional-sampling.yml`. It has no
scheduled trigger. A one-time manual run accepts `pr-smoke`, `validation`,
`external-high-dimensional`, `benchmark`, or `all` as its layer. The external
environment pins `pyvinecopulib==0.7.5` through the `external` optional
dependency.

## Local commands

Activate a workspace venv as described in the
[installation guide](../getting-started/installation.md), then build/install the
native extension before running the suite:

```bash
python -B tools/run_in_workspace.py -- -m pip install -e ".[test]"
```

PR smoke:

```bash
python -B tools/run_in_workspace.py -- -m pytest -q tests/conditional --strict-markers \
  -m "not validation and not benchmark and not external"
```

Distributional validation, including non-external high-dimensional cases:

```bash
python -B tools/run_in_workspace.py -- -m pytest -q tests/conditional --strict-markers --run-validation \
  -m "validation and not benchmark and not external"
```

Pinned external and high-dimensional one-time layer:

```bash
python -B tools/run_in_workspace.py -- -m pip install -e ".[test,external]"
python -B tools/run_in_workspace.py -- -m pytest -q tests/conditional --strict-markers --run-validation \
  -m "(external or high_dimensional) and not benchmark"
```

Manual benchmark artifact:

```bash
python -B tools/run_in_workspace.py -- tools/benchmark_conditional_sampling.py \
  --profile full --n-draws 1024 --mcmc-draws 8 \
  --repeats 5 --warmups 1 --n-threads 4 --include-mcmc
```

Set `PYSCA_RUN_BENCHMARKS=1` only when directly running pytest cases marked
`benchmark`.  The permanent benchmark CLI does not need that variable.

## Public input boundary matrix

The following checks call the production entry points. Object/API means both
`model.fit/sample/predict` and the corresponding functions in `pyscarcopula.api`.
Validation performed by a test adapter is not evidence for these contracts.

| Models | Method | Entry points | Boundary and expected behavior | Test module |
|---|---|---|---|---|
| Gumbel | MLE, GAS, SCAR-TM-OU, SCAR-TM-JACOBI | Object/API fit, then predict | C/F, strided and read-only inputs; fitting owns its history; later caller mutation leaves seeded prediction unchanged | `test_fit_input_contracts` |
| Independent | MLE | Object/API fit | Saved observations do not share the caller's buffer | `test_fit_input_contracts` |
| Gumbel, Independent, Gaussian, Student | MLE | Object/API fit | Ties use ordinal ranks in input order; raw data remains unchanged | `test_fit_input_contracts` |
| Gumbel, Independent, Gaussian, Student, equicorrelated Gaussian, stochastic Student, C-vine | MLE | Object/API fit | Empty input rejected without publishing fitted state | `test_fit_input_contracts` |
| Gumbel | MLE, GAS | Object/API fit | One observation accepted | `test_fit_input_contracts` |
| Gumbel | SCAR-TM-OU, SCAR-TM-JACOBI | Object/API fit | At least two observations required | `test_fit_input_contracts` |
| Independent, equicorrelated Gaussian, stochastic Student with fixed R | MLE | Object/API fit | One observation accepted | `test_fit_input_contracts` |
| Gaussian/Student with estimated R, automatically selected C-vine | MLE | Object/API fit | One observation rejected | `test_fit_input_contracts` |
| Independent, Gaussian, Student, equicorrelated Gaussian, stochastic Student | MLE | Object/API fit | Exact 0/1 and their inward `nextafter` neighbors accepted | `test_fit_input_contracts` |
| Automatically selected C-vine | MLE | Object/API fit | Exact 0/1 rejected; inward `nextafter` neighbors accepted | `test_fit_input_contracts` |
| Six pair families, four multivariate families, generic vine | MLE | Object/API sample/predict | Negative, Boolean, floating and string sizes rejected before RNG consumption; NumPy integer sizes accepted; fully conditioned predict also validates size | `conditional/test_api_contracts` |
| Same MLE families | MLE | Object/API sample/predict | Zero returns an empty array for pair/multivariate models; vine rejects it; RNG unchanged | `conditional/test_api_contracts` |
| Gumbel after real fitting | GAS, SCAR-TM-OU, SCAR-TM-JACOBI | Object/API sample/predict | Invalid sizes rejected; NumPy integer accepted; zero rejected by GAS and accepted by SCAR without RNG consumption | `test_dynamic_sampling_input_contracts` |

These are explicit coverage subsets, not the complete model/method Cartesian
product. Singleton acceptance describes input handling, not parameter
identifiability. Closed-interval input acceptance does not guarantee a finite
likelihood at singular family boundaries. Distributional and tail tests remain
necessary in addition to the input matrix.

Shape/dtype/nonfinite inputs, refit rollback, resource limits, conditioning,
persistence and dynamic horizons have additional coverage in
`test_static_correlation_acceptance`, `test_real_numeric_inputs`,
`test_prepared_input_contracts`, `test_sampling_resource_limits`,
`test_strategy_state_validation` and the conditional test suite.

## Statistical stability policy

Monte Carlo bounds are defined from sampling error and verified by independent
oracle tests. Calibration captures are development evidence rather than part
of the product repository or CI contract.

Do not loosen a tolerance after inspecting a production failure.  First
reproduce the node ID and seed, run the same assertion on oracle draws, and
determine whether the problem is a sampler defect, a numerical-boundary case,
or an unstable test budget.

## Benchmark evidence

Native performance workloads are defined in
`benchmarks/native_performance_v3.json`.
Run `python tools/run_native_benchmarks.py --help` for capture and comparison
options. The workload manifest uses schema 3
and captures use schema 5. Record both a new baseline and candidate with these
formats; captures from earlier formats are rejected for regression comparison.
Keep local evidence below `build/`, for example by passing
`--artifact-root build/benchmarks/reference-run`. Evidence tools accept this
generated directory while rejecting outputs placed among product source files.

The benchmark CLI writes both JSON and CSV.  The artifact includes commit and
runtime/compiler/CPU metadata; each record includes model case, path, seed,
`d`, `k_free`, draw and thread counts, warm-up/repeat counts, median
throughput, Python allocation peak, process RSS, and fixed-column/open-unit
invariants.

Wall time is not gated on GitHub-hosted or otherwise shared runners.  Compare
medians only across repeated runs on the same dedicated runner, after warm-up.
The forced DAG+MCMC records are diagnostic and may carry expected convergence
warning codes; they are not a substitute for the analytical MCMC validation
suite.

## Failure triage

1. Copy the exact pytest node ID from the uploaded JUnit artifact and rerun it
   with the same marker selection and `--run-validation` setting.
2. For a contract or routing failure, inspect fixed columns, seed parity, and
   `conditional_method` diagnostics before running a larger sample.
3. For an analytical/distributional failure, preserve the failing seed and
   compare the standardized error with the reported Monte Carlo budget.  Run
   the oracle-only calibration; do not repeatedly rerun until green.
4. For an external failure, confirm the pinned pyvine version and determine
   whether the mismatch is edge conversion, h/h-inverse direction, rotation,
   or vine-matrix convention.
5. For a high-dimensional failure, separate correctness from memory-budget
   and factor-compactness contracts.  Do not turn a hosted-runner wall time
   excursion into a correctness regression.
6. For DAG+MCMC, inspect acceptance, accepted moves per chain, transition
   budget, warning codes, and scale-free oracle errors together.  A warning is
   not automatically a failed distributional gate, and agreement between two
   MCMC chains is not proof of exactness.

Every CI layer uploads JUnit output and runner metadata even when pytest
fails. The one-time external/high-dimensional layer uploads JUnit results and
runner metadata. The support inventory remains in
`tests/conditional/support_matrix.json`; oracle assertions run inside pytest.
No separate inventory or calibration report is generated by that job. Manual
benchmark runs retain JSON/CSV evidence for 90 days.
