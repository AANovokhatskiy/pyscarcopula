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
| Legacy `CVineCopula` | prefix/general legacy conditional algorithms |

`RVineCopula` is a compatibility alias for `VineCopula`, not a thirteenth
runtime.  Unsupported method/correlation combinations are explicit negative
contract cases in the registry.

## Test layers

| Layer | Trigger | Selection | Purpose |
|---|---|---|---|
| PR smoke | pull request and push to `master` | non-validation, non-benchmark, non-external | API contracts, deterministic parity, routing, seeds, fixed columns |
| Validation | push to `master`, nightly schedule, or manual | `validation` excluding external/benchmark | analytical and distributional gates, including non-external `d=50` cases |
| Nightly | nightly schedule or manual | `external or high_dimensional`, excluding benchmark | pinned pyvine parity, full high-dimensional matrix, 20-seed oracle calibration |
| Benchmark | manual only | benchmark contracts plus permanent runner | warmed JSON/CSV measurements; never a wall-clock correctness gate |

The workflow is `.github/workflows/conditional-sampling.yml`.  A manual run
accepts `pr-smoke`, `validation`, `nightly`, `benchmark`, or `all` as its
layer.  The nightly external environment pins `pyvinecopulib==0.7.5` through
the `external` optional dependency.

## Local commands

Build/install the native extension before running the suite:

```bash
python -m pip install -e ".[test]"
```

PR smoke:

```bash
python -m pytest -q tests/conditional --strict-markers \
  -m "not validation and not benchmark and not external"
```

Distributional validation, including non-external high-dimensional cases:

```bash
python -m pytest -q tests/conditional --strict-markers --run-validation \
  -m "validation and not benchmark and not external"
```

Pinned external and high-dimensional nightly layer:

```bash
python -m pip install -e ".[test,external]"
python -m pytest -q tests/conditional --strict-markers --run-validation \
  -m "(external or high_dimensional) and not benchmark"
```

Oracle-only false-failure calibration:

```bash
python tools/calibrate_conditional_statistical_gates.py \
  --runs 20 --max-failure-rate 0.01 \
  --output benchmark_artifacts/conditional_sampling_calibration.json
```

Manual benchmark artifact:

```bash
python tools/benchmark_conditional_sampling.py \
  --profile full --n-draws 1024 --mcmc-draws 8 \
  --repeats 5 --warmups 1 --n-threads 4 --include-mcmc
```

Set `PYSCA_RUN_BENCHMARKS=1` only when directly running pytest cases marked
`benchmark`.  The permanent benchmark CLI does not need that variable.

## Statistical stability policy

Monte Carlo bounds are defined from sampling error and calibrated only on
oracle-generated draws.  The calibration does not import a production model
or call a production sampler.  With 20 replicas, the empirical rate can only
move in increments of 5%, so the 1% acceptance threshold requires zero
observed false failures per named gate.

Do not loosen a tolerance after inspecting a production failure.  First
reproduce the node ID and seed, run the same assertion on oracle draws, and
determine whether the problem is a sampler defect, a numerical-boundary case,
or an unstable test budget.

## Benchmark evidence

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
fails.  Nightly additionally uploads the support inventory and calibration
report; manual benchmark runs retain JSON/CSV evidence for 90 days.
