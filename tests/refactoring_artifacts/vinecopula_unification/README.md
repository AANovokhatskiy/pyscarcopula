# VineCopula unification artifacts

`test_stage0_contract.py` freezes the pre-unification `RVineCopula` behavior.
It may be removed after the generic `VineCopula` tests cover all permanent
contracts and every intentionally changed assertion is documented in
`VINECOPULA_UNIFICATION_PLAN.md`.

The format-v2 RVine persistence fixture is not temporary. It remains under
`tests/fixtures/persistence/` to guard backward-compatible loading.

`test_stage8_oracles.py` contains independent release-gate comparisons with
`pyvinecopulib`, permutation invariance, and Monte Carlo density
normalization. It is intentionally opt-in via `--run-validation` and may be
removed after the unification release gate is accepted.

`test_stage8_performance.py` records bounded auto/fixed fit, static sampling,
and suffix-conditioning workloads for the Stage 8 release evidence. It runs
only when `PYSCA_RUN_VINE_BENCHMARKS=1` and is marked `benchmark`.
