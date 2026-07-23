# Phase 8 Equicorr validation

## Verdict

The Phase-8 implementation passes all validation available in the local
Windows/MSVC environment. No blocker or high-severity finding was found.

The production release gate is **pending** until the updated workflow runs
successfully from a commit and publishes the aggregate
`parallel-release-validation` artifact. Windows cannot substitute for Linux
ASan/UBSan, ThreadSanitizer, glibc allocation/RSS, Unix process lifecycle, or
macOS arm64 checks.

## Functional and numerical validation

| Check | Result |
|---|---:|
| Prepared/static/GAS/SCAR/sampling focused suite | `149 passed` |
| Dedicated fitted batching and memory guards | included, passed |
| Full default suite | `1824 passed, 106 skipped` |
| Strict documentation build | passed |
| Whitespace/conflict check | passed |

The focused suite covers:

- dense versus prepared sufficient statistics and likelihood;
- deterministic reduction for row and dimension parallel axes;
- direct MLE, GAS and SCAR-TM-OU prepared consumers;
- immutable ownership plus NPZ and read-only mmap round trips;
- fitted `sample_batches` and `predict_batches`;
- per-row GAS recursion and SCAR path/posterior semantics;
- negative equicorrelation sampling without a dense matrix;
- pre-allocation memory rejection and bounded grid/sample output.

## Scaling validation

The hard gate used `PYSCA_ENFORCE_PERFORMANCE_GATES=1`. Results are medians
from the local 18-logical-CPU Windows host:

| `T` | `d` | 1 thread | 4 threads | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 100,000 | 3.379 ms | 1.088 ms | 3.11x | 0.78 |
| 32 | 10,000 | 10.732 ms | 2.837 ms | 3.78x | 0.95 |
| 32 | 100,000 | 107.318 ms | 26.885 ms | 3.99x | 1.00 |
| 1,000 | 10,000 | 334.244 ms | 83.458 ms | 4.00x | 1.00 |
| 1 | 1,000,000 | 33.976 ms | 8.791 ms | 3.86x | 0.97 |

Every measured workload exceeded `0.60`. The hard efficiency threshold applies
to workloads with at least 320,000 transformed elements; the smaller
`T=1,d=100,000` case remains a correctness and dimension-axis measurement.
Values near or above 1.0 can reflect cache and CPU turbo effects; they are
measurements, not a theoretical scaling guarantee.

## Local release-gate evidence

| Gate | Result |
|---|---|
| Release tests on Windows | `30 passed, 17 Unix/Linux skipped` |
| Forced strict native build | MSVC `/W4 /WX`, passed |
| Strict wheel | passed |
| Wheel import outside source tree | passed |
| Installed-wheel Phase-8 smoke | passed |
| PE dependency audit | passed, no forbidden runtime |
| Subinterpreter contract | immediate rejection, passed |
| Workflow YAML structure | parsed; all required jobs present |

Wheel:

- file: `pyscarcopula-0.19.0-cp312-cp312-win_amd64.whl`;
- SHA-256:
  `fd659ed3398e5666338308cdc74dc6ad51a79012ca1ed6ba5784591b52481520`;
- native extension SHA-256:
  `f0c1c5dd010bcf5a27bf33b9f4852941314ba9edef80bda96989fd85bcc8da53`.

The dependency audit found only the expected CPython, MSVC runtime, Windows
kernel, and universal CRT imports.

## Required remote evidence

After committing the Phase-8 validation changes, run
`.github/workflows/parallel-release-gates.yml`. The aggregate verdict must
include:

- strict GCC and Clang Linux wheels and full suites;
- strict MSVC and Apple Clang arm64 wheels and full suites;
- Linux ASan+UBSan and ThreadSanitizer runs including
  `test_equicorr_prepared.py`;
- `spawn`, `fork`, and `forkserver` lifecycle stress with Phase-8 preparation;
- glibc allocation and `/proc` RSS gates for prepared Equicorr hot paths;
- the enforced Phase-8 scaling job;
- dependency audits, wheel hashes, platform metadata, and subinterpreter
  rejection.

Phase 8 is ready for that workflow. Section 11 should begin only after its
aggregate artifact reports `verdict: passed`.
