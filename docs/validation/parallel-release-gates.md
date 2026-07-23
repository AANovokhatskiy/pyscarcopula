# Parallel Release Validation

This report defines the evidence required before production work begins on
large-dimensional `EquicorrGaussianCopula` support.

## Current status

The release-gate automation is implemented in
`.github/workflows/parallel-release-gates.yml`. The gate is closed only when
one workflow run completes successfully and publishes the aggregate
`parallel-release-validation` artifact. Adding the jobs does not by itself
constitute a passed gate.

Local Windows evidence for the implementation commit:

- targeted release suite: `99 passed, 29 skipped` (Unix-only gates skipped);
- complete default suite: `1800 passed, 100 skipped`;
- strict MSVC `/W4 /WX` wheel built and imported outside the source tree;
- wheel SHA-256:
  `7DABC61AAAE3734B4AD1862FC7A4048BE1816ED8C06EA99DBC81ED12FC830EEA`;
- PE dependency audit: no OpenMP, BLAS, LAPACK, MKL, or OpenBLAS imports;
- embedded-subinterpreter rejection smoke: passed;
- strict MkDocs build and `git diff --check`: passed.

Linux GCC/Clang, macOS arm64, Unix stress, sanitizer, and glibc allocation
evidence remains pending until the workflow runs remotely.

## Required matrix

| Gate | Configuration | Acceptance |
|------|---------------|------------|
| Platform/compiler | Linux GCC on Python 3.10 and 3.14; Linux Clang on 3.12; Windows MSVC on 3.12; macOS arm64 Apple Clang on 3.12 | Strict warning build, wheel smoke outside the source tree, full non-benchmark suite |
| Binary dependencies | Every platform wheel | No OpenMP, BLAS, LAPACK, MKL, or OpenBLAS runtime |
| ASan/UBSan | Linux Clang | No project-local memory error or undefined behavior |
| ThreadSanitizer | Linux GCC | No data race or lock-order finding in parallel runtime and multivariate kernels |
| Unix lifecycle | Linux `spawn`, `fork`, and `forkserver` | `n_threads={1,2,4}`, failure recovery, PID ownership, and 100 child lifecycles without timeout/orphans |
| Allocation/memory | Linux glibc allocation probe and `/proc` RSS | No allocation per grid cell; bounded warm-call allocation count; no material RSS growth across 1000 repeated prepared calls |
| Subinterpreters | Embedded CPython smoke | Immediate, predictable rejection because subinterpreters are unsupported |

## Instrumentation

The process-local runtime exposes private counters for worker starts, submitted
tasks, and peak queued tasks. Student kernels additionally report workspace
growth and peak workspace bytes. A CI-only `LD_PRELOAD` allocation probe counts
glibc allocation calls and requested bytes around a native call; it is not
compiled into or distributed with wheels.

The binary audit uses `ldd` on Linux, `otool -L` on macOS, and an internal PE
import-table parser on Windows. Each report records the extension SHA-256,
platform, Python version, toolchain, and dependency list.

## Subinterpreter decision

The supported contract is explicit rejection. The pybind11 module remains
declared with `multiple_interpreters::not_supported()`. No claim of
subinterpreter safety is inferred from thread or multiprocessing tests.

## Published evidence

Every successful job uploads machine-readable JSON. The final workflow job
rejects any incomplete or failed dependency and aggregates the artifacts into:

- `release-validation.json`;
- `release-validation.md`.

The report is valid only for the commit SHA of that workflow run. Compiler
versions, wheel hashes, test logs, platform skips, sanitizer status, lifecycle
metrics, allocation counts, and memory growth must be retained with the
workflow artifacts.
