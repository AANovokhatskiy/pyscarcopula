# Parallel Release Validation

This report defines the evidence required before production work begins on
large-dimensional `EquicorrGaussianCopula` support.

## Current status

The release-gate automation is implemented in
`.github/workflows/parallel-release-gates.yml`. The gate is closed only when
one workflow run completes successfully and publishes the aggregate
`parallel-release-validation` artifact. Adding the jobs does not by itself
constitute a passed gate.

The baseline gate for commit `2ca2792` was reported successful by the
repository owner before Phase 8 began. Phase-8 changes require a fresh run
because they add native preparation, prepared GAS/SCAR consumers, and new
parallel scaling paths.

Local Windows evidence for the current Phase-8 working tree:

- targeted Phase-8 suite: `149 passed`;
- targeted release suite: `30 passed, 17 skipped` (Unix-only gates skipped);
- complete default suite: `1824 passed, 106 skipped`;
- strict MSVC `/W4 /WX` wheel built and imported outside the source tree;
- wheel SHA-256:
  `FD659ED3398E5666338308CDC74DC6AD51A79012CA1ED6BA5784591B52481520`;
- PE dependency audit: no OpenMP, BLAS, LAPACK, MKL, or OpenBLAS imports;
- embedded-subinterpreter rejection smoke: passed;
- enforced Phase-8 scaling matrix: `5 passed`; every four-thread efficiency
  result exceeded the `0.60` target;
- strict MkDocs build and `git diff --check`: passed.

Linux GCC/Clang, macOS arm64, Unix stress, sanitizer, and glibc allocation
evidence for the Phase-8 delta remains pending until the updated workflow
runs remotely on a committed revision. See
[Phase 8 Validation](phase8-validation.md) for the exact local results.

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
| Phase-8 scaling | Linux CPU, `T={1,32,1000}`, `d={10^4,10^5}` plus `T=1,d=10^6` | Exact cross-thread results and at least 60% efficiency on four threads for workloads of at least 320,000 transformed elements |

## Instrumentation

The process-local runtime exposes private counters for worker starts, submitted
tasks, and peak queued tasks. Student kernels additionally report workspace
growth and peak workspace bytes. A CI-only `LD_PRELOAD` allocation probe counts
glibc allocation calls and requested bytes around a native call; it is not
compiled into or distributed with wheels.

The binary audit uses `ldd` on Linux, `otool -L` on macOS, and an internal PE
import-table parser on Windows. Each report records the extension SHA-256,
platform, Python version, toolchain, and dependency list.

The Phase-8 allocation gate also compares preparation at `d=10^4` and
`d=10^5`; allocation count must remain tile-bounded rather than scaling per
dimension element. The lifecycle child executes the new dimension-parallel
preparation path, and sanitizer jobs include the complete prepared-data suite.

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
