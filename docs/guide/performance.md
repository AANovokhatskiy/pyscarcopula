# Performance Tuning

Tune performance only after selecting the statistical model and fitting its
default configuration. Fit diagnostics should identify whether time is spent
in optimization, latent-state integration, high-dimensional emissions, or
repeated independent fits.

The complete parameter tables and backend algorithms are in
[Numerical Backends](numerical-backends.md). CPU ownership, deterministic
threading, and oversubscription rules are in
[CPU Parallelism](parallelism.md).

## Decision Guide

| Observed condition | First action | Detailed reference |
|---|---|---|
| Optimizer stops at its evaluation limit | Increase the method-specific `maxfun` after checking that the initial point and data are valid | [Optimizer controls](numerical-backends.md#bivariate-models) |
| GAS fit is sensitive to finite-difference settings | Use `scaling="unit"` as the baseline and inspect `ftol`, `gtol`, and `eps` | [GAS](numerical-backends.md#gas) |
| SCAR-TM-OU reports a narrow transition kernel | Leave `transition_method="auto"` so it can select the local path | [OU transfer methods](numerical-backends.md#transfer-methods) |
| SCAR-TM-OU spectral evaluation fails | Inspect the recorded spectral-to-matrix and matrix-to-local fallbacks before forcing a backend | [Spectral Hermite likelihood](numerical-backends.md#spectral-hermite-likelihood) |
| SCAR-TM-JACOBI reports negative spectral mass or invalid row sums | Leave `transition_method="auto"` or compare against `local`; do not raise basis order without checking memory | [Jacobi transfer methods](numerical-backends.md#jacobi-transfer-methods) |
| A multivariate result would allocate a dense `d*d` matrix | Select equicorrelation or `corr_mode="factor"` according to the model assumptions | [Multivariate native paths](numerical-backends.md#multivariate-native-paths) |
| Full static correlation optimization grows too quickly | Use `shrinkage`; use `cholesky` only for small `d`, or factor mode for a justified low-rank model | [Multivariate native paths](numerical-backends.md#multivariate-native-paths) |
| Only a fast static baseline is needed | Keep the default `corr_mode="fixed"`; inspect `corr_estimator` to distinguish supplied and plug-in correlation | [Estimation methods](estimation-methods.md#mle) |
| Output sampling exceeds the memory budget | Use the model's batch iterator and set `batch_rows` or `memory_budget_bytes` | [Multivariate native paths](numerical-backends.md#multivariate-native-paths) |
| Many independent fits dominate runtime | Use process-level `n_jobs`; keep per-fit `n_threads=1` unless measured otherwise | [Independent fit parallelism](numerical-backends.md#independent-fit-parallelism) |
| Vine fitting attempts too many dynamic edges | Set `truncation_level` or `min_edge_logL` from the modelling requirement | [Generic VineCopula](numerical-backends.md#generic-vinecopula) |

## Minimal Configuration

`NumericalConfig` bundles native thread ownership and method-specific optimizer
settings:

```python
import numpy as np

from pyscarcopula import GumbelCopula, LBFGSBConfig, NumericalConfig
from pyscarcopula.api import fit

source = GumbelCopula()
u = source.sample_at_parameter(
    200,
    np.full(200, 1.5),
    rng=np.random.default_rng(2026),
)
copula = GumbelCopula()
config = NumericalConfig(
    n_threads=1,
    mle_optimizer=LBFGSBConfig(maxiter=200, maxfun=500),
)
result = fit(copula, u, method="mle", config=config)
```

Direct strategy keyword arguments override the corresponding configuration
values for that call. Keep those overrides local to an experiment so the
baseline remains reproducible.

## Measurement Rules

1. Compare configurations on the same data, initial point, and thread count.
2. Record both wall time and fit diagnostics; optimizer success alone does not
   establish numerical agreement.
3. For stochastic sampling comparisons, create a fresh
   `np.random.default_rng(seed)` for each run.
4. Compare log-likelihoods and parameter estimates before accepting a faster
   backend.
5. Set explicit memory budgets before allocating outputs whose size scales as
   `T*K`, `n*d`, or `d*d`.

For all supported keys, defaults, fallback order, and memory formulas, continue
with [Numerical Backends](numerical-backends.md).
