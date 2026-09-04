# Vine Numerical Execution

## Generic VineCopula

`VineCopula.fit` has an explicit `config` argument and forwards strategy
options through the shared edge-fitting core. With `structure=None`, the
Dissmann selector builds the structure; with a fixed `RVineMatrix`, fitting
starts directly from decoded trees and skips MST selection.

```python
from pyscarcopula import VineCopula
from pyscarcopula import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=3000, maxiter=3000))

vine = VineCopula(
    truncation_level=2,
    truncation_fill='independent',
    threshold=0.02,
    min_edge_logL=5.0,
)

vine.fit(
    u,
    method='gas',
    config=cfg,
    gamma_bound=30.0,
    ftol=1e-12,
)
```

Strategy-specific optimizer and numerical options are forwarded to every
non-independent, non-truncated edge selected for dynamic fitting. R-vine
structure controls are:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | instance value / `None` | Tree levels $\ge$ `truncation_level` are truncated. |
| `truncation_fill` | `'independent'` | Truncated trees become independent edges or MLE-only edges (`'mle'`). |
| `threshold` | `0.0` | Edges with $\lvert\text{Kendall tau}\rvert < \texttt{threshold}$ are made independent before fitting. |
| `min_edge_logL` | `None` | Fitted weak edges below the threshold are replaced by independence. |
| `structure_search` | `'beam'` | Conditional-structure search mode when `given_vars` is used. |
| `beam_width` | `4` | Number of partial candidate structures retained by beam search. |
| `transform_type` | instance value / `'softplus'` | Parameter transform used for Archimedean candidate copulas. |

For every vine structure, automatic family selection is MLE-based. `gtol`,
`ftol`, `gamma_bound`, `K`, and similar strategy controls affect the dynamic
edge refit after a family has been selected. If `method='gas'`, a too-loose
`ftol` can make some edges stop early with `success=True`; set `ftol=1e-12` and increase
`maxfun` for difficult edges.

Fitted generic vines use sequential native hot paths where the model contract permits
them. GAS unconditional sampling executes the row recursion and causal score
updates in one native call while preserving RNG and edge-update order. Repeated
SCAR-TM-OU prediction against unchanged fitted history reuses
pseudo-observations and terminal posterior state. A new `fit`, a changed
explicit history, or an edge replacement invalidates the relevant transient
cache. These optimizations do not parallelize edge or sample execution.
The Python reference sampler and native GAS executor consume the same
model-independent R-vine traversal plan, so matrix order, conditioned nodes,
and edge orientation have one authoritative representation.

The regular-vine native capability matrix is deliberately narrower than the
public `VineCopula` API:

| Operation | Native capability | Production fallback |
|-----------|-------------------|---------------------|
| Static unconditional sampling | Fixed C-, D-, or R-vine; exact built-in Independent, Clayton, Gumbel, Joe, Frank, or bivariate Gaussian edges; scalar parameters | Unsupported exact-type combinations are rejected |
| Suffix/DAG conditional execution | The same exact built-in families, including supported rotations/orientations and scalar or row-specific parameter paths | Unsupported active edges are rejected |
| Row log-density and conditional MCMC | The same exact built-in families with scalar, row-specific, or mixed parameter storage; MCMC selects bounded incremental or full recomputation before consuming RNG | Unsupported active edges are rejected |
| R-vine Rosenblatt and GoF | Supported static and dynamic fitted edges from the exact built-in family set | Unsupported exact-type combinations are rejected |

Support is exact-type only. Unknown subclasses cannot opt in through class
flags or similarly named Python methods. Validation or numerical failures
after entering a supported native call are reported and never retried in
Python.

`gof_test(..., bootstrap=True)` also supports fitted `VineCopula` models. Each
replication simulates from the captured fitted vine, optionally refits an
independent worker-owned vine with the same structure and fitting contract,
then evaluates the R-vine Rosenblatt transform and Cramer-von Mises statistic.
Independent `SeedSequence` streams make the bootstrap statistics deterministic
across `n_jobs`; omitted native thread settings resolve to one thread per
worker.

Static `VineCopula` sampling bounds temporary vectorized workspace by processing at
most 8192 rows at a time. Use `sample(..., batch_rows=...)` to trade throughput
against peak memory. `memory_budget_bytes=` checks the estimated output and
workspace requirement before allocation. Dynamic edge trajectories are not
split because their row order is part of the fitted time-series model.

Arbitrary conditional MCMC checks its complete adapter, binding, state,
log-density, node-cache, contribution-cache, and replay-draw footprint against
the internal memory budget before allocation or random-number consumption.
Draws are validated and transferred in bounded chunks. Empty batches retain
the same diagnostics schema without undefined acceptance-rate arithmetic.

When benchmarking dynamic vines, report `vine.fit_result.actual_methods` and
`vine.fit_result.fallback_count`: unsuccessful dynamic edge fits are retained
as MLE fallbacks and otherwise make GAS or SCAR timings look artificially fast.
For prediction caches, measure cold and warm calls separately.
