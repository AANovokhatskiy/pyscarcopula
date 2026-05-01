# R-vine Conditional Sampling

This note documents the implementation contract for
`RVineCopula.predict(..., given=...)`.

## Current Status

R-vine conditional prediction has two execution paths:

- **suffix exact path** for conditioning sets that can be placed at the end of
  the R-vine variable order;
- **runtime DAG + MCMC path** for arbitrary non-suffix conditioning sets.

The suffix path is exact for the fitted pair-copula construction. The DAG +
MCMC path is general and targets the full fitted R-vine density with the fixed
variables held constant, but it is approximate and more expensive.

All `given` values are pseudo-observations in `(0, 1)`.

## Public API

```python
import numpy as np
from pyscarcopula import PredictConfig, RVineCopula

vine = RVineCopula().fit(u, method="mle")

samples = vine.predict(
    n=5000,
    u_train=u,
    given={0: 0.4, 3: 0.8},
    rng=np.random.default_rng(2026),
)

cfg = PredictConfig(
    given={0: 0.4, 3: 0.8},
    horizon="next",
    dynamic_conditioning="given_only",
    mcmc_steps=300,
    mcmc_burnin=100,
    return_diagnostics=True,
)

samples, diagnostics = vine.predict(
    5000,
    u_train=u,
    predict_config=cfg,
    rng=np.random.default_rng(2027),
)
```

Important fields:

- `given`: predict-time fixed pseudo-observation values;
- `given_vars`: fit-time target set for suffix-compatible exact structures;
- `horizon`: `current` or `next` dynamic edge state;
- `dynamic_conditioning`: whether fixed suffix observations update dynamic
  edge states;
- `mcmc_steps` and `mcmc_burnin`: componentwise Metropolis controls for the
  arbitrary DAG + MCMC fallback;
- `return_diagnostics`: whether to return execution diagnostics.

The mathematical meaning of these fields is documented in
`docs/guide/prediction-semantics.md`.

## Suffix Exact Path

A `given` set uses this path when one of these conditions holds:

- the fixed variables are already the last entries of the fitted R-vine
  variable order, read from the anti-diagonal of the natural-order matrix;
- the fitted tree structure can be rebuilt into an equivalent natural-order
  matrix whose variable order has those fixed variables last;
- all variables are fixed, in which case `predict` returns constant rows.

The exact sampler works in two phases:

1. Validate `given` and find a natural-order matrix whose variable order ends
   with the fixed variables. This is either the fitted matrix itself or a
   rebuilt matrix with remapped pair-copula positions.
2. Propagate fixed trailing-order pseudo-observations through h-functions, then
   sample remaining variables by inverse h-functions using the natural-order
   recursion.

Relevant implementation:

- `pyscarcopula/vine/rvine.py`
- `RVineCopula._given_suffix_start_col`
- `RVineCopula._suffix_sampling_state`
- `RVineCopula._find_peel_order_for_given_suffix`
- `RVineCopula._sample_suffix_given_with_r`
- `pyscarcopula/vine/_conditional_rvine.py`
- `pyscarcopula/vine/_rvine_matrix_builder.py`

## Runtime DAG + MCMC Path

If the suffix path is not available, `RVineCopula.predict` builds a runtime
DAG from the fitted edge map.

The DAG planner marks fixed base variables as known, propagates available
h-function nodes, and inserts inverse-h sampling actions for free base
variables when enough conditioning information is available. This produces a
feasible initial sample for arbitrary `given`.

When several inverse chains are feasible, the planner uses a deterministic
R-vine-geometry tie-break: prefer the deepest tree level, then the rightmost
natural-order matrix column, and use the partner variable id only as a final
stable tie-break. This avoids making the initializer depend primarily on
variable numbering.

The initializer alone is not a full posterior sampler for all arbitrary
conditioning patterns. Therefore the implementation refines the initialized
rows with MCMC over the free variables, targeting the full fitted vine copula
density under the fixed `given` values.

Relevant implementation:

- `pyscarcopula/vine/_rvine_dag.py`
- `build_runtime_rvine_dag`
- `plan_conditional_sample`
- `execute_conditional_plan`
- `RVineCopula._sample_arbitrary_given_mcmc`
- `RVineCopula._log_pdf_rows_with_r`

Diagnostics use `conditional_method='dag_mcmc'` and include `dag_steps`,
`dag_edges_used`, and `mcmc` acceptance statistics. The MCMC block reports
burn-in steps, retained update steps, total updates, per-variable acceptance
rates, acceptance summaries, and a `low_acceptance_warning` flag.

## Fit-Time `given_vars`

`RVineCopula.fit(..., given_vars=...)` is a structure-selection target. It is
not a prediction value. The builder prioritizes structures that can support
the target set through the suffix exact path.

Fit-time controls:

- `conditional_strict=True`: reject unsupported exact structures;
- `conditional_strict=False`: keep the best available structure;
- `structure_search='beam'`: per-tree candidate-path search;
- `beam_width=<positive int>`: beam pruning width;
- `structure_search='multi-start'`: smaller whole-structure search.

With non-strict fitting, prediction for the target set may still succeed via
the runtime DAG + MCMC fallback.

Fit diagnostics are stored in `RVineCopula.fit_diagnostics`.

## Dynamic Edge Semantics

Dynamic edges first form a predictive state from the fitted history:

- MLE: point state with constant parameter;
- GAS: point state from the score recursion;
- SCAR-TM: grid distribution over the latent state.

`horizon='current'` uses the fitted end-of-sample state. `horizon='next'`
uses the one-step-ahead predictive state.

With `dynamic_conditioning='ignore'`, these states are not updated by
prediction-time `given` values. With `dynamic_conditioning='given_only'`,
supported fixed suffix observations update GAS and SCAR-TM states before
downstream sampling. Updates are exposed through `updated_edges` and
`skipped_edges` diagnostics.

Dynamic conditioning is currently applied on the suffix exact path, where the
fixed pseudo-observations have a deterministic update order.

GAS dynamic conditioning is deliberately strict. A GAS score update advances
the deterministic filter. Therefore `given_only` updates GAS edges only when
`horizon='current'`. With `horizon='next'`, the predictive state has already
been advanced one score step; applying `condition_state` again would produce
the next filter state, not a posterior update of the same forecast state. Such
edges are skipped with reason `gas_next_horizon_would_advance_filter`.

## Verification

Focused checks:

```powershell
pytest tests\test_rvine_copula.py -q
pytest tests\test_conditional_sampling_plan.py --run-validation -q
pytest tests\test_api.py tests\test_types.py -q
```

The full suite should also pass:

```powershell
pytest
```
