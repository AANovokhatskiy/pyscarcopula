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

All `given` values are pseudo-observations in $(0, 1)$.

## Public API

```python
import numpy as np
from pyscarcopula import PredictConfig, RVineCopula

vine = RVineCopula().fit(u, method="mle")

samples = vine.predict(
    n=5000,
    u=u,
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
    u=u,
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
- `pyscarcopula/vine/_rvine_suffix.py`
- `given_suffix_start_col`
- `suffix_sampling_state`
- `sample_suffix_given_with_r`
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
- `pyscarcopula/vine/_rvine_conditional_runtime.py`
- `sample_arbitrary_given_mcmc`
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

Dynamic conditioning is a predict-time refinement of already fitted edge
models. RVine orchestration must not inspect dynamic model formulas. It uses
only the strategy-facing contract exposed by fitted edge results:

- `predictive_params(copula, u_train_pair, result, n, ...)` builds the initial
  predictive parameter samples for each edge;
- `predictive_state(copula, u_train_pair, result, ...)` builds a reusable
  state when a fixed suffix observation may further condition that edge;
- `condition_state(copula, state, u_observed_pair, result)` applies the
  fully observed edge pair;
- `sample_params(copula, state, n, ...)` converts the conditioned state back
  to a parameter vector;
- `model_sample_state(copula, result)` marks observation-driven edges whose
  model-reproduction sampling must be updated step by step.

`horizon='current'` uses the fitted end-of-sample state. `horizon='next'`
uses the one-step-ahead predictive state. Static edges ignore this option.

With `dynamic_conditioning='ignore'`, these states are not updated by
prediction-time `given` values. With `dynamic_conditioning='given_only'`,
supported fixed suffix observations update strategy-owned dynamic states
before downstream sampling. Updates are exposed through `updated_edges` and
`skipped_edges` diagnostics.

### Execution Modes

- **No `given`**: `predict` draws all edge parameter vectors through
  `_predict_r_for_edges` and samples unconditionally. Dynamic conditioning does
  not run.
- **All variables fixed**: `predict` returns constant rows and, when requested,
  empty `updated_edges` / `skipped_edges` diagnostics.
- **Suffix exact path + `ignore`**: fixed suffix variables are used by the
  exact sampler, but they do not update dynamic edge states.
- **Suffix exact path + `given_only`**: fixed suffix variables are propagated in
  matrix order. Whenever an edge pair is fully observed before free variables
  are sampled, RVine calls the strategy state contract above and replaces that
  edge's predictive parameter vector if conditioning changes the state.
- **DAG + MCMC fallback + `ignore`**: arbitrary non-suffix `given` values use
  the DAG initializer and MCMC without dynamic conditioning.
- **DAG + MCMC fallback + `given_only`**: no partial dynamic updates are
  applied. Eligible dynamic edges are reported as skipped with reason
  `dag_mcmc_not_suffix_supported`.

### Skip Reasons

Current skip reasons are part of the diagnostics contract:

- `next_horizon_would_advance_filter`: the edge has a strategy-owned
  stepwise model state and `horizon='next'`; applying another observation would
  advance the filter rather than condition the same predictive state.
- `no_training_history`: the edge needs fitted-history pseudo-observations to
  construct a predictive state, but no `u_train` / stored fit data is
  available.
- `unsupported_or_noop`: the edge is dynamic but the strategy state did not
  change, or the edge has no supported dynamic conditioning action.
- `dag_mcmc_not_suffix_supported`: `given_only` was requested for a non-suffix
  conditioning set that uses the DAG + MCMC fallback.

### Diagnostics Contract

When `return_diagnostics=True`, dynamic conditioning diagnostics preserve this
shape:

- top-level fields: `given`, `dynamic_conditioning`, `suffix_start_col`,
  `matrix_rebuilt`, `conditional_method`, `updated_edges`, `skipped_edges`;
- update/skip records: `key`, `tree`, `col`, `conditioned`, `conditioning`,
  `method`, `family`, `status`;
- optional record fields: `reason`, `r_before_mean`, `r_after_mean`;
- DAG + MCMC diagnostics additionally include `dag_steps`, `dag_edges_used`
  and `mcmc`.

### Regression Coverage

The current behaviour is covered mainly by `tests/test_rvine_copula.py`:

- `test_dynamic_conditioning_ignore_matches_default`
- `test_api_predict_forwards_dynamic_conditioning_to_rvine`
- `test_given_only_*_not_fully_observed`
- `test_predictive_given_only_reweights_grid_by_observed_pair_likelihood`
- `test_predictive_given_only_noop_detects_equal_prob_copy`
- `test_predictive_state_cache_reused_for_given_only`
- `test_dynamic_conditioning_return_diagnostics_lists_updated_edges`
- `test_stateful_given_only_skips_next_horizon_to_avoid_double_advance`
- `test_dynamic_conditioning_multi_edge_order_updates_conditional_predictive_state`
- `test_dynamic_conditioning_mixed_vine_diagnostics`
- `test_given_only_reports_skip_for_dag_mcmc_path`
- `test_predict_config_return_diagnostics_via_api`

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
