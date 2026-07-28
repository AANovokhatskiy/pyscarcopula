# R-vine Conditioning

## Overview

`VineCopula.predict(..., given=...)` supports conditional sampling in
pseudo-observation space. The fixed variables are passed as
`{var_index: u_value}` and the returned sample keeps those columns fixed.

There are two R-vine conditional paths:

- **suffix exact path**: used when the fixed variables can be placed at the end
  of the R-vine variable order;
- **approximate fallback path**: used for arbitrary non-suffix `given` sets.

The suffix path is fast and exact for the fitted pair-copula construction. The
fallback path is general, but approximate and more expensive.

For the mathematical meaning of `given`, `given_vars`, `horizon`, predictive
sampling, and dynamic conditioning, see
[Prediction Semantics](prediction-semantics.md).

## Suffix Exact Path

The simplest exact case is to condition on the last variables in the fitted
R-vine order:

```python
import numpy as np
from pyscarcopula import VineCopula

TARGET_VARIABLE = 0
vine = VineCopula().fit(
    u,
    method="mle",
    given_vars=[TARGET_VARIABLE],
)

samples = vine.predict(
    n=5000,
    u=u,
    given={TARGET_VARIABLE: 0.6},
    rng=np.random.default_rng(2026),
)
```

Some sets that are not trailing in the fitted matrix still use the exact path
because the fitted tree structure can be rebuilt with those variables last.

If all variables are fixed, `predict` returns constant rows with the supplied
values.

## Arbitrary `given`: Approximate Fallback

If a `given` set cannot be handled by the suffix exact path, `predict` uses an
approximate fallback that supports arbitrary conditioning sets:

```python
from pyscarcopula import PredictConfig

cfg = PredictConfig(
    given={2: 0.45},
    horizon="next",
    mcmc_steps=300,
    mcmc_burnin=100,
    return_diagnostics=True,
)

samples, diagnostics = vine.predict(
    n=5000,
    u=u,
    predict_config=cfg,
    rng=np.random.default_rng(2027),
)

assert diagnostics["conditional_method"] in {"suffix", "dag_mcmc"}
```

When `conditional_method == "dag_mcmc"`, diagnostics include advanced details:

- `dag_steps`: initialization action records;
- `dag_edges_used`: fitted R-vine edges used by the initializer;
- `mcmc`: proposed moves, accepted moves, acceptance rate, and step count.

This fallback supports arbitrary conditioning but performs MCMC refinement.
For repeated predictions with the same conditioning variables, fit with
`given_vars` so the exact suffix path can avoid that per-call refinement.

For strongly dependent or high-dimensional conditional targets, increase
`mcmc_steps` and inspect `diagnostics["mcmc"]["low_acceptance_warning"]`.

## Fit-Time Targeting with `given_vars`

If the conditioning set is known before fit, pass it into `VineCopula.fit`:

```python
vine = VineCopula().fit(
    u,
    method="scar-tm-ou",
    given_vars=[0, 2],
)
```

`given_vars` changes structure selection. The builder prioritizes R-vine trees
that are compatible with the exact suffix sampler for that target set.

The public fit-time controls are:

- `given_vars=[...]`: target conditioning set;
- `conditional_strict=True`: reject fitted structures that cannot support the
  target through the exact suffix path;
- `conditional_mode='suffix'`: require exact fit-time support;
- `structure_search='beam'`: search over per-tree builder-mode paths;
- `beam_width=4`: number of partial candidates kept per tree level;
- `structure_search='multi-start'`: smaller whole-structure candidate search.

The default fit-time search is `structure_search='beam'`.

With `conditional_strict=False`, fit may keep a structure that is not exact for
the target. Prediction uses the approximate fallback when the exact suffix path
is not available.

## Dynamic Conditioning

Dynamic conditioning is separate from ordinary conditional sampling. Ordinary
conditioning fixes columns in the output sample. Dynamic conditioning updates
supported time-varying edge states using fixed prediction-time observations.

```python
samples, diagnostics = vine.predict(
    n=5000,
    u=u,
    given={variable_order[-1]: 0.6},
    dynamic_conditioning="given_only",
    return_diagnostics=True,
    rng=np.random.default_rng(2028),
)
```

Modes:

- `dynamic_conditioning='ignore'`: default; predict edge parameters from the
  training data only;
- `dynamic_conditioning='given_only'`: use fixed suffix observations to update
  supported strategy-owned dynamic edge states before downstream sampling.

For R-vines, dynamic conditioning is applied on the suffix exact path where
fixed pseudo-observations have a deterministic propagation order. Diagnostics
record `updated_edges` and `skipped_edges`.

Stateful observation-driven edges update only under `horizon='current'`.
Under `horizon='next'`, updating would perform an extra state-recursion step
rather than condition the same predictive state, so the edge is skipped with
reason `next_horizon_would_advance_filter`.

## Diagnostics

After `fit`, structure diagnostics are available via `vine.fit_diagnostics`.
They include the target `given_vars`, selected candidate, candidate-level
reachability statistics, and for beam search the selected per-tree
`mode_path`.

The `edge_fits` entry records the method that was requested and the methods
actually retained on fitted edges. A dynamic GAS or SCAR edge that does not
converge is replaced by its successful MLE family-selection result; this keeps
the existing fit contract but means the vine-level requested method alone does
not describe every edge. Inspect:

- `actual_methods` and `family_counts` for final edge counts;
- `dynamic_attempted_count` and `dynamic_success_count` for dynamic refits;
- `fallback_count` and `fallback_edges` for MLE replacements;
- `selection_nfev_total`, `dynamic_attempted_nfev_total`, and
  `fallback_discarded_nfev` for optimizer work;
- `edges[*].attempted_message` and `edges[*].timings_ms` for edge-level failure
  messages and selection, dynamic-fit, and total timings.

The compact `actual_methods`, `fallback_count`, and `fallback_edges` fields are
also copied to `vine.fit_result`, and `vine.summary()` prints the requested
method, actual method counts, and dynamic fallback count.

At prediction time:

```python
samples, diagnostics = vine.predict(
    n=5000,
    u=u,
    given={0: 0.4, 3: 0.8},
    return_diagnostics=True,
    rng=np.random.default_rng(2029),
)
```

Common fields:

- `conditional_method`: `unconditional`, `suffix`, or `dag_mcmc`;
- `given`: normalized fixed values;
- `dynamic_conditioning`: active dynamic-conditioning mode;
- `updated_edges` and `skipped_edges`: dynamic-conditioning records;
- `matrix_rebuilt`: whether the suffix exact path used a rebuilt matrix.

Common skip reasons:

- `next_horizon_would_advance_filter`: updating a stateful edge under
  `horizon='next'` would advance the filter one extra step;
- `no_training_history`: fitted history is unavailable for a dynamic edge;
- `unsupported_or_noop`: the edge has no supported update or the update leaves
  the state unchanged;
- `dag_mcmc_not_suffix_supported`: `given_only` was requested for the
  approximate fallback path.

## Practical Guidance

Use `given_vars` during fit when the conditioning indices are known in advance
and prediction must avoid MCMC refinement. Use direct `given` at prediction
time for ad hoc conditioning.

Use `return_diagnostics=True` when validating a new conditional workflow. It
shows whether prediction used the exact suffix path or the approximate
fallback.

Use a fresh `np.random.default_rng(seed)` for reproducible conditional samples.
Reusing the same generator object advances the stream and intentionally
produces different draws.
