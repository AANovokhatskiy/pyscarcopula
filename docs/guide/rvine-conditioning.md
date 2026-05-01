# R-vine Conditioning

## Overview

`RVineCopula.predict(..., given=...)` supports conditional sampling in
pseudo-observation space. The fixed variables are passed as
`{var_index: u_value}` and the returned sample keeps those columns fixed.

There are two R-vine conditional paths:

- **suffix exact path**: used when the fixed variables can be placed at the end
  of the R-vine variable order;
- **runtime DAG + MCMC path**: used for arbitrary non-suffix `given` sets.

The suffix path is fast and exact for the fitted pair-copula construction. The
DAG + MCMC path is general, but approximate and more expensive.

For the mathematical meaning of `given`, `given_vars`, `horizon`, predictive
sampling, and dynamic conditioning, see
[Prediction Semantics](prediction-semantics.md).

## Suffix Exact Path

The simplest exact case is to condition on the last variables in the fitted
R-vine order. The order is read from the anti-diagonal of the natural-order
matrix:

```python
import numpy as np
from pyscarcopula import RVineCopula

vine = RVineCopula().fit(u, method="mle")

variable_order = [
    int(vine.matrix[vine.d - 1 - col, col])
    for col in range(vine.d)
]

samples = vine.predict(
    n=5000,
    u_train=u,
    given={variable_order[-1]: 0.6},
    rng=np.random.default_rng(2026),
)
```

Some sets that are not trailing in the fitted matrix still use the exact path
because the fitted tree structure can be rebuilt into an equivalent
natural-order matrix with those variables last.

## Arbitrary `given`: DAG + MCMC

If a `given` set cannot be handled by the suffix exact path, `predict` builds a
runtime DAG from the fitted R-vine edges. The DAG determines which
h-functions and inverse h-functions are available from the fixed nodes and
uses them to construct an initial conditional sample.

That initial sample is then refined by MCMC targeting the full fitted R-vine
density with the fixed variables held constant:

```python
from pyscarcopula import PredictConfig

cfg = PredictConfig(
    given={variable_order[0]: 0.45},
    horizon="next",
    mcmc_steps=300,
    mcmc_burnin=100,
    return_diagnostics=True,
)

samples, diagnostics = vine.predict(
    n=5000,
    u_train=u,
    predict_config=cfg,
    rng=np.random.default_rng(2027),
)

assert diagnostics["conditional_method"] in {"suffix", "dag_mcmc"}
```

When `conditional_method == "dag_mcmc"`, diagnostics include:

- `dag_steps`: runtime DAG action records;
- `dag_edges_used`: fitted R-vine edges used by the DAG initializer;
- `mcmc`: proposed moves, accepted moves, acceptance rate, and step count.

This fallback supports arbitrary conditioning, but it is more expensive than
suffix sampling. Prefer suffix-compatible structures for high-volume
production paths when possible.

For strongly dependent or high-dimensional conditional targets, increase
`mcmc_steps` and inspect `diagnostics["mcmc"]["low_acceptance_warning"]`.

## Fit-Time Targeting with `given_vars`

If the conditioning set is known before fit, pass it into `RVineCopula.fit`:

```python
vine = RVineCopula().fit(
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
- `conditional_mode='suffix'`: exact fit-time support contract;
- `structure_search='beam'`: search over per-tree builder-mode paths;
- `beam_width=4`: number of partial candidates kept per tree level;
- `structure_search='multi-start'`: smaller whole-structure candidate search.

The default fit-time search is `structure_search='beam'`.

With `conditional_strict=False`, fit may keep a structure that is not exact for
the target. Prediction can still use the arbitrary DAG + MCMC fallback.

## Dynamic Conditioning

Dynamic conditioning is separate from ordinary conditional sampling. Ordinary
conditioning fixes columns in the output sample. Dynamic conditioning updates
supported time-varying edge states using fixed prediction-time observations.

```python
samples, diagnostics = vine.predict(
    n=5000,
    u_train=u,
    given={variable_order[-1]: 0.6},
    dynamic_conditioning="given_only",
    return_diagnostics=True,
    rng=np.random.default_rng(2028),
)
```

Current modes:

- `dynamic_conditioning='ignore'`: default; predict edge parameters from the
  training data only;
- `dynamic_conditioning='given_only'`: use fixed suffix observations to update
  supported GAS and SCAR-TM edge states before downstream sampling.

For R-vines, dynamic conditioning is applied on the suffix exact path where
fixed pseudo-observations have a deterministic propagation order. Diagnostics
record `updated_edges` and `skipped_edges`.

For GAS edges, `given_only` updates only under `horizon='current'`. Under
`horizon='next'`, updating would perform an extra score-recursion step rather
than condition the same predictive state, so the edge is skipped with reason
`gas_next_horizon_would_advance_filter`.

## Diagnostics

After `fit`, structure diagnostics are available via `vine.fit_diagnostics`.
They include the target `given_vars`, selected candidate, candidate-level
reachability statistics, and for beam search the selected per-tree
`mode_path`.

At prediction time:

```python
samples, diagnostics = vine.predict(
    n=5000,
    u_train=u,
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

## Practical Guidance

Use `given_vars` during fit when the production conditioning set is known and
must be fast. Use direct `given` at prediction time for ad hoc conditioning.

Use `return_diagnostics=True` when validating a new conditional workflow. It
shows whether prediction used the exact suffix path or the arbitrary DAG +
MCMC fallback.

Use a fresh `np.random.default_rng(seed)` for reproducible conditional samples.
Reusing the same generator object advances the stream and intentionally
produces different draws.
