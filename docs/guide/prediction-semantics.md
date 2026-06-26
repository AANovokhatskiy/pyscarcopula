# Prediction Semantics

This page defines the prediction-time terms used by `predict`,
`PredictConfig`, bivariate copulas, and vine copulas.

## Three Different Questions

The library separates three questions that are easy to mix up:

- **Predictive sampling** asks what the next copula observation may look like
  after fitting on historical data.
- **Conditional sampling** fixes some components of that next observation,
  for example $U_2 = 0.7$, and samples the remaining components.
- **Dynamic conditioning** optionally updates time-varying edge states using
  fixed prediction-time values before the remaining components are sampled.

In notation, after training data $D_T = \{u_1, \ldots, u_T\}$, plain prediction
draws from

$$
U_* \sim C(\cdot \mid D_T).
$$

Conditional prediction with `given={j: a}` draws from

$$
U_{*, -G} \sim C(U_{*, -G} \mid U_{*, G}=a, D_T),
$$

where $G$ is the set of fixed variables.

Dynamic conditioning changes the state used to form the copula parameters:

$$
p(x_* \mid D_T)
    \quad\rightarrow\quad
p(x_* \mid D_T, U_{*, G}=a)
$$

for supported dynamic edges. This is off by default.

The mathematical difference between filtered point states, posterior latent
states, and one-step predictive latent states is summarized in
[Mathematical Contracts](mathematical-contracts.md).

## `predict` vs `sample`

Sampling names have the same meaning across model types:

- `pyscarcopula.api.sample(copula, data, result, n)` reproduces a fitted
  bivariate or multivariate model;
- fitted bivariate, multivariate, C-vine, and R-vine objects expose the same
  operation as `model.sample(n, u=None, rng=None)`;
- models with a scalar dependence parameter expose low-level generation as
  `model.sample_at_parameter(n, r, rng=None)`.

For stochastic fitted models, reproduction simulates a new latent or
score-driven path.

`predict(n, u=training_data)` on a fitted object generates forecast
observations conditional on the supplied history. Omitting `u` uses the
history stored by the last `fit`. The stateless equivalent is
`pyscarcopula.api.predict(copula, data, result, n)`. For MLE this uses the same
constant-parameter copula. Dynamic strategies such as GAS and SCAR-TM use
their fitted time-varying state.

Most sampling APIs accept `rng=np.random.default_rng(seed)`. Use a fresh
generator with the same seed for exact reproducibility; reusing a generator
object advances its stream.

## `horizon`

`horizon` selects which dynamic state is used before prediction:

- `horizon='current'` uses the filtered or posterior state at the end of the
  observed sample, conceptually time $T$.
- `horizon='next'` uses the one-step-ahead predictive state, conceptually
  time $T+1$.

For SCAR-TM, `current` means the posterior latent state after the observed
sample, for example $p(x_T \mid D_T)$ in the OU model or
$p(\tau_T \mid D_T)$ in the Jacobi model. `next` means the corresponding
one-step-ahead state, such as $p(x_{T+1} \mid D_T)$ or
$p(\tau_{T+1} \mid D_T)$. For GAS, `current` uses the last filtered score state
and `next` applies the one-step score recursion. For MLE there is no dynamic
state, so the two horizons are equivalent.

The default is `horizon='next'`, because `predict` is primarily a forecasting
API.

## `given`

`given` is a predict-time conditioning value in pseudo-observation space:

```python
samples = model.predict(
    n=10_000,
    u=u_train,
    given={2: 0.7},
    rng=np.random.default_rng(2026),
)
```

Keys are zero-based variable indices. Values must be in $(0, 1)$. The returned
sample keeps fixed columns equal to the supplied values and samples the
remaining columns from the fitted conditional copula.

For bivariate copulas, `given` can fix variable `0`, variable `1`, or both.

For C-vines, prediction uses C-vine-specific prefix and general conditional
paths.

For R-vines, `predict` uses two paths:

- **suffix exact path**: used when the fixed variables can be placed at the end
  of the R-vine variable order, either directly or after rebuilding an
  equivalent natural-order matrix;
- **runtime DAG + MCMC path**: used for arbitrary non-suffix conditioning
  patterns. The DAG provides a feasible h/inverse-h initialization and MCMC
  refines samples against the full vine density with the fixed variables held
  constant.

The suffix path is exact and fast. The arbitrary path is general but
approximate and more expensive.

If `given` fixes every variable, prediction returns constant rows equal to the
supplied values.

## `given_vars`

`given_vars` is a fit-time structure-selection hint for `RVineCopula.fit`:

```python
vine = RVineCopula().fit(
    u_train,
    method="scar-tm-ou",
    given_vars=[0, 2],
)
```

It does not supply conditioning values. It says: "when building the R-vine,
prefer structures that make this set easy to condition on exactly."

With `conditional_strict=True`, fit rejects a structure that cannot support
the target set through the exact suffix sampler. With
`conditional_strict=False`, fit keeps the best available structure and
prediction uses the arbitrary DAG + MCMC path when the exact suffix path is
not available.

Use `given_vars` when the production conditioning set is known before fitting.
Use `given` when calling `predict`.

## `dynamic_conditioning`

`dynamic_conditioning` controls whether fixed prediction-time values update
dynamic edge states before sampling.

- `dynamic_conditioning='ignore'` is the default. Edge parameters are predicted
  from $D_T$ only, then conditional sampling treats `given` as fixed values in
  the copula recursion.
- `dynamic_conditioning='given_only'` lets supported fixed observations update
  strategy-owned predictive states before downstream edge parameters are
  sampled.

This is intentionally separate from ordinary conditional sampling. Conditional
sampling changes which variables are drawn. Dynamic conditioning changes the
parameter state used by dynamic edges.

For R-vines, dynamic conditioning is applied on the suffix exact path, where
fixed pseudo-observations can be propagated in a deterministic order through
the vine. Diagnostics report which edges were updated and which were skipped.

For stateful observation-driven edges, `given_only` is intentionally strict:
updates are applied only with `horizon='current'`. With `horizon='next'`, the
predictive state has already been advanced one step, so another
prediction-time update would advance the state again rather than condition the
same forecast state. Those edges are skipped with reason
`next_horizon_would_advance_filter`.

## `PredictConfig`

Prediction options can be passed as explicit kwargs or as a `PredictConfig`:

```python
import numpy as np
from pyscarcopula import PredictConfig

cfg = PredictConfig(
    given={2: 0.7},
    horizon="next",
    dynamic_conditioning="given_only",
    mcmc_steps=300,
    mcmc_burnin=100,
    return_diagnostics=True,
)

samples, diagnostics = vine.predict(
    10_000,
    u=u_train,
    predict_config=cfg,
    rng=np.random.default_rng(2027),
)
```

Explicit kwargs override the corresponding fields in `PredictConfig`, so
call-site options and reusable configuration objects can be mixed deliberately.

`mcmc_steps` and `mcmc_burnin` apply only to arbitrary R-vine conditioning
when `conditional_method='dag_mcmc'`. They control the number of componentwise
Metropolis updates after and before burn-in. They do not affect suffix exact
sampling.

## Diagnostics

For `RVineCopula.predict(..., return_diagnostics=True)`, the result is
`(samples, diagnostics)`.

Useful diagnostic fields include:

- `conditional_method`: `unconditional`, `suffix`, or `dag_mcmc`;
- `given`: normalized fixed values;
- `dynamic_conditioning`: active dynamic-conditioning mode;
- `updated_edges` and `skipped_edges`: dynamic-conditioning edge records;
- `dag_steps`, `dag_edges_used`, and `mcmc`: arbitrary-conditioning details
  when the DAG + MCMC fallback is used.

The `mcmc` block includes per-variable acceptance rates, summary acceptance
statistics, burn-in count, total update count, and a `low_acceptance_warning`
flag. Low acceptance means the fallback may need more steps or a stronger
proposal strategy.

Common dynamic-conditioning skip reasons include:

- `next_horizon_would_advance_filter`: `given_only` was requested with
  `horizon='next'` for a stateful edge;
- `no_training_history`: the edge needs fitted history to build a predictive
  state;
- `unsupported_or_noop`: the edge has no supported update or the update leaves
  the state unchanged;
- `dag_mcmc_not_suffix_supported`: `given_only` was requested for an arbitrary
  DAG + MCMC conditioning path.
