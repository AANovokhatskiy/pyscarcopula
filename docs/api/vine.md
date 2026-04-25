# Vine API

Both `CVineCopula.predict(...)` and `RVineCopula.predict(...)` support:

- `given={var_index: u_value}` for conditional generation;
- `horizon='current'|'next'` for dynamic edge prediction;
- `predict_config=PredictConfig(...)` as an explicit options object;
- `rng=np.random.default_rng(seed)` for reproducible Monte Carlo output.

`RVineCopula.predict(...)` also supports:

- `dynamic_conditioning='ignore'|'given_only'`;
- `mcmc_steps=<non-negative int>` and `mcmc_burnin=<non-negative int>` for
  arbitrary `dag_mcmc` conditioning;
- `return_diagnostics=True`.

`RVineCopula.fit(...)` supports:

- `given_vars=[...]` to target a known conditioning set at fit time;
- `conditional_strict=True|False` to reject or keep fitted structures that are
  not suffix-compatible for `given_vars`;
- `conditional_mode='suffix'` for the exact fit-time support contract;
- `structure_search='beam'|'multi-start'` to control fit-time structure search;
- `beam_width=<positive int>` to control beam-search width for `given_vars`.

## Prediction Semantics

`predict` is a forecasting API. It samples from the fitted copula conditional
on the training data. `given` additionally fixes selected components of the
next pseudo-observation. `dynamic_conditioning` optionally lets those fixed
components update supported dynamic edge states before the remaining
components are sampled.

See [Prediction Semantics](../guide/prediction-semantics.md) for the
mathematical contract.

## R-vine Conditional Paths

`RVineCopula.predict(..., given=...)` uses the suffix exact path when the fixed
variables can be placed at the end of the R-vine variable order. This order is
read from the anti-diagonal of the natural-order matrix. The fixed variables
must already be last in the fitted matrix, or the same fitted tree structure
must be rebuildable into an equivalent natural-order matrix where they are
last.

If the `given` set is not suffix-compatible, `predict` uses the arbitrary
runtime DAG + MCMC fallback. The DAG builds an h/inverse-h execution plan from
available nodes; MCMC then targets the full fitted R-vine density with fixed
`given` values.

`mcmc_steps` and `mcmc_burnin` tune only this fallback. Diagnostics include
acceptance-rate summaries and `low_acceptance_warning`.

With `return_diagnostics=True`, `predict` returns `(samples, diagnostics)`.
The `diagnostics["conditional_method"]` value is one of:

- `unconditional`;
- `suffix`;
- `dag_mcmc`.

When the model was fitted with `given_vars=...`, that set becomes the
advertised exact-support target. With `conditional_strict=True`, fit rejects
unsupported structures. With `conditional_strict=False`, fit may keep the
structure and prediction can still use the DAG + MCMC fallback.

Fit-time diagnostics are available through `vine.fit_diagnostics`. The
selection block includes the chosen candidate, all scored candidates, and for
beam search the selected per-tree `mode_path`.

## CVineCopula

::: pyscarcopula.vine.cvine.CVineCopula
    options:
      members:
        - fit
        - log_likelihood
        - sample
        - predict
        - summary

## RVineCopula

::: pyscarcopula.vine.rvine.RVineCopula
    options:
      members:
        - fit
        - log_likelihood
        - sample
        - predict
        - summary

## PredictConfig

::: pyscarcopula._types.PredictConfig

## RVineMatrix

::: pyscarcopula.vine._structure.RVineMatrix
    options:
      members:
        - edge
        - edges_at_tree
        - n_trees
        - n_edges

## VineEdge

::: pyscarcopula.vine._edge.VineEdge
    options:
      members: false

## select_best_copula

::: pyscarcopula.vine._selection.select_best_copula
