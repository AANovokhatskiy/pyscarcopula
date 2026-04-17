# Vine API

Both `CVineCopula.predict(...)` and `RVineCopula.predict(...)` support:

- `given={var_index: u_value}` for conditional generation
- `horizon='current'|'next'` for SCAR-TM-OU edge mixtures

`RVineCopula.predict(...)` also accepts `quad_order` for arbitrary
conditional generation. Smaller values are faster but use coarser numerical
integration for non-prefix `given` sets. When several latent variables must
be conditioned on future fixed coordinates, RVine prediction uses a joint
posterior grid; the grid resolution is controlled by `quad_order`.
Use `conditional_method='auto'` for the default behavior,
`conditional_method='graph'` to require the no-posterior or flexible graph
path, `conditional_method='grid'` to require the joint-grid fast path, or
`conditional_method='exact'` to force recursive quadrature.

`RVineCopula(structure_mode='conditional', conditional_vars={...})` can fit a
structure for a known future conditioning pattern. Use
`conditional_plan(given)` to inspect posterior dimension, grid size, and graph
feasibility before calling `predict`, and `flexible_graph_plan(given)` for
experimental graph reachability diagnostics.

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
        - conditional_plan
        - flexible_graph_plan
        - is_conditioning_optimized_for
        - summary

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
