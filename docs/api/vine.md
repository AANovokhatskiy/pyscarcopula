# Vine API

Both `CVineCopula.predict(...)` and `RVineCopula.predict(...)` support:

- `given={var_index: u_value}` for conditional generation
- `horizon='current'|'next'` for SCAR-TM-OU edge mixtures

`RVineCopula.predict(...)` supports conditional generation when the fixed
variables can be placed at the end of the R-vine variable order. This order is
read from the anti-diagonal of the natural-order matrix. The fixed variables
must already be last in the fitted matrix, or the same fitted tree structure
must be rebuildable into an equivalent natural-order matrix where they are
last. Internally this is the suffix rebuild path. If the `given` set cannot be
placed last this way, `predict` raises `ValueError`.

The current R-vine API does not include arbitrary conditional posterior
samplers. `conditional_method`, R-vine `quad_order`, `conditional_plan`, and
`flexible_graph_plan` are not supported.

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
