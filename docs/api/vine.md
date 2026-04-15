# Vine API

Both `CVineCopula.predict(...)` and `RVineCopula.predict(...)` support:

- `given={var_index: u_value}` for conditional generation
- `horizon='current'|'next'` for SCAR-TM-OU edge mixtures

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
