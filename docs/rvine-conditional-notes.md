# R-vine Conditional Sampling

This note documents the current R-vine conditional sampling contract.

## Current Status

`RVineCopula.predict(..., given=...)` supports conditional sampling only when
the fixed variables can be placed at the end of the R-vine variable order. In
the implementation this is the suffix rebuild path.

A `given` set is accepted when one of these conditions holds:

- The fixed variables are already the last entries of the fitted R-vine
  variable order, read from the anti-diagonal of the natural-order matrix.
- The fitted tree structure can be rebuilt into an equivalent natural-order
  matrix whose variable order has those fixed variables last.
- All variables are fixed, in which case `predict` returns constant rows.

If the requested conditioning pattern cannot be represented as the end of a
natural-order matrix's variable order, `RVineCopula.predict` raises
`ValueError`. Arbitrary DAG, graph, grid, and exact posterior conditional
samplers are not part of the current R-vine implementation.

## Public API

Use pseudo-observation values in `(0, 1)`:

```python
from pyscarcopula import RVineCopula

vine = RVineCopula().fit(u, method="mle")

variable_order = [
    int(vine.matrix[vine.d - 1 - col, col])
    for col in range(vine.d)
]

# Fix the last variable in the R-vine order.
samples = vine.predict(
    n=5000,
    u=u,
    given={variable_order[-1]: 0.6},
    horizon="next",
)

# Fix multiple variables at the end of the R-vine order.
samples2 = vine.predict(
    n=5000,
    u=u,
    given={
        variable_order[-2]: 0.35,
        variable_order[-1]: 0.75,
    },
)
```

For some fitted structures, a set that is not last in the current matrix may
still work because the same vine can be rebuilt with that set last in the
variable order:

```python
given = {variable_order[0]: 0.45}

try:
    samples = vine.predict(5000, u=u, given=given)
except ValueError:
    # This pattern cannot be placed last in an equivalent natural-order matrix.
    samples = None
```

## Implementation

The relevant implementation files are:

- `pyscarcopula/vine/rvine.py`
  - `RVineCopula._given_suffix_start_col`
  - `RVineCopula._suffix_sampling_state`
  - `RVineCopula._find_peel_order_for_given_suffix`
  - `RVineCopula._sample_suffix_given_with_r`
  - `RVineCopula.predict`
- `pyscarcopula/vine/_conditional_rvine.py`
  - `validate_rvine_given`
- `pyscarcopula/vine/_rvine_matrix_builder.py`
  - rebuilds natural-order matrices from the fitted tree representation

The sampler works in two phases:

1. It validates `given` and finds a natural-order matrix whose variable order
   ends with the fixed variables. This is either the fitted matrix itself or a
   rebuilt matrix with remapped pair-copula positions.
2. It propagates fixed trailing-order pseudo-observations forward through
   h-functions, then samples the remaining variables by inverse h-functions
   using the same natural-order recursion as unconditional sampling.

Dynamic edges keep the same predictive parameter handling as unconditional
`predict`: `horizon="current"` uses the current fitted edge state and
`horizon="next"` uses the one-step-ahead predictive state.

## Removed Paths

The previous arbitrary-conditioning experiment has been removed from code and
tests. In particular, the current implementation does not provide:

- `conditional_method`
- `quad_order` for R-vine conditional prediction
- `conditional_plan`
- `flexible_graph_plan`
- DAG/graph conditional sampling
- grid or exact posterior conditional sampling for R-vines

The dedicated DAG module and validation tests were also removed:

- `pyscarcopula/vine/_rvine_dag.py`
- `tests/test_rvine_dag.py`
- `tests/test_rvine_conditional_dag.py`
- `tests/test_vine_validation.py`

## Verification

The current variable-order suffix implementation is covered by:

```powershell
pytest tests\test_rvine_copula.py -q
pytest tests -q
```

Observed local result after the cleanup:

```text
pytest tests\test_rvine_copula.py -q
106 passed

pytest tests -q
528 passed, 2 skipped
```

The only observed warning was a `.pytest_cache` permission warning on Windows;
it is unrelated to conditional sampling.

## Limitations

- Supported conditioning patterns depend on the fitted R-vine structure.
- Arbitrary `given` sets that cannot be placed last in an equivalent
  natural-order matrix are rejected instead of approximated.
- There is no posterior reweighting for fixed variables that cannot be placed
  at the end of the variable order.
- C-vine conditional sampling is separate and still has its own prefix/general
  implementation in `pyscarcopula/vine/_conditional_cvine.py`.
