# Contrib API

Optional modules for risk metrics, independent fit batches, and marginal
models. Not part of the core copula API.

```python
from pyscarcopula.contrib.risk_metrics import risk_metrics
from pyscarcopula.contrib.parallel_fit import fit_independent
from pyscarcopula.contrib.marginal import MarginalModel
```

`risk_metrics(..., n_jobs=...)` is used for both rolling marginal fits and
rolling copula/risk windows. `n_jobs=-1` uses all available workers where the
selected marginal model supports parallel fitting.

When more than one worker process is used, both `risk_metrics` and
`fit_independent` default to one native thread per worker. An explicit
`n_threads` value greater than one opts into nested parallelism. The resolved
worker count, native thread count, multiprocessing start method, and ownership
policy are available in result diagnostics.

## fit_independent

::: pyscarcopula.contrib.parallel_fit.fit_independent

`IndependentFitBatch.models` contains one independently reconstructed fitted
model per task. No model, prepared evaluator, or transient fit cache is shared
between tasks.

## risk_metrics

::: pyscarcopula.contrib.risk_metrics.risk_metrics

## MarginalModel

::: pyscarcopula.contrib.marginal.MarginalModel
    options:
      members:
        - create
        - fit_rolling
        - ppf
