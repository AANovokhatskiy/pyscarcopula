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

When no native thread count is supplied anywhere, the result is always one
thread, independently of environment variables. Each process owns a separate
model and prepared evaluator. For `n_jobs=1`, rolling windows remain
sequential and may reuse the caller's model; each fit invalidates transient
prepared state before the next window.

See [CPU Parallelism](../guide/parallelism.md) for oversubscription,
reproducibility, and thread-safety guidance.

## fit_independent

::: pyscarcopula.contrib.parallel_fit.fit_independent

`IndependentFitBatch.models` contains one independently reconstructed fitted
model per task. No model, prepared evaluator, or transient fit cache is shared
between tasks.

Nonreal observations and unknown or wrong-method `fit_kwargs` are rejected
before any fit or worker process starts. This includes task-specific keyword
dictionaries: an invalid later task prevents earlier tasks from running.
Optimizer nonconvergence remains a returned result; inspect each
`batch.results[i].success` before using its model. An unsuccessful fit does
not publish that result as the model's fitted state.

## risk_metrics

::: pyscarcopula.contrib.risk_metrics.risk_metrics

## MarginalModel

::: pyscarcopula.contrib.marginal.MarginalModel
    options:
      members:
        - create
        - fit_rolling
        - ppf
