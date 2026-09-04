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
`batch.results[i].success` before using its model. Whether an unsuccessful
result is attached to `model.fit_result` follows that model's fit contract;
the presence of fitted state alone does not establish convergence.

## risk_metrics

::: pyscarcopula.contrib.risk_metrics.risk_metrics

`failure_policy="raise"` is the default. A final unsuccessful copula fit
stops that window before prediction can use older fitted state. A final
unsuccessful SLSQP result stops before its VaR, CVaR or weights are consumed.
Iteration/evaluation limit exhaustion counts as failure even when the
returned candidate is finite. Internal rejected objective trials, unsuccessful
restarts and successful model fallbacks do not count as final failure; for a
vine, the final selected edge results determine overall fit success.

The exception identifies the stage and zero-based window end index. With
multiple processes, the first error received stops collection and terminates
remaining workers; other windows may already have run. This does not promise
the chronologically earliest failing window or rollback of the caller's model.

`failure_policy="continue"` retains the previous handling of unsuccessful
results: prediction uses the available fitted state and the portfolio step
uses the returned optimizer candidate. In particular, a rejected refit can
leave an older fitted state in use. A fresh model with no fitted state may
still raise. Exceptions are never suppressed under either policy.

This policy checks final copula and portfolio results. Marginal fits return
parameter arrays rather than a common success result, so hidden marginal
optimizer convergence flags are not inspected; marginal exceptions propagate
unchanged. The policy does not make unsupported marginal prediction modes
available.

Nonreal input and unknown or wrong-owner fit keywords are rejected before
marginal fitting or worker submission under both policies. Gaussian/Student
models retain their documented MLE method override and receive the supplied
optimizer/numerical settings, including resolved `n_threads`.

## MarginalModel

::: pyscarcopula.contrib.marginal.MarginalModel
    options:
      members:
        - create
        - fit_rolling
        - ppf
