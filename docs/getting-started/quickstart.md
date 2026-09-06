# Quick Start

This complete example fits a constant bivariate model, checks the result,
and generates unconditional and conditional samples. See
[Installation](installation.md) before running it.

## Prepare data

Rows are observations and columns are variables. Copulas use
pseudo-observations strictly inside `(0, 1)`; this simulation already has
uniform margins.

```python
import numpy as np
from scipy.stats import norm
from pyscarcopula import GumbelCopula

rng = np.random.default_rng(2026)
R = np.array([[1.0, 0.45], [0.45, 1.0]])
u = norm.cdf(rng.multivariate_normal(np.zeros(2), R, size=400))
```

For continuous application data, pass `to_pobs=True` to `fit` to rank each
column and divide the ranks by `n + 1`. Otherwise pass existing
pseudo-observations with the default `to_pobs=False`.

## Fit a bivariate copula

```python
copula = GumbelCopula(rotate=180)
result = copula.fit(u, method="mle")
print(result.success, result.message)
print(result.copula_param, result.log_likelihood)
if not result.success:
    raise RuntimeError(result.message)
```

The result is also available as `copula.fit_result`. Inspect `success`
before using a fit: a finite likelihood alone does not establish optimizer
convergence. See [Configuration and Results](../api/configuration.md#fit-results)
for fields returned by each method.

## Sample and predict

```python
v = copula.sample(500, rng=np.random.default_rng(2024))
u_pred = copula.predict(500, rng=np.random.default_rng(2025))
u_cond = copula.predict(
    500, given={0: 0.35}, rng=np.random.default_rng(2026),
)
assert v.shape == u_pred.shape == u_cond.shape == (500, 2)
np.testing.assert_array_equal(u_cond[:, 0], np.full(500, 0.35))
```

`sample` reproduces the fitted model; `predict` draws forecast observations.
Their distributions coincide for MLE. `given` fixes zero-based columns in
pseudo-observation space. Use a fresh seeded generator to repeat a draw
sequence; reusing a generator advances its stream.

## Goodness-of-fit test

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=result, to_pobs=False)
print(f"p-value = {gof.pvalue:.4f}")
```

GoF evaluates a Rosenblatt transform and a Cramer-von Mises statistic.
See [Diagnostics](../api/diagnostics.md) for parametric-bootstrap calibration
and its fit-success diagnostics.

## Predictive mean copula parameter

The MLE parameter is constant. For observation-driven or latent parameter
paths, continue with [Bivariate Copulas](../guide/bivariate.md) and its
[predictive mean example](../guide/bivariate.md#predictive-mean-parameter).

## Next steps

Use [Choosing a Model](choosing-a-model.md) for alternative structures,
[Estimation Methods](../guide/estimation-methods.md) for GAS and SCAR, and
[Prediction Semantics](../guide/prediction-semantics.md) for dynamic horizons.
