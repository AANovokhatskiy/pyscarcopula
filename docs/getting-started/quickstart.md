# Quick Start

## Prepare data

pyscarcopula works with pseudo-observations: uniform marginals obtained from
ranked data.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(2026)
d = 2
rho = 0.45
R = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
u = norm.cdf(rng.multivariate_normal(np.zeros(d), R, size=400))
```

For application data, replace this simulated array with column-wise ranks
divided by `n + 1`, or pass raw continuous observations with `to_pobs=True`.

## Fit a bivariate copula

```python
from pyscarcopula import GumbelCopula

copula = GumbelCopula(rotate=180)

# Constant parameter (MLE)
result_mle = copula.fit(u, method="mle")

# Time-varying parameter (SCAR)
result_tm = copula.fit(u, method="scar-tm-ou")

print(f"MLE:  logL = {result_mle.log_likelihood:.2f}")
print(f"SCAR: logL = {result_tm.log_likelihood:.2f}")
```

## Goodness-of-fit test

```python
from pyscarcopula.stattests import gof_test

gof = gof_test(copula, u, fit_result=result_tm, to_pobs=False)
print(f"p-value = {gof.pvalue:.4f}")
```

## Predictive mean copula parameter

```python
from pyscarcopula.api import predictive_mean

r_t = predictive_mean(copula, u, result_tm)
# r_t[k] = E[Psi(x_k) | u_{1:k-1}]
```

## Sample and predict

```python
# sample: reproduce the fitted model (for validation)
v = copula.sample(2000, rng=np.random.default_rng(2024))
copula_refit = GumbelCopula(rotate=180)
result_refit = copula_refit.fit(v, method="scar-tm-ou")
gof_v = gof_test(copula_refit, v, fit_result=result_refit, to_pobs=False)
print(f"GoF on sample: p={gof_v.pvalue:.4f}")

# predict: next-step forecast (for risk metrics)
u_pred = copula.predict(100_000, rng=np.random.default_rng(2025))

# conditional forecast in pseudo-observation space
u_cond = copula.predict(
    20_000,
    given={0: 0.35},
    horizon="current",
    rng=np.random.default_rng(2026),
)
```

Use a fresh `np.random.default_rng(seed)` when you need exactly reproducible
Monte Carlo output.

For multivariate dynamic models, continue with
[Multivariate Models](../guide/multivariate_models.md) or
[Factor Models](../guide/factor-models.md). For vine construction and
conditional sampling, see [Vine Copulas](../guide/vine.md),
[Prediction Semantics](../guide/prediction-semantics.md) and
[R-vine Conditioning](../guide/rvine-conditioning.md).

For guidance on which family and estimation method to use, continue with
[Choosing a Model](choosing-a-model.md) and
[Estimation Methods](../guide/estimation-methods.md).
