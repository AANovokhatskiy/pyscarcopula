# Quick Start

## Prepare data

pyscarcopula works with pseudo-observations: uniform marginals obtained from
ranked data.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(2026)
d = 6
rho = 0.45
R = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
u_6d = norm.cdf(rng.multivariate_normal(np.zeros(d), R, size=400))
u = u_6d[:, :2]
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

## Fit a stochastic Student-t copula

```python
from pyscarcopula import StochasticStudentCopula

student = StochasticStudentCopula(d=u_6d.shape[1], corr_mode="shrinkage")
student_result = student.fit(u_6d, method="scar-tm-ou")

# The dynamic parameter is the Student-t degrees of freedom.
df_t = student.predictive_mean()
u_student_pred = student.predict(10_000, rng=np.random.default_rng(2027))
```

## Fit a multivariate vine

```python
from pyscarcopula import VineCopula

vine = VineCopula()
vine.fit(
    u_6d,
    method="scar-tm-ou",
    truncation_level=2,
    min_edge_logL=10,
    given_vars=[2],
)
vine.summary()

# Vine sampling and prediction
v6 = vine.sample(2_000, rng=np.random.default_rng(2028))
u_pred_6d = vine.predict(100_000, rng=np.random.default_rng(2029))

# Conditional vine forecast: fix one variable
u_pred_6d_cond = vine.predict(
    20_000,
    given={2: 0.6},
    horizon="next",
    rng=np.random.default_rng(2030),
)
```

This lets the R-vine fit choose a structure suited to the variables you expect
to condition on later.

Use a fresh `np.random.default_rng(seed)` when you need exactly reproducible
Monte Carlo output.

For more conditional prediction controls, see
[Prediction Semantics](../guide/prediction-semantics.md) and
[R-vine Conditioning](../guide/rvine-conditioning.md).

## Available copula families

| Family | Class | Rotations | SCAR-OU | SCAR-Jacobi |
|--------|-------|-----------|---------|--------------|
| Gumbel | `GumbelCopula` | 0, 90, 180, 270 | Yes | Yes |
| Clayton | `ClaytonCopula` | 0, 90, 180, 270 | Yes | Yes |
| Frank | `FrankCopula` | 0 | Yes | Yes |
| Joe | `JoeCopula` | 0, 90, 180, 270 | Yes | Yes |
| Independence | `IndependentCopula` | - | - | - |
| Gaussian | `BivariateGaussianCopula` | - | Yes | Yes |
| Equicorrelation | `EquicorrGaussianCopula` | - | Yes | No |
| Stochastic Student-t | `StochasticStudentCopula` | - | Yes | No |
| Gaussian (d-dim) | `GaussianCopula` | - | MLE only | No |
| Student-t (d-dim) | `StudentCopula` | - | MLE only | No |

`scar-tm-jacobi` additionally requires a Kendall-tau parameter mapping; this is
implemented for Gumbel, Clayton, Frank, Joe, and bivariate Gaussian copulas.

Multivariate models can be imported from `pyscarcopula` or from
`pyscarcopula.copula.multivariate`.

## Available estimation methods

| Method | Key | Description |
|--------|-----|-------------|
| MLE | `'mle'` | Constant copula parameter |
| SCAR-TM-OU | `'scar-tm-ou'` | Transfer matrix with OU latent process |
| SCAR-TM-JACOBI | `'scar-tm-jacobi'` | Transfer matrix with Jacobi Kendall-tau dynamics |
| GAS | `'gas'` | Observation-driven score model |

For guidance on which family and estimation method to use, continue with
[Choosing a Model](choosing-a-model.md).
