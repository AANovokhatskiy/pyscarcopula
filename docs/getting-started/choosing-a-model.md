# Choosing a Model

Choose the dependence structure first, then decide whether it needs to vary
over time. A constant MLE fit is usually the best starting point: it provides
a baseline before adding GAS or latent SCAR dynamics.

## Model Selection

| Goal | Suggested model | Start with |
|---|---|---|
| Dependence between two variables | `GumbelCopula`, `ClaytonCopula`, `FrankCopula`, `JoeCopula`, or `BivariateGaussianCopula` | MLE |
| Time-varying dependence between two variables | The same bivariate families | GAS or SCAR-TM |
| One unrestricted static correlation matrix | `GaussianCopula` or `StudentCopula` | MLE |
| One common correlation that changes over time | `EquicorrGaussianCopula` | GAS or SCAR-TM-OU |
| Time-varying multivariate tail thickness | `StochasticStudentCopula` | GAS or SCAR-TM-OU |
| Different pairwise families and dependence strengths | `CVineCopula` or `RVineCopula` | MLE, then dynamic edges if needed |
| Very large dimension with low-rank correlation | Gaussian or stochastic Student model with `corr_mode="factor"` | Two-stage factor MLE |

Use the [Bivariate Copulas](../guide/bivariate.md) guide for one pair. For
three or more variables, compare the assumptions in
[Multivariate Models](../guide/multivariate_models.md),
[Factor Models](../guide/factor-models.md), and
[Vine Copulas](../guide/vine.md).

## Estimation Choice

- **MLE** uses one constant copula parameter and is the simplest baseline.
- **GAS** makes dependence observation-driven and avoids latent-state
  integration.
- **SCAR-TM-OU** models an unobserved mean-reverting state.
- **SCAR-TM-JACOBI** models positive Kendall dependence directly on `(0, 1)`.

See [Estimation Methods](../guide/estimation-methods.md) for supported
model-method combinations and fitting examples.

## Prediction Choice

Use `sample` to reproduce a fitted model and `predict` to condition on the
fitted history. Pass `given={column: value}` for conditional generation in
pseudo-observation space.

The distinction between sampling, forecasting, and conditioning is covered in
[Prediction Semantics](../guide/prediction-semantics.md). For arbitrary
R-vine conditioning sets, continue with
[R-vine Conditioning](../guide/rvine-conditioning.md).

## Recommended Reading Path

1. Run the [Quick Start](quickstart.md).
2. Select a model family using the table above.
3. Fit an MLE baseline and inspect goodness of fit.
4. Add GAS or SCAR only when time variation is part of the modeling question.
5. Read [Performance Tuning](../guide/performance.md) after the statistical
   model and estimation method are settled.
