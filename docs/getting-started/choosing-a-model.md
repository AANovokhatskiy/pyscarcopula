# Choosing a Model

Choose the dependence structure first, then decide whether it needs to vary
over time. Fit a constant MLE baseline before adding GAS or latent SCAR
dynamics so that the dynamic model can be compared against a simpler nested
question.

## Model Selection

| Goal | Suggested model | Start with |
|---|---|---|
| Dependence between two variables | `GumbelCopula`, `ClaytonCopula`, `FrankCopula`, `JoeCopula`, or `BivariateGaussianCopula` | MLE |
| Time-varying dependence between two variables | The same bivariate families | GAS or SCAR-TM |
| Static Gaussian/Student dependence with supplied, plug-in, shrinkage, full, or factor correlation | `GaussianCopula` or `StudentCopula` | MLE plus an explicit `corr_mode` when the default `fixed` policy is not desired |
| One common correlation that changes over time | `EquicorrGaussianCopula` | GAS or SCAR-TM-OU |
| Time-varying multivariate tail thickness | `StochasticStudentCopula` | GAS or SCAR-TM-OU |
| Different pairwise families and dependence strengths | `VineCopula` | Auto R-vine or fixed C/D structure; MLE, then dynamic edges if needed |
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

1. Select a model family using the table above.
2. Run the [Quick Start](quickstart.md).
3. Fit an MLE baseline and inspect goodness of fit.
4. Add GAS or SCAR only when time variation is part of the modeling question.
5. Read [Performance Tuning](../guide/performance.md) after the statistical
   model and estimation method are settled.
