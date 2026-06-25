# Diagnostics API

Diagnostics expose the numerical contract that was actually used for a fit or
goodness-of-fit calculation. They are especially important for dynamic models,
where optimizer convergence and approximation convergence are separate
questions.

## Goodness-of-fit tests

`gof_test` evaluates the Rosenblatt transform and a Cramer-von Mises statistic
for fitted bivariate and multivariate models. With a supplied `fit_result`, the
state semantics follow the fitted strategy:

- MLE uses the fitted constant parameter.
- GAS uses the filtered point state path.
- SCAR-TM integrates conditional h-functions over the predictive latent-state
  distribution.

Bootstrap calibration, when requested, simulates from the fitted model and
recomputes the statistic on generated samples. For stochastic latent-state
models this means resampling both the latent path and the copula observations,
not only perturbing the observed pseudo-observations.

Common fit diagnostics to inspect before interpreting GoF results include:

- optimizer fields such as `success`, `message`, objective evaluations, and
  gradient kind;
- SCAR-TM-OU transition attempts and fallback counters such as
  `fallback_spectral_to_matrix`, `fallback_matrix_to_local`,
  `matrix_failures`, and `matrix_capped`;
- SCAR-TM-JACOBI fields such as `transition_method`, `gradient_kind`,
  `setup_derivative`, `filter_derivative`, and spectral negative-mass
  indicators;
- Stochastic Student correlation preprocessing fields such as
  `corr_initialization_source`, `corr_projection_applied`,
  `corr_min_eigenvalue_before`, `corr_min_eigenvalue_after`, and
  `corr_nonfinite_kendall_pairs`.

For the formulas behind the dynamic Rosenblatt transform and the distinction
between optimizer and approximation convergence, see
[Mathematical Contracts](../guide/mathematical-contracts.md).

::: pyscarcopula.stattests.gof_test

::: pyscarcopula.stattests.vine_gof_test

::: pyscarcopula.stattests.rvine_gof_test

::: pyscarcopula.stattests.vine_rosenblatt_transform

::: pyscarcopula.stattests.rvine_rosenblatt_transform
