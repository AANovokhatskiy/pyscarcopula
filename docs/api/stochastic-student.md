# Stochastic Student API

## StochasticStudentCopula

A multivariate Student copula with dynamic degrees of freedom and either fixed
or jointly estimated static correlation. The dynamic scalar parameter is

$$\operatorname{df}_t = 2 + 10^{-6} + \mathrm{softplus}(g_t),$$

and the row density is the standard Student copula density

$$c(u_t;R,\nu_t)=
\frac{t_d(T_{\nu_t}^{-1}(u_t);0,R,\nu_t)}
     {\prod_j t_1(T_{\nu_t}^{-1}(u_{tj});\nu_t)}.$$

`method='mle'` estimates a constant $\nu$, `method='gas'` estimates a
score-driven recursion for $g_t$, and `method='scar-tm-ou'` treats $g_t$ as a
latent OU process integrated by transfer matrix.

SCAR-TM-OU accepts and returns the physical OU parameters
`(kappa, mu, nu)`. For this model only, optimization is performed internally
in `(log(kappa), mu, log(sigma_x))`, where
`sigma_x = nu / sqrt(2 * kappa)`. The representation used by the optimizer is
reported in `result.diagnostics['optimizer_parameterization']`; it does not
change `alpha0` or the fitted parameter values exposed by the API.

### Stochastic Student copula with estimated static correlation

Static correlation modes are selected with `corr_mode`:

```python
StochasticStudentCopula(d=5, R=R, corr_mode="fixed")
StochasticStudentCopula(d=5, corr_mode="shrinkage")
StochasticStudentCopula(d=5, corr_mode="cholesky")
StochasticStudentCopula(
    d=100_000, corr_mode="factor", factor_rank=8)
```

`shrinkage` estimates one additional static parameter. `cholesky` estimates
`d(d-1)/2` static parameters and is intended for low-dimensional problems.
For estimated modes, the initialization/base matrix is selected in this order:
an explicit `corr_base`, then `R`, then a Kendall estimate from the fit data.
Both estimated-correlation modes are available for MLE and SCAR-TM-OU.
GAS supports fixed correlation and the one-parameter `shrinkage` mode; GAS
with `corr_mode="cholesky"` is rejected before fitting. Setting
`analytical_grad=False` retains a fully numerical optimizer gradient. See the
multivariate guide for fitting details and diagnostic fields.

`factor` stores `O(d*k + k^2)` state and supports explicit
`initialize_factor`, static row likelihood, tiled latent-grid evaluation,
MLE, GAS, SCAR-TM-OU, bounded batch sampling, and exact conditional sampling.
It forbids `R` and `corr_base`, and its `R` property never silently allocates
a dense matrix. Factor sampling and conditioning retain the compact
representation.

Dynamic emission densities normally use an interpolated, precomputed Student
quantile (PPF) table of shape `(n_df_nodes, T, d)`, covering the model boundary
through `df = 1000`. Its size is capped at `DEFAULT_MAX_TABLE_BYTES`
(256 MiB). If the values table is skipped, native evaluation keeps the node
range metadata, uses exact quantiles through the final node, and switches to
a controlled third-order normal-quantile asymptotic above it. Quantile
derivatives with respect to `df` are analytical both in the exact and
large-`df` paths, so an out-of-cache gradient does not repeat the expensive
quantile inversion through finite differences. See the performance guide for
accuracy and memory details.

Pass `NumericalConfig(n_threads=N)` to `fit` to parallelize eligible native
emission, row-likelihood, and Monte Carlo work. Methods with a direct
`n_threads` parameter, including conditional sampling and row/grid evaluation,
can opt in per call. Omitting it always selects one native thread.

::: pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula
    options:
      members:
        - fit
        - sample_at_parameter
        - sample_at_parameter_batches
        - sample
        - sample_batches
        - sample_conditional
        - predict
        - predict_batches
        - predictive_mean
        - xT_distribution
        - log_likelihood
        - log_pdf_rows
        - log_pdf_and_dlog_dr_rows
        - pdf_on_grid
        - pdf_and_grad_on_grid
        - transform
        - inv_transform
        - dtransform
