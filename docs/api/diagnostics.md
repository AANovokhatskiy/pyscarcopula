# Diagnostics API

Diagnostics help explain which numerical method was used for a fit or
goodness-of-fit calculation. They are especially useful for dynamic models,
where optimizer convergence and numerical approximation accuracy are separate
questions.

## Goodness-of-fit tests

`gof_test` evaluates the Rosenblatt transform and a Cramer-von Mises statistic
for fitted bivariate and multivariate models. With a supplied `fit_result`, the
calculation follows the fitted strategy:

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
- SCAR-TM-JACOBI fields such as `transition_method`, `transition_storage`,
  `stationarity_correction`, `gradient_kind`, `setup_derivative`,
  `filter_derivative`, and spectral negative-mass indicators. Sparse
  numerical and validation diagnostics additionally expose `nnz`, `max_width`,
  `retained_bytes`, `dense_bytes`, and `stationary_error`, together with
  MH/IPFP-specific correction fields when those experimental corrections are
  evaluated. Adaptive-order fit diagnostics store the complete initial and
  fitted-parameter reports as `adaptive_quad_order_initial` and
  `adaptive_quad_order_final`. Options that determine later Jacobi likelihood,
  sampling, and prediction semantics are stored as typed `LatentResult` fields
  rather than only in diagnostics, so they survive stateless dispatch and JSON
  persistence;
- Stochastic Student correlation preprocessing fields such as
  `corr_initialization_source`, `corr_projection_applied`,
  `corr_min_eigenvalue_before`, `corr_min_eigenvalue_after`, and
  `corr_nonfinite_kendall_pairs`.
- `optimizer_parameterization`, which is
  `log_kappa_mu_log_stationary_sigma` for Stochastic Student SCAR-TM-OU fits;
  the fitted parameters themselves remain in public `(kappa, mu, nu)` units.
- `n_threads`, the resolved native thread count used by MLE, GAS, or SCAR.
  Omission always resolves to `1`; environment variables do not override it.

Independent fit batches and rolling risk results additionally expose:

- `n_jobs_requested` and resolved `n_jobs`;
- `n_threads_requested` and resolved `n_threads`;
- `multiprocessing_start_method`;
- `nested_parallelism`;
- `worker_model_ownership='per_task'` and
  `prepared_evaluator_sharing=False`.

For `VineCopula`, `vine.fit_diagnostics["edge_fits"]` additionally separates
the requested vine method from the methods retained on individual edges. It
contains actual method and family counts, dynamic attempt/success counts,
fallback edges, selection and attempted-dynamic `nfev`, discarded fallback
work, failure messages, and edge-level timings. Unsuccessful dynamic edge fits
are replaced by their successful MLE selection results, so performance and
model audits should check `actual_methods` and `fallback_count` rather than the
vine-level method label alone.

For the formulas behind the dynamic Rosenblatt transform and the distinction
between optimizer and approximation convergence, see
[Mathematical Contracts](../guide/mathematical-contracts.md).

::: pyscarcopula.stattests.gof_test

::: pyscarcopula.stattests.vine_gof_test

::: pyscarcopula.stattests.rvine_gof_test

::: pyscarcopula.stattests.vine_rosenblatt_transform

::: pyscarcopula.stattests.rvine_rosenblatt_transform
