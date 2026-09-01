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

An explicit static Gaussian or Student `fit_result` also supplies correlation
state (dense or factor) and Student degrees of freedom, taking precedence over
state attached to the model without changing that model. GAS diagnostics keep
the fitted `scaling` and `score_eps`. Dynamic multivariate SCAR diagnostics
require an OU result and preserve its grid settings and `auto_small_kdt`;
`K` and `grid_range` on the GoF call override the corresponding grid sizes.

Bootstrap calibration, when requested, simulates from the fitted model and
recomputes the statistic on generated samples. For stochastic latent-state
models this means resampling both the latent path and the copula observations,
not only perturbing the observed pseudo-observations.

`bootstrap_fit_kwargs` accepts the selected model's normal fit options,
including strategy constructor settings such as GAS `scaling` or OU `K`.
These settings control bootstrap refitting. For bivariate,
`EquicorrGaussianCopula`, and `StochasticStudentCopula` GAS refits, the fitted
`score_eps` is retained even when a new `gamma0` is supplied.
For these models, an explicit `score_eps` takes priority over an explicit
`config.gas_score_eps`, which takes priority over the fitted score step.
Passing `score_eps=None` selects the config or fitted default.
GAS Vine refits do not restore fitted score steps from individual edges.
Set `bootstrap_fit_kwargs={'score_eps': value}` to use one chosen step for
all refitted GAS edges; otherwise they use an explicit `config.gas_score_eps`
or the library default.
Unknown or misplaced keys are rejected before bootstrap random streams or
workers are created, including when `bootstrap_refit=False`. In that mode,
valid fit-only options have no effect. Samplers that support native threads
receive the resolved `config.n_threads`, including static Student and
Equicorr models; parallel bootstrap uses one native thread per worker.
The dictionary cannot override `to_pobs`: generated bootstrap samples are
already pseudo-observations.

Fitted `VineCopula` models follow the same parametric-bootstrap contract. A
replication simulates from the captured fitted R-vine, optionally refits a
worker-owned vine with the same structure and requested fitting settings,
applies the R-vine Rosenblatt transform, and recomputes the Cramer-von Mises
statistic.
Static exact built-in edges use the native Rosenblatt runtime; unsupported or
dynamic edges use the preserved Python transform without changing the result
schema or random-stream policy.

The returned `BootstrapGoFResult` exposes `statistic`, the calibrated
`pvalue`, `bootstrap_statistics`, `n_bootstrap`, and
`bootstrap_diagnostics`. Parallel execution metadata is available as
`n_jobs_requested`, resolved `n_jobs`, `n_threads`, and `backend`; reproducible
execution policy is recorded in `rng_policy` and `worker_model_ownership`.
Unsuccessful refits retain `bootstrap_fit_success=False` and their messages
in the diagnostics. Their statistics remain in the calibration sample; check
these flags before interpreting the calibrated p-value.

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

::: pyscarcopula.stattests.rvine_gof_test

::: pyscarcopula.stattests.rvine_rosenblatt_transform
