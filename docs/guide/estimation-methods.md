# Estimation Methods

This page describes fitting methods exposed by `pyscarcopula.api.fit`,
including the static multivariate model contract. Performance controls are covered in
[Performance Tuning](performance.md).

For the compact formulas behind each dynamic model, including the transfer
filters, predictive Rosenblatt transform, and numerical convergence criteria,
see [Mathematical Contracts](mathematical-contracts.md).

Each estimation method supports a defined set of model families. Unsupported
combinations fail before optimization starts.

Built-in strategies separate constructor settings from arguments to each
operation. For example, `K` configures SCAR-TM-OU numerical evaluation, while
`alpha0`, `initial_mle_result`, and `maxiter` belong to fitting; passing those
fit arguments to `mlog_likelihood` or a post-fit likelihood call raises
`TypeError`. Unknown keywords also raise `TypeError` instead of being ignored.
Sampling and prediction arguments such as `rng` and `given` are routed to the
sampler, separately from numerical constructor overrides.

For automatic SCAR-TM-OU initialization, `config.mle_optimizer` controls the
internal static MLE even when `smart_init=True`. The resulting static estimate
is reused by the initialization heuristics and their fallbacks. Supplying
`initial_mle_result` avoids that static fit; an explicit `alpha0` bypasses
automatic initialization entirely. If the internal MLE raises with
`smart_init=True`, initialization uses the constant fallback and records the
error in its diagnostics. With `smart_init=False`, the MLE error propagates.

## Method Summary

| Method | Key | State | Main use |
|--------|-----|-------|----------|
| MLE | `'mle'` | Static/constant model parameters | Baseline fit, family selection, and static multivariate fitting |
| GAS | `'gas'` | Observation-driven score recursion | Fast dynamic dependence without latent integration |
| SCAR-TM-OU | `'scar-tm-ou'` | OU latent state mapped to bivariate dependence or Student degrees of freedom | Deterministic stochastic-latent likelihood |
| SCAR-TM-JACOBI | `'scar-tm-jacobi'` | Jacobi diffusion for Kendall's tau | Bounded tau dynamics with deterministic filtering |

All dynamic methods return a `LatentResult` with `params`,
`log_likelihood`, optimizer status, and enough metadata for `predict`,
`predictive_mean`, and GoF utilities. Model sampling is available where the
strategy implements a path simulator. SCAR-TM-JACOBI supports both
unconditional `sample` and conditional or unconditional `predict`.
Its default unconditional sampler is the likelihood-consistent
`sampling_method='tm_grid'`; experimental `sampling_method='lamperti_euler'`
is available as an opt-in continuous-path approximation.

## Gradient capability matrix

The model score used inside a recursion and the optimizer gradient are
different quantities. The following table describes what is passed to the
outer optimizer and the corresponding diagnostics.

| Method | Configuration | Optimizer gradient | `model_score` | `gradient_kind` |
|--------|---------------|--------------------|---------------|-----------------|
| MLE | Built-in supported model | Analytical | `not_applicable` | `analytical` |
| GAS | Any supported scaling | Native finite differences | `native` | `native_finite_difference` |
| SCAR-TM-OU | `analytical_grad=True` | Analytical native Jacobian | `not_applicable` | `analytical` |
| SCAR-TM-OU | `analytical_grad=False` | Numerical finite differences | `not_applicable` | `numerical` |
| SCAR-TM-JACOBI | `analytical_grad=False` | Numerical finite differences | `not_applicable` | `numerical` |
| SCAR-TM-JACOBI | `local_fixed`, analytical gradient | Model-provided | `not_applicable` | `analytical` |
| SCAR-TM-JACOBI | `local`, `spectral_matrix`, or `auto`, analytical gradient | Model-provided | `not_applicable` | `semi_analytical` |
| SCAR-TM-JACOBI | `spectral_coeff`, `analytical_grad=True` | Native finite differences | `not_applicable` | `native_finite_difference` |

For joint Stochastic Student SCAR-TM-OU fits, the analytical-gradient path
includes OU and static-correlation derivatives. Result diagnostics report the
correlation and joint-gradient routes in `correlation_gradient` and
`joint_gradient`. Some fits also include
`prepared_native_evaluator`, `prepared_native_evaluator_count`, and
`prepared_native_fallback`.

For SCAR-TM-OU and SCAR-TM-JACOBI,
`result.diagnostics["initialization"]` records how the optimizer initial point
was obtained. It contains `requested_method`, `selected_method`, the final
`alpha0`, and an ordered `attempts` list. Failed attempts retain only a
serializable `error_type` and `error_message`; traceback and exception objects
are not stored. A user-provided `alpha0` is reported as `user_provided`.

## MLE

MLE is the label for a static model. For bivariate scalar families it
estimates one constant copula parameter and remains the default baseline for
family screening and dynamic initialization. For multivariate Gaussian and
Student models, the correlation procedure is an independent policy choice:
an MLE-labelled result can use supplied, plug-in, or jointly optimized
correlation.

The optimizer works directly in the natural copula parameter. An explicit
`alpha0` therefore uses natural units:

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

copula = GumbelCopula(rotate=180)
result = fit(copula, u, method='mle', alpha0=[2.0])
```

For static and stochastic Student copulas fitted with `method='mle'`, the
optimized scalar is `df` itself and is constrained above the model's
finite-variance threshold. No latent softplus transform is applied inside the
MLE objective. The softplus transform remains part of dynamic SCAR/GAS models,
where a latent state drives time-varying degrees of freedom.

For static `GaussianCopula` and `StudentCopula`:

| `corr_mode` | Optimizer treatment | Intended use |
|---|---|---|
| `fixed` | No correlation optimizer coordinates; supplied `R` is fixed, otherwise correlation is a counted plug-in estimate | Default compatibility and fast baseline |
| `shrinkage` | Joint analytical correlation score for one raw weight | Parsimonious joint static fit |
| `cholesky` | Joint analytical score for `d*(d-1)/2` raw coordinates | Small dimensions |
| `factor` | Two-stage compact loadings; static Student also supports joint identified loadings | Large dimensions under a factor assumption |

The result fields `corr_n_params`, `corr_plugin_n_params`, and
`corr_effective_n_params` separate optimizer coordinates from data-derived
plug-in parameters. `corr_estimator` records the actual procedure. Thus
`fixed` alone is not enough to tell whether correlation was supplied or
estimated.

For static and stochastic Student models with `corr_mode='fixed'`, the analytical `df`
derivative is passed to L-BFGS-B. Joint `df` and estimated-correlation MLE
combines that derivative with the Student correlation score, then maps it into
the
configured shrinkage or Cholesky raw parameters. It reports
`gradient_mode='analytical_joint'` in result diagnostics.

## GAS

GAS is observation-driven. The copula parameter is

$$
r_t = \Psi(g_t),
$$

where the unbounded recursion state follows

$$
g_{t+1} = \omega + \beta g_t + \gamma s_t.
$$

The score $s_t$ is the scaled derivative of the current copula log-density
with respect to the recursion state. Conditional on past data, GAS has a point
state rather than a latent-state distribution, so its likelihood avoids the
latent-state integration required by SCAR.

```python
result = fit(
    copula,
    u,
    method='gas',
    scaling='unit',
    ftol=1e-12,
    maxfun=3000,
)
```

GAS uses the compiled evaluator for likelihood, score recursion, state
updates, prediction, and the bivariate Rosenblatt path for supported built-in
copulas. Unsupported copulas and missing compiled support fail immediately.

Use `scaling='unit'` as the numerical baseline. `scaling='fisher'` uses the
analytical copula score as its numerator and a finite-difference curvature
estimate with clipping/floor thresholds; its fitted optimum can be sensitive
to optimizer finite-difference steps.

The GAS copula score and filtering recursion are model calculations.
They are not the optimizer Jacobian with respect to
`(omega, gamma, beta)`. The compiled evaluator computes that outer gradient
with two-point finite differences matching SciPy's forward-step and bound
adjustment conventions, and returns the objective and gradient through
one L-BFGS-B callback. Result diagnostics distinguish these concepts with
`model_score='native'`, `optimizer_gradient='native'`, and
`gradient_kind='native_finite_difference'`. `maxfun` and `nfev` use scalar
objective budget units, including numerical-gradient probes: four units
per native provider call, or five for joint GAS/shrinkage fitting. These equal
the scalar evaluation counts when the finite-difference call completes.
A numerical failure can abort the native call earlier while still charging
the full budget units, so `nfev` is not a physical likelihood-call counter.
An explicit `finite_diff_rel_step` takes precedence over the absolute `eps` step.

## SCAR-TM-OU

SCAR-TM-OU uses a one-dimensional Ornstein-Uhlenbeck latent state,

$$
dx_t = \kappa(\mu - x_t)\,dt + \nu\,dW_t.
$$

What the state controls depends on the model. For a bivariate copula it is
mapped to that family's scalar dependence parameter. For
`StochasticStudentCopula` it is mapped to the time-varying degrees of freedom;
the multivariate correlation structure remains a separate static component.

The likelihood integrates the latent Markov path deterministically. The
bivariate and Stochastic Student models use the same OU transition backends,
but they do not use the same optimizer coordinates by default.

SCAR-TM-OU requires the package C++ extension; no Python fallback implements
this public fitting path.

```python
result = fit(
    copula,
    u,
    method='scar-tm-ou',
    transition_method='auto',
    analytical_grad=True,
)
```

The public OU parameters are always `kappa`, `mu`, and `nu`. A user-supplied
`alpha0` must use these physical coordinates in that order.

### Bivariate copulas

For a bivariate family $C$, the dynamic copula parameter is

$$
\theta_t = \Psi_C(x_t).
$$

Thus SCAR-TM-OU describes stochastic dependence: examples include a dynamic
Gumbel parameter and a dynamic Gaussian correlation. The transform
$\Psi_C$ and its admissible range are family-specific.

By default, the bivariate optimizer uses scaled physical
`[kappa, mu, nu]` coordinates. Set
`log_stationary_scale_optimization=True` to optimize instead in

$$
(\log\kappa,\ \mu,\ \log\sigma_x),
\qquad \sigma_x=\frac{\nu}{\sqrt{2\kappa}}.
$$

### Stochastic Student copula

For `StochasticStudentCopula`, the OU state instead controls tail thickness:

$$
\operatorname{df}_t = 2 + 10^{-6} + \operatorname{softplus}(x_t).
$$

The correlation structure remains static. This model uses the log coordinates
above by default. Set `log_stationary_scale_optimization=False` to use scaled
physical `[kappa, mu, nu]` coordinates instead.

### Shared transition likelihood

The exact OU one-step transition is Gaussian, so all SCAR-TM-OU likelihood
backends evaluate the same latent Markov model. They differ only in how the
one-dimensional transition integral is approximated: Hermite spectral
projection, a finite-grid transition matrix, or local Gauss-Hermite
quadrature.

By default, `transition_method='auto'` uses a Hermite spectral likelihood
except in narrow-kernel regimes, where it uses local Gauss-Hermite. If spectral
evaluation fails numerically, `auto` tries the matrix grid likelihood first
and then the local method when the matrix path is not accepted.

`LatentResult.diagnostics` records objective evaluations,
spectral/matrix/local attempts, and transition fallback counters such as
`fallback_spectral_to_matrix`,
`fallback_matrix_to_local`, `matrix_failures`, and `matrix_capped`.

By default, `spectral_basis_order='auto'` selects the Hermite basis size inside
each objective evaluation from the current `kappa / (T - 1)`: 128 below
`0.015`, 96 below `0.025`, 64 below `0.06`, and 32 otherwise. Use a fixed
positive integer when exact basis-size reproducibility is needed for numerical
comparisons.

## SCAR-TM-JACOBI

SCAR-TM-JACOBI evolves Kendall's tau directly on `(0, 1)`:

$$
d\tau_t =
\kappa(m - \tau_t)\,dt
+ \xi\sqrt{\tau_t(1-\tau_t)}\,dW_t.
$$

The copula parameter is recovered from tau through the copula's
`tau_to_param` mapping. This method is therefore available for copulas that
implement both `tau_to_param` and `param_to_tau`: Gumbel, Clayton, Frank,
Joe, and bivariate Gaussian copulas.

```python
result = fit(
    copula,
    u,
    method='scar-tm-jacobi',
    transition_method='auto',
)

print(result.params.kappa, result.params.m, result.params.xi)
```

The fitted parameters are:

- `kappa`: mean-reversion speed
- `m`: long-run Kendall's tau level
- `xi`: Jacobi volatility

The Jacobi stationary law is beta, and the transition operators are built in
tau space rather than in an unconstrained OU coordinate. The local backend uses
the Lamperti coordinate to apply Gauss-Hermite steps while keeping tau inside
its bounded state space.

`transition_method='auto'` first tries a Jacobi spectral transition matrix on
the tau quadrature grid. If the truncated spectral matrix has material
negative mass, or if row normalization fails numerically, it falls back to the
local Lamperti/Gauss-Hermite transition. The default tolerance for accepting
small spectral truncation errors is `negative_mass_tol=1e-5`.

The explicit Jacobi transition backends are `spectral_matrix`, `local`,
`local_fixed`, and `spectral_coeff`. With `analytical_grad=True`, the optimizer
receives a model-provided Jacobian. For `local_fixed`, both setup and filtering
derivatives are analytical. For `local`, `spectral_matrix`, and either backend
selected by `auto`, setup arrays are differentiated by finite differences and
the filtering recursion is differentiated analytically; these modes are
therefore semi-analytical. `spectral_coeff` is a coefficient-space comparison
backend. With `analytical_grad=True`, it supplies a complete native central
finite-difference objective gradient and reports
`gradient_kind='native_finite_difference'`; it is not an analytical derivative.

Jacobi fitting requires at least two observations because its transition time
step is `dt = 1 / (T - 1)`. A one-row prepared evaluator remains valid for
conditioning an existing state, which does not construct a transition.

`LatentResult.diagnostics` reports `gradient_requested`, `gradient_used`,
`gradient_kind`, `setup_derivative`, `filter_derivative`, and the transition
backend actually selected at the fitted parameters.

The fitted result also retains every Jacobi option that changes subsequent
likelihood or prediction semantics: `transition_method`, `gh_order`,
`spectral_basis_order`, `spectral_quad_order`, `tau_eps`, `theta_cap`,
`clip_negative`, `negative_mass_tol`, `stationary_shape_max`,
`transition_storage`, `stationarity_correction`,
`sampling_method`, `lamperti_substeps`, `lamperti_boundary`, `lamperti_eps`,
`lamperti_engine`, `lamperti_chunk_observations`, and a configured
`memory_budget_bytes`. Stateless API calls and models restored from JSON
reconstruct the strategy from these fields. Explicit kwargs passed to a later
API call still override the stored values.

Unconditional sampling defaults to the transition-matrix grid used by the
likelihood. An experimental `sampling_method='lamperti_euler'` instead
simulates a continuous tau path with substepped Euler--Maruyama in the
Lamperti coordinate. It is useful as an independent discretization comparison
but does not change the fitting backend and is not an exact diffusion sampler.
Its mandatory native C++ kernel is strictly sequential and consumes Gaussian
innovations created by the caller's NumPy generator in bounded chunks. Legacy
`lamperti_engine='numba'` and `'python'` labels normalize to `'native'`; they
do not select separate production implementations.

For an explicit moving-grid local transition,
`transition_storage='sparse'` selects the `O(K * gh_order)` sparse filtering
and prediction backend. The default is `'dense'`; sparse storage is not
currently available for `auto` or spectral transitions. Explicit
`local_fixed` supports sparse analytical-gradient filtering with shared
transition/derivative indices.
`stationarity_correction='mh'` and `stationarity_correction='ipfp'` are
experimental sparse-only options applied consistently to likelihood,
prediction, state filtering, and `tm_grid` sampling for the moving-grid
`local` backend. IPFP preserves the original sparse support and therefore
fails explicitly if that support cannot be balanced to both stationary
marginals. Neither correction is available for `local_fixed`; uncorrected
sparse `local_fixed` supports `analytical_grad=True`.

`adaptive_quad_order=True` is an experimental option for the uncorrected
sparse moving-grid `local` backend. It selects a quadrature order once, from
`adaptive_quad_orders`, using deterministic full-horizon gates before
optimization. The chosen order remains fixed for likelihood, prediction, and
sampling and is persisted in the fitted result. The final-parameter gate is
diagnostic only and never changes the fitted order.

For high-frequency data, the code uses `dt = 1 / (T - 1)`. Large `T` therefore
produces very narrow one-step Jacobi transitions. In this regime the local
transition produces a nonnegative row-normalized matrix. Change
`basis_order` only when comparing the spectral approximation against the local
backend; otherwise leave backend selection to `transition_method='auto'`.

## Sampling, Prediction, and Diagnostics

`predictive_mean(copula, u, result)` returns the predictive mean copula
parameter at each time step before the current observation is absorbed.

For `predict`, SCAR-TM methods support `horizon='current'` for the posterior
state after the last observation and `horizon='next'` for the one-step-ahead
state. The shared prediction terminology is described in
[Prediction Semantics](prediction-semantics.md).
