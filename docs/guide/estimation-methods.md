# Estimation Methods

This page describes the bivariate fitting methods exposed by
`pyscarcopula.api.fit`. Performance controls for these methods are covered in
[Performance Tuning](performance.md).

For the compact formulas behind each dynamic model, including the transfer
filters, predictive Rosenblatt transform, and numerical convergence criteria,
see [Mathematical Contracts](mathematical-contracts.md).

Strategy compatibility is determined by explicit model capabilities. See
[Model Architecture](architecture.md) for the distinction between class
hierarchy, protocols, and native strategy support.

## Method Summary

| Method | Key | State | Main use |
|--------|-----|-------|----------|
| MLE | `'mle'` | Constant copula parameter | Baseline fit and family selection |
| GAS | `'gas'` | Observation-driven score recursion | Fast dynamic dependence without latent integration |
| SCAR-TM-OU | `'scar-tm-ou'` | OU latent state mapped to the copula parameter | Deterministic stochastic-latent likelihood |
| SCAR-TM-JACOBI | `'scar-tm-jacobi'` | Jacobi diffusion for Kendall's tau | Bounded tau dynamics with deterministic filtering |

All dynamic methods return a `LatentResult` with `params`,
`log_likelihood`, optimizer status, and enough metadata for `predict`,
`predictive_mean`, and GoF utilities. Model sampling is available where the
strategy implements a path simulator; SCAR-TM-JACOBI supports prediction but
not `sample`.

The historical Monte Carlo SCAR strategies, `'scar-p-ou'` and `'scar-m-ou'`,
remain available for reproducing earlier experiments. For routine dynamic
fits, prefer deterministic TM or GAS methods.

## Gradient capability matrix

The model score used inside a recursion and the optimizer gradient are
different quantities. The following table describes what is passed to the
outer optimizer and the corresponding diagnostics.

| Method | Configuration | Optimizer gradient | `model_score` | `gradient_kind` |
|--------|---------------|--------------------|---------------|-----------------|
| MLE | Built-in supported model | Analytical | `not_applicable` | `analytical` |
| GAS | Any supported scaling | Numerical finite differences | `native` | `numerical_optimizer` |
| SCAR-TM-OU | `analytical_grad=True` | Analytical native Jacobian | `not_applicable` | `analytical` |
| SCAR-TM-OU | `analytical_grad=False` | Numerical finite differences | `not_applicable` | `numerical` |
| SCAR-TM-JACOBI | `analytical_grad=False` | Numerical finite differences | `not_applicable` | `numerical` |
| SCAR-TM-JACOBI | `local_fixed`, analytical gradient | Model-provided | `not_applicable` | `analytical` |
| SCAR-TM-JACOBI | `local`, `spectral_matrix`, or `auto`, analytical gradient | Model-provided | `not_applicable` | `semi_analytical` |

For joint Stochastic Student SCAR-TM-OU fits, C++ supplies OU and
static-correlation derivatives. Python applies the correlation
parameterization chain rule. Result diagnostics report the correlation and
joint-gradient routes in `correlation_gradient` and `joint_gradient`.

For SCAR-TM-OU and SCAR-TM-JACOBI,
`result.diagnostics["initialization"]` records how the optimizer initial point
was obtained. It contains `requested_method`, `selected_method`, the final
`alpha0`, and an ordered `attempts` list. Failed attempts retain only a
serializable `error_type` and `error_message`; traceback and exception objects
are not stored. A user-provided `alpha0` is reported as `user_provided`.

## MLE

MLE estimates one constant copula parameter. It is the default baseline for
family screening and for initializing dynamic methods.

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

For `StochasticStudentCopula(corr_mode='fixed')`, native `dL/ddf` is passed to
L-BFGS-B. Joint `df` and estimated-correlation MLE combines native `dL/ddf`
and the native Student correlation score, then maps the latter into the
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
state rather than a latent-state distribution. It is usually faster than SCAR
because there is no latent-state integral.

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

The compiled evaluator is the single GAS numerical implementation. It owns
the likelihood, score recursion, state updates, prediction, and bivariate
Rosenblatt path for supported built-in copulas. The Python layer owns
optimization orchestration, RNG, and sampling. Unsupported copulas and a
missing extension fail immediately; there is no backend selector or fallback.

`scaling='unit'` is the recommended production mode. `scaling='fisher'` uses
nested finite differences and clipping/floor thresholds; its fitted optimum
can be sensitive to optimizer finite-difference steps.

The GAS copula score and filtering recursion are native model calculations.
They are not the optimizer Jacobian with respect to
`(omega, gamma, beta)`. GAS passes objective values to L-BFGS-B, which
therefore computes that outer gradient numerically. Result
diagnostics distinguish these concepts with `model_score='native'` and
`optimizer_gradient='numerical'`.

## SCAR-TM-OU

SCAR-TM-OU uses an Ornstein-Uhlenbeck latent state,

$$
r_t = \Psi(x_t), \qquad
dx_t = \kappa(\mu - x_t)\,dt + \nu\,dW_t.
$$

The transfer-matrix likelihood integrates the latent Markov path
deterministically. By default, `transition_method='auto'` uses a Hermite
spectral likelihood except in narrow-kernel regimes, where it uses local
Gauss-Hermite. If spectral evaluation fails numerically, `auto` tries the
matrix grid likelihood first and then the local method when the matrix path is
not accepted.

SCAR-TM-OU uses the package C++ extension as its only production numerical
engine.

```python
result = fit(
    copula,
    u,
    method='scar-tm-ou',
    transition_method='auto',
    analytical_grad=True,
)
```

The fitted parameters are `kappa`, `mu`, and `nu`.

The exact OU one-step transition is Gaussian, so all SCAR-TM-OU likelihood
backends evaluate the same latent Markov model. They differ only in how the
one-dimensional transition integral is approximated: Hermite spectral
projection, a finite-grid transition matrix, or local Gauss-Hermite
quadrature.

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
backend and explicitly rejects `analytical_grad=True`.

`LatentResult.diagnostics` reports `gradient_requested`, `gradient_used`,
`gradient_kind`, `setup_derivative`, `filter_derivative`, and the transition
backend actually selected at the fitted parameters.

For high-frequency data, the code uses `dt = 1 / (T - 1)`. Large `T` therefore
produces very narrow one-step Jacobi transitions. In this regime the local
transition is often the stable and accurate default; increasing
`basis_order` can be useful as a diagnostic but is not usually needed for
routine fitting.

## Sampling, Prediction, and Diagnostics

`predictive_mean(copula, u, result)` returns the predictive mean copula
parameter at each time step before the current observation is absorbed.

For `predict`, SCAR-TM methods support `horizon='current'` for the posterior
state after the last observation and `horizon='next'` for the one-step-ahead
state. The shared prediction terminology is described in
[Prediction Semantics](prediction-semantics.md).
