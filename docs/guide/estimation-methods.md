# Estimation Methods

This page describes the bivariate fitting methods exposed by
`pyscarcopula.api.fit`. Performance controls for these methods are covered in
[Performance Tuning](performance.md).

## Method Summary

| Method | Key | State | Main use |
|--------|-----|-------|----------|
| MLE | `'mle'` | Constant copula parameter | Baseline fit and family selection |
| GAS | `'gas'` | Observation-driven score recursion | Fast dynamic dependence without latent integration |
| SCAR-TM-OU | `'scar-tm-ou'` | OU latent state mapped to the copula parameter | Deterministic stochastic-latent likelihood |
| SCAR-TM-JACOBI | `'scar-tm-jacobi'` | Jacobi diffusion for Kendall's tau | Bounded tau dynamics with deterministic filtering |
| SCAR-MC-OU | `'scar-p-ou'`, `'scar-m-ou'` | OU latent state | Monte Carlo alternatives for SCAR experiments |

All dynamic methods return a `LatentResult` with `params`,
`log_likelihood`, optimizer status, and enough metadata for `predict`,
`predictive_mean`, and GoF utilities. Model sampling is available where the
strategy implements a path simulator; SCAR-TM-JACOBI currently supports
prediction but not `sample`.

## MLE

MLE estimates one constant copula parameter. It is the default baseline for
family screening and for initializing dynamic methods.

```python
from pyscarcopula import GumbelCopula
from pyscarcopula.api import fit

copula = GumbelCopula(rotate=180)
result = fit(copula, u, method='mle')
```

## GAS

GAS is observation-driven. The copula parameter is

$$
r_t = \Psi(g_t),
$$

where the unbounded recursion state follows

$$
g_{t+1} = \omega + \beta g_t + \gamma s_t.
$$

It is usually faster than SCAR because there is no latent-state integral.

```python
result = fit(copula, u, method='gas', ftol=1e-12, maxfun=3000)
```

## SCAR-TM-OU

SCAR-TM-OU uses an Ornstein-Uhlenbeck latent state,

$$
r_t = \Psi(x_t), \qquad
dx_t = \kappa(\mu - x_t)\,dt + \nu\,dW_t.
$$

The transfer-matrix likelihood integrates the latent Markov path
deterministically. By default, `transition_method='auto'` uses a Hermite
spectral likelihood except in narrow-kernel regimes, where it uses local
Gauss-Hermite. If spectral evaluation fails numerically, `auto` falls back to
the local method.

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

## SCAR-TM-JACOBI

SCAR-TM-JACOBI evolves Kendall's tau directly on `(0, 1)`:

$$
d\tau_t =
\kappa(m - \tau_t)\,dt
+ \xi\sqrt{\tau_t(1-\tau_t)}\,dW_t.
$$

The copula parameter is recovered from tau through the copula's
`tau_to_param` mapping. This method is therefore available for copulas that
implement both `tau_to_param` and `param_to_tau`; currently this includes
Gumbel, Clayton, Frank, Joe, and bivariate Gaussian copulas.

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

`transition_method='auto'` first tries a Jacobi spectral transition matrix on
the tau quadrature grid. If the truncated spectral matrix has material
negative mass, or if row normalization fails numerically, it falls back to the
local Lamperti/Gauss-Hermite transition. The default tolerance for accepting
small spectral truncation errors is `negative_mass_tol=1e-5`.

The explicit Jacobi transition backends are `spectral_matrix`, `local`,
`local_fixed`, and `spectral_coeff`. `local_fixed` is intended for
`analytical_grad=True`; `spectral_coeff` is a coefficient-space comparison
backend and does not support analytical gradients.

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
