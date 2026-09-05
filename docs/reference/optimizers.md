# Optimizer Controls

## Bivariate Models

All bivariate fits go through the strategy registry:

```python
from pyscarcopula import JoeCopula
from pyscarcopula.api import fit

copula = JoeCopula(rotate=180)

# Default GAS fit.
result = fit(copula, u, method='gas')

# Tighten or relax optimizer controls for this run.
result = fit(
    copula,
    u,
    method='gas',
    gamma_bound=30.0,
    beta_bound=0.995,
    gtol=1e-4,
    ftol=1e-12,
    maxfun=3000,
)
```

The object method is equivalent:

```python
result = copula.fit(u, method='scar-tm-ou', K=500, gtol=5e-3)
```

### MLE

MLE labels a static model. For scalar bivariate families it estimates one
constant copula parameter; the table in this subsection applies to that
optimizer. Static multivariate Gaussian and Student correlation controls are
described under [Multivariate Native Paths](../guide/numerical-backends.md#multivariate-native-paths).

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | auto | Initial point in the copula's natural parameter space. |
| `gtol` | fit kwarg / `mle_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `maxls` | fit kwarg / `mle_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |

```python
from pyscarcopula import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(mle_optimizer=LBFGSBConfig(gtol=1e-6))
result = fit(copula, u, method='mle', config=cfg)
```

MLE evaluates the likelihood directly in the natural copula parameter. For
example, `alpha0=[2.0]` means a Gumbel parameter of `2.0`, while a stochastic Student
initial value of `5.0` means five degrees of freedom. Parameter transforms are
used by dynamic latent-state methods, not by the MLE objective.

When `alpha0` is omitted for a bivariate copula, the implementation calls
`copula.transform([1.5])` only to obtain a family-valid natural starting
value. The returned value is passed directly to the MLE optimizer; it is not
treated as a latent coordinate.

### GAS

GAS estimates an observation-driven recursion
$g_t = \omega + \beta g_{t-1} + \gamma\,score_{t-1}$.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `gamma0` | fit kwarg | Two MLE-based starts | Explicit initial $[\omega, \gamma, \beta]$ selects a single start. |
| `gtol` | fit kwarg / `gas_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `ftol` | fit kwarg / `gas_optimizer.ftol` | `1e-9` | Relative objective decrease tolerance. |
| `maxfun` | fit kwarg / `gas_optimizer.maxfun` | `4000` | Maximum scalar objective evaluations, including numerical-gradient probes. |
| `maxiter` | fit kwarg / `gas_optimizer.maxiter` | `1000` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `gas_optimizer.maxls` | `100` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `gas_optimizer.eps` | `1e-8` | Absolute native two-point step for the outer optimizer gradient when `finite_diff_rel_step` is unset. |
| `finite_diff_rel_step` | fit kwarg / `gas_optimizer.finite_diff_rel_step` | `None` | Relative two-point step in optimizer coordinates; takes precedence over `eps`. |
| `score_eps` | fit kwarg / `gas_score_eps` | `1e-4` | Finite-difference step for Fisher curvature. |
| `gamma_bound` | fit kwarg / `gas_gamma_bound` | `20.0` | Bounds score sensitivity to $[-\texttt{gamma\_bound}, \texttt{gamma\_bound}]$. |
| `beta_bound` | fit kwarg / `gas_beta_bound` | `0.999` | Bounds persistence to $[-\texttt{beta\_bound}, \texttt{beta\_bound}]$; must be in $(0, 1)$. |
| `scaling` | strategy kwarg | `'unit'` | Recommended score scaling mode. `'fisher'` is numerically sensitive. |

```python
result = fit(
    JoeCopula(rotate=180),
    u,
    method='gas',
    scaling='unit',
    ftol=1e-12,
    maxfun=3000,
)
```

Automatic fitting tries the standard MLE-based score-driven start and a
nested static start with `gamma=0`. The latter shares the standard start's
intercept and persistence and exactly reproduces the static MLE path. An explicit `gamma0`
retains single-start semantics, including joint shrinkage fits. When `ftol`
is omitted, the best candidate is refined with `ftol=1e-12`. Each start and
refinement has its own `maxfun` and `maxiter` budget; `nfev` sums the runs.
The best finite terminal candidate is retained even if a worse candidate
reported convergence. A trial point that improves log likelihood by more
than `0.001` is retained with `success=False` if convergence there was not
established.

`success` additionally requires the optimizer's objective to agree with the
reported likelihood within `1e-6`, and the likelihood to be no more than
`0.001` below either the initial nested static likelihood (automatic starts)
or the constant path at the final intercept and persistence. Joint Student
shrinkage fitting preserves the quantile interpolation cache so optimization
and reporting evaluate the same function.

`GASResult.diagnostics` includes `optimizer_stages` (starts, final parameters,
objectives, raw convergence messages and evaluation counts),
`optimizer_success`, `optimizer_message`, `projected_gradient_inf_norm`,
`static_baseline_log_likelihood`, `objective_discrepancy`, and
`likelihood_validation_passed`. The projected gradient is informational; a
small relative function decrease can still occur with a large gradient.
Neither these checks nor multistart establish global optimality or guarantee
agreement with another version to `0.001`. Inspect alternative starts and
finite-difference steps for sensitive fits. The smaller default `eps` limits
perturbations amplified by the recursion; explicit `eps` and
`finite_diff_rel_step` retain their documented precedence and behavior.

GAS uses the compiled numerical engine for likelihood, score recursion,
filtering, prediction, and Rosenblatt operations. Unsupported copulas fail
immediately.

The model score driving the GAS recursion is not an analytical gradient of the
complete likelihood with respect to `omega`, `gamma`, and `beta`. The compiled
evaluator obtains that optimizer gradient by two-point finite differences and
returns it together with the objective to SciPy L-BFGS-B. It follows SciPy's
forward-step convention, adjusts steps at parameter bounds, and divides by
the actual representable displacement. An explicit relative step follows
the sign and magnitude of the optimizer coordinate.
`GASResult.diagnostics` records these as `model_score='native'`,
`optimizer_gradient='native'`, and
`gradient_kind='native_finite_difference'`. `maxfun` and `nfev` use scalar
objective budget units, preserving the previous counts for completed
finite-difference calls. Each native objective/gradient call is charged four
units, or five for joint GAS/shrinkage fitting. An early numerical failure
may execute fewer likelihood evaluations but still incurs that full charge;
`nfev` therefore does not count physical likelihood calls. As with SciPy
numerical gradients, an iteration or line search can exceed `maxfun`.

Fisher scaling uses the analytical copula score
$\partial\log c/\partial r\,\partial r/\partial g$ as its numerator and
computes only the local curvature by a second finite difference inside the GAS
recursion. The native evaluator still differentiates the outer objective
numerically. Together with the Fisher floor and score clipping, this can
produce a piecewise, step-sensitive objective. Prefer `scaling='unit'` unless
Fisher behavior is specifically under study.

Post-fit API operations inherit `GASResult.scaling` unless an explicit
`scaling=` override is supplied. The override applies to likelihood,
filtering, sampling, and prediction without changing the fitted result.
Likewise, `GASStrategy()` inherits the result's scaling for post-fit calls;
its fit default remains `unit`. Overriding scaling when predicting requires
the observation history, since the cached final parameter belongs to the
original scaling.

GAS `sample` and `predict` require a positive integer draw count. Both accept
`memory_budget_bytes=` as a pre-allocation guard. Fused bivariate `sample`
accounts for its caller-owned RNG draws, native result staging, and final
NumPy output: `6 * n * sizeof(float64)` bytes. Other sample paths account for
their output. `predict` accounts for its output plus the predictive parameter
path. Unconditional bivariate sampling performs its causal score recursion in
one fused native call; conditional and multivariate sampling retain the
stepwise model-specific path. The causal GAS sample recursion is not split
into batches.

## NumericalConfig

Use `NumericalConfig` when a setting should apply to many fits:

`NumericalConfig()` always uses `n_threads=1`, independently of process
environment variables. Native parallelism is enabled only by explicitly
passing `n_threads` to a method or by constructing
`NumericalConfig(n_threads=N)`.

```python
from pyscarcopula import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(
        gtol=1e-4,
        ftol=1e-12,
        maxfun=3000,
        maxiter=3000,
        maxls=50,
    ),
    scar_optimizer=LBFGSBConfig(
        gtol=1e-4,
        maxls=50,
    ),
    default_K=500,
)

result = fit(copula, u, method='gas', config=cfg)
```

Per-call keyword arguments override the config values for that fit.

For conditional bivariate `api.sample` and `api.predict`,
`bisection_tol` and `bisection_maxiter` configure the iterative inverse-h
solvers used by Gumbel and Joe (including rotations). The default tolerance
is `1e-10`, and the iteration limit is `60`. Gumbel tests a relative
transformed-equation residual; Joe tests a `log(-log(h))` residual or a
certified root bracket at float64 resolution. Tolerance is an upper bound:
the kernels can apply a tighter criterion to preserve tail accuracy.
Exhausting the iteration budget raises an error instead of returning an
unchecked approximation. These settings do not affect analytical inverse-h
families or unconditional sampling. Direct low-level inverse-h calls use
80 iterations and `8e-15` for Gumbel, and 50 iterations and `1e-10` for Joe.

GAS fit uses `config.fail_value` when a numerical objective evaluation raises
`FloatingPointError`, including joint shrinkage fits. Argument and programming
errors such as `TypeError` and `ValueError` propagate to the caller.

Multivariate Student models use separate GAS optimizer defaults,
so changing them does not affect bivariate GAS fits or vine edges:

```python
cfg = NumericalConfig(
    stochastic_student_gas_optimizer=LBFGSBConfig(ftol=1e-9),
)
```
