# Numerical Backends

This page is the detailed reference for optimizer controls, transfer methods,
native kernels, memory guards, and vine fitting options. Start with
[Performance Tuning](performance.md) for the short decision guide and change a
backend option only when diagnostics identify the corresponding numerical
condition.

For the statistical meaning of each method, see
[Estimation Methods](estimation-methods.md).

For the complete native-thread, process-worker, determinism, thread-safety,
and dimensional-scaling contract, see [CPU Parallelism](parallelism.md).

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
described under [Multivariate Native Paths](#multivariate-native-paths).

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
| `gamma0` | fit kwarg | MLE-based | Initial $[\omega, \gamma, \beta]$. |
| `gtol` | fit kwarg / `gas_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `ftol` | fit kwarg / `gas_optimizer.ftol` | `1e-9` | Relative objective decrease tolerance. |
| `maxfun` | fit kwarg / `gas_optimizer.maxfun` | `4000` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `gas_optimizer.maxiter` | `1000` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `gas_optimizer.maxls` | `100` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `gas_optimizer.eps` | `1e-5` | L-BFGS-B finite-difference step. |
| `score_eps` | fit kwarg / `gas_score_eps` | `1e-4` | Finite-difference step for score calculations where needed. |
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

`ftol` matters for GAS because L-BFGS-B can otherwise report `success=True`
after a small relative objective decrease even when the gradient is still
large. If a GAS result looks sensitive to `gamma_bound` even though the fitted
$\gamma$ is far from the bound, rerun with tighter `ftol` and larger `maxfun`.

GAS uses the compiled numerical engine for likelihood, score recursion,
filtering, prediction, and Rosenblatt operations. Unsupported copulas fail
immediately.

The model score driving the GAS recursion is not an analytical gradient of the
complete likelihood with respect to `omega`, `gamma`, and `beta`. SciPy
L-BFGS-B obtains that optimizer gradient by finite differences.
`GASResult.diagnostics` records these as
`model_score='native'` and `optimizer_gradient='numerical'`.

Fisher scaling computes curvature by a second finite difference inside the GAS
recursion, while L-BFGS-B also differentiates the outer objective numerically.
Together with the Fisher floor and score clipping, this can produce a
piecewise, step-sensitive objective. Prefer `scaling='unit'` unless Fisher
behavior is specifically under study.

GAS `sample` and `predict` require a positive integer draw count. Both accept
`memory_budget_bytes=` as a pre-allocation guard; `sample` accounts for its
output and `predict` accounts for its output plus the predictive parameter
path. The causal GAS sample recursion is not split into batches.

### SCAR-TM-OU

SCAR-TM-OU uses a deterministic transfer-matrix likelihood for an OU latent
state. Unlike Monte Carlo likelihoods, repeated evaluation at fixed inputs
does not introduce simulation noise.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | smart/MLE-based | Initial $[\kappa, \mu, \nu]$. |
| `gtol` | fit kwarg / `scar_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. Larger values are faster but less precise. |
| `maxfun` | fit kwarg / `scar_optimizer.maxfun` | `300` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `scar_optimizer.maxiter` | `100` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `scar_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | L-BFGS-B finite-difference step for numerical-gradient fits. |
| `K` | strategy kwarg / `default_K` | `300` | Minimum latent grid size. May be increased by the adaptive rule. |
| `grid_range` | strategy kwarg / `default_grid_range` | `5.0` | Grid spans $[-\texttt{grid\_range}\,\sigma, +\texttt{grid\_range}\,\sigma]$. |
| `grid_method` | strategy kwarg / `default_grid_method` | `'auto'` | `'auto'`, `'dense'`, or `'sparse'`. Use sparse for large grids. |
| `adaptive` | strategy kwarg / `default_adaptive` | `True` | Enlarges `K` when the OU transition kernel needs more resolution. |
| `pts_per_sigma` | strategy kwarg / `default_pts_per_sigma` | `4` | Minimum grid points per conditional standard deviation. |
| `transition_method` | strategy kwarg | `'auto'` | `'auto'`, `'matrix'`, `'local'`, or `'spectral'`. See below. |
| `auto_small_kdt` | strategy kwarg | `1e-2` | In `transition_method='auto'`, use the local transition when $\kappa\,dt$ is below this value. |
| `spectral_basis_order` | strategy kwarg | `'auto'` | Hermite basis size for the spectral likelihood. The auto policy uses 128, 96, 64, or 32 from the current $\kappa\,dt$; pass an integer to fix the basis size. |
| `spectral_quad_order` | strategy kwarg | auto | Gauss-Hermite quadrature order for spectral multiplication. |
| `analytical_grad` | strategy kwarg | `True` | Uses the analytical gradient and avoids optimizer finite differences. |
| `smart_init` | strategy kwarg | `True` | Uses a heuristic initial point before falling back to MLE-based init. |
| `log_stationary_scale_optimization` | strategy kwarg | `None` | For bivariate models, `True` enables `[log(kappa), mu, log(sigma_x)]` with `kappa, sigma_x >= 0.001`; `False` forces scaled physical coordinates; `None` keeps the model default. |
| `stationary_scale_bounds` | strategy kwarg | `None` | Overrides model bounds for `sigma_x` in log-stationary coordinates. `StochasticStudentCopula` and bivariate models use separate policies, both defaulting to `(0.001, 10000.0)`. |

For `StochasticStudentCopula`, `alpha0` is still supplied as
`[kappa, mu, nu]`, but the optimizer internally uses
`[log(kappa), mu, log(sigma_x)]`, where
`sigma_x = nu / sqrt(2 * kappa)`. The result is converted back to
`[kappa, mu, nu]`. For a bivariate SCAR-TM-OU model this conditioning measure
is opt-in via `log_stationary_scale_optimization=True`. The log-coordinate
fit guarantees positive `nu` after conversion but does not apply the legacy
`nu >= 0.001` bound. `StochasticStudentCopula` uses
`0.001 <= sigma_x <= 10000`, while retaining final validation against the
physical `nu >= 0.001` bound and the other OU acceptance checks. Its
model-specific SCAR optimizer defaults are `maxiter=1000` and `maxls=200`;
they come from `stochastic_student_scar_optimizer`. Ordinary bivariate
SCAR-TM-OU fits use `bivariate_scar_optimizer`; when it is omitted, that field
resolves from `scar_optimizer` for backward compatibility. The
optional bivariate log-coordinate mode instead uses the independent
`bivariate_log_scar_optimizer`, whose defaults are `maxiter=1000` and
`maxls=200`. Other SCAR models retain their existing `scar_optimizer`
configuration. Inspect
`result.diagnostics['optimizer_parameterization']` when auditing fits.

```python
from pyscarcopula import LBFGSBConfig, NumericalConfig

config = NumericalConfig(
    scar_optimizer=LBFGSBConfig(maxiter=200),
    # Optional explicit override; otherwise this inherits scar_optimizer.
    bivariate_scar_optimizer=LBFGSBConfig(maxiter=300),
    bivariate_log_scar_optimizer=LBFGSBConfig(maxiter=1200),
    stochastic_student_scar_optimizer=LBFGSBConfig(maxiter=1500),
)

# Optional only for bivariate models; the default remains scaled [kappa, mu, nu].
result = fit(
    copula,
    u,
    method="scar-tm-ou",
    config=config,
    log_stationary_scale_optimization=True,
)
```

```python
result = fit(
    copula,
    u,
    method='scar-tm-ou',
    K=500,
    grid_method='sparse',
    transition_method='auto',
    gtol=5e-3,
    analytical_grad=True,
)
```

When `adaptive=True`, the grid is enlarged so the OU transition kernel is
resolved with at least `pts_per_sigma` points per conditional standard
deviation. For slow mean reversion this can produce large grids. If that is too
expensive, use `grid_method='sparse'`, reduce `pts_per_sigma`, or set
`adaptive=False` with an explicit `K`.

#### Compiled engine

The compiled engine implements the SCAR-TM-OU likelihood, analytical gradient,
grid forward quantities, and pointwise copula `h`/`h_inverse` kernels. No
backend argument is accepted.

SCAR-TM-OU likelihood and gradient support:

| Family | Rotations | Transform |
|--------|-----------|-----------|
| Clayton | 0, 90, 180, 270 | `softplus`, `xtanh` |
| Gumbel | 0, 90, 180, 270 | `softplus`, `xtanh` |
| Joe | 0, 90, 180, 270 | `softplus`, `xtanh` |
| Frank | 0 | `softplus`, `xtanh` |
| Independent | 0 | identity |
| Bivariate Gaussian | 0 | Gaussian tanh transform |
| Equicorr Gaussian | 0 | dimension-aware Gaussian tanh |
| Stochastic Student | 0 | shifted softplus df transform |

Pointwise `h`/`h_inverse` support is broader than likelihood support for some
families and transforms.

The likelihood accepts `transition_method='auto'`, `'spectral'`, `'matrix'`,
and `'local'`. Forward and prediction quantities are grid-based.

The compiled kernels impose direct numerical-configuration limits:

- `K <= 100000` for local and sparse-grid paths;
- `K <= 10000` for the dense matrix path;
- `basis_order`, `quad_order`, and `gh_order` must not exceed `1024`.

Observation count, Student dimension, and derived allocation sizes are not
artificially capped, with one exception: the multivariate Student PPF table
(see below). Their memory and runtime cost depend on the data size and
chosen numerical options.

#### Student PPF table memory cap

Multivariate Student models normally precompute a quantile (inverse-CDF) table
of shape `(n_df_nodes, T, d)` (about 360 df nodes by default) to speed up
emission-density evaluations. The table costs `n_nodes × T × d × 8` bytes and
is bounded by `DEFAULT_MAX_TABLE_BYTES` (256 MiB) from
`pyscarcopula.copula.multivariate.student_ppf_cache`. The default nodes include
a dense geometric layer above `df = 2 + 1e-6`, where the quantile changes
rapidly, and extend through `df = 1000`.

When the estimated size exceeds the limit, the values table is not built.
Python-level table calls then use exact `scipy.special.stdtrit` values. Native
dynamic-emission specs retain the df nodes even without the values table:
they use exact quantiles up to the final node and a third-order Cornish-Fisher
normal-quantile expansion above it. Static Student likelihoods carry no
dynamic node metadata and retain exact quantiles at all finite `df` values.

The native exact path obtains the quantile's `df` derivative by implicit
differentiation of the Student CDF. The large-`df` expansion has a matching
analytical derivative. Consequently, an analytical SCAR gradient outside the
cache no longer evaluates the full emission likelihood at perturbed `df`
values. Cached interpolation, exact evaluation, and the controlled asymptotic
are close but not bit-identical, so small likelihood or gradient differences
are possible at their boundaries.

`StudentPPFTable` accepts a `max_table_bytes` constructor argument for direct
internal table construction. `StochasticStudentCopula` currently uses the
package-wide default; there is no model-level fit or constructor option for
overriding this cap.

With `transition_method='auto'`, a `numerical_failure` from the spectral path
may be recovered by trying matrix and then local transition methods. Forced
transition methods raise an error instead of trying other transition methods.
Invalid configurations and unsupported combinations are not treated as
recoverable numerical fallbacks.

#### Transfer methods

SCAR-TM supports four likelihood transition modes:

- `transition_method='auto'` (default): use `spectral` except in the
  narrow-kernel regime, where it uses `local`; if the spectral likelihood
  fails numerically, it falls back to `matrix` first and then to `local` if the
  matrix result is not accepted.
- `transition_method='matrix'`: use the original transfer matrix on the latent
  grid for the likelihood.
- `transition_method='local'`: use the local Gauss-Hermite transition on the
  latent grid. This is useful when the OU transition kernel is very narrow.
- `transition_method='spectral'`: force the Hermite spectral likelihood.

The forward quantities used for prediction, mixture h-functions, and
Rosenblatt GoF still need a grid posterior state. If `spectral` is selected for
the likelihood, those forward passes use the grid `auto` fallback internally.

#### Matrix transfer likelihood

The matrix method is the most direct discretization of the latent-state
integral. It builds a grid for the OU state,

$$
x_j = \mu + z_j,\qquad
z_j \in [-R \sigma,\,
          R \sigma],
\qquad
\sigma^2 = \frac{\nu^2}{2\kappa},
$$
where $R$ is `grid_range`.

The OU transition over one observation step is Gaussian:

$$
X_k \mid X_{k-1}=x_i
\sim
N\left(
  \mu + \rho(x_i-\mu),
  \sigma^2(1-\rho^2)
\right),
\qquad
\rho=\exp(-\kappa\,dt).
$$

On the grid this becomes a weighted transition matrix

$$
T_{ji} \approx
p(x_j \mid x_i)\,w_j,
$$

where $w_j$ are trapezoidal quadrature weights. If

$$
f_k(x_j)=c(u_{1,k},u_{2,k};\Psi(x_j)),
$$

then the backward likelihood recursion has the form

$$
m_{k-1,i}
=
\sum_j T_{ji}\,f_k(x_j)\,m_{k,j}.
$$

The parameters `K`, `grid_range`, `adaptive`, and `pts_per_sigma` control the
state grid. `grid_method='dense'` stores the full matrix, while
`grid_method='sparse'` stores only a band around the Gaussian kernel. The
`grid_method='auto'` setting chooses between those two matrix layouts.

#### Local Gauss-Hermite transition

When $\kappa\,dt$ is very small, the conditional OU variance
$\sigma^2(1-\rho^2)$ is tiny. A fixed grid may then need many points to resolve
the narrow Gaussian transition kernel. The local Gauss-Hermite method avoids
building the full transition matrix. For each previous grid point $x_i$, it
approximates

$$
\int g(x)\,p(x\mid x_i)\,dx
\approx
\sum_{\ell=1}^{q}
\omega_\ell\,
g\left(
  \mu+\rho(x_i-\mu)+
  \sigma\sqrt{1-\rho^2}\,\sqrt{2}\,\xi_\ell
\right),
$$

where $\xi_\ell$ and $\omega_\ell$ are Gauss-Hermite nodes and weights and $q$ is
`gh_order`. The values at off-grid locations are interpolated from the latent
grid. This makes the transition local: the cost scales with $K\,q$
rather than with a dense $K \times K$ matrix.

The parameters `max_K` and `r_gh` are safeguards for this regime. With
`transition_method='auto'`, the grid path uses the local transition when the
adaptive grid would hit `max_K` or when the transition kernel is narrow
relative to the grid spacing. Increasing `gh_order` improves the local
quadrature but does not fix a poor latent grid; `K`, `grid_range`, and
`pts_per_sigma` still determine where the posterior can live.

#### Spectral Hermite likelihood

The spectral method uses stationarity of the OU process. Write

$$
X_t = \mu + \sigma Z_t,
\qquad
\sigma^2 = \frac{\nu^2}{2\kappa},
\qquad
Z_t \sim N(0, 1).
$$

For observations $u_k = (u_{1,k}, u_{2,k})$, define the emission factor

$$
f_k(z) =
c\left(u_{1,k}, u_{2,k}; \Psi(\mu + \sigma z)\right),
$$

where $c$ is the copula density and $\Psi$ maps the latent state to the valid
copula parameter range. The likelihood is the latent OU path integral

$$
L =
\int f_0(z_0)
    \prod_{k=1}^{T-1} p_\rho(z_k \mid z_{k-1}) f_k(z_k)
    \prod_{k=0}^{T-1} \phi(z_k)\,dz_k,
$$

with OU correlation

$$
\rho = \exp(-\kappa\,dt).
$$

Here $p_\rho$ is the OU transition density with respect to the standard normal
measure in $z$ coordinates. This is the measure in which the Hermite basis is
orthonormal.

In the orthonormal probabilists-Hermite basis
$\{\psi_n\}_{n \ge 0}$ under the standard normal measure, the OU transition is
diagonal:

$$
P_\rho \psi_n = \rho^n \psi_n.
$$

Each observation only requires projecting multiplication by $f_k$ back to the
truncated basis:

$$
a_{k-1,n}
=
\rho^n
\sum_m
\left\langle \psi_n, f_k \psi_m \right\rangle_\phi
a_{k,m}.
$$

The inner products are evaluated by Gauss-Hermite quadrature. Therefore the
high-dimensional latent integral is approximated by repeated multiplication of
small dense operators in Hermite coordinates, while the OU transition itself is
just diagonal scaling by $\rho^n$.

This is fastest when $\kappa\,dt$ is not too small: higher Hermite modes are
damped by $\rho^n$, so a moderate basis order is enough. When $\kappa\,dt$ is
very small, $\rho$ is close to one, high modes decay slowly, and a global grid
does not resolve the narrow transition kernel. The default `auto` mode sends the
narrow-kernel regime to `local`; all other regimes try `spectral`, with
`matrix` and then `local` as numerical fallbacks.

### SCAR-TM-JACOBI

SCAR-TM-JACOBI uses a Jacobi diffusion for Kendall's tau on `(0, 1)`. It is
available only for copulas with a Kendall-tau parameter mapping. The main
numerical difference from SCAR-TM-OU is that the transition is built on a
Jacobi quadrature grid in tau space instead of an OU grid in an unbounded
latent coordinate.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | smart/MLE-based | Initial $[\kappa, m, \xi]$. |
| `gtol` | fit kwarg / `scar_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `maxfun` | fit kwarg / `scar_optimizer.maxfun` | `300` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `scar_optimizer.maxiter` | `100` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `scar_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | L-BFGS-B finite-difference step. |
| `transition_method` | strategy kwarg | `'auto'` | `'auto'`, `'spectral_matrix'`, `'local'`, `'local_fixed'`, or `'spectral_coeff'`. |
| `transition_storage` | strategy kwarg | `'dense'` | Dense transition storage, or opt-in `'sparse'` storage for explicit `local` and `local_fixed` backends. |
| `stationarity_correction` | strategy kwarg | `'none'` | Experimental sparse moving-grid correction: `'none'`, `'mh'`, or `'ipfp'`. |
| `adaptive_quad_order` | strategy kwarg | `False` | Experimentally select and then freeze a sparse moving-grid quadrature order before optimization. |
| `adaptive_quad_orders` | strategy kwarg | `(48, 80, 128, 192, 384, 768)` | Strictly increasing candidate ladder for adaptive quadrature. |
| `adaptive_max_full_horizon_tv` | strategy kwarg | `0.02` | Maximum full-horizon total-variation error in the adaptive gate. |
| `adaptive_max_relative_variance_error` | strategy kwarg | `0.10` | Maximum relative stationary-variance error in the adaptive gate. |
| `adaptive_max_conditional_mean_rmse` | strategy kwarg | `1e-3` | Maximum conditional-mean RMSE in the adaptive gate. |
| `adaptive_max_lag_one_correlation_error` | strategy kwarg | `1e-2` | Maximum absolute lag-one correlation error in the adaptive gate. |
| `adaptive_require_pass` | strategy kwarg | `False` | Raise instead of returning an exhausted adaptive-order result when no candidate passes. |
| `spectral_basis_order` / `basis_order` | strategy kwarg | `32` | Number of Jacobi basis functions. |
| `spectral_quad_order` / `quad_order` | strategy kwarg | auto | Jacobi quadrature order; default is `max(2 * basis_order + 16, 48)`. |
| `analytical_grad` | strategy kwarg | `False` | Passes the Jacobi matrix-filter Jacobian to the optimizer. Fully analytical for `local_fixed`; semi-analytical for `local`, `spectral_matrix`, and `auto`. Not available with `spectral_coeff`. |
| `negative_mass_tol` | strategy kwarg | `1e-5` | Maximum spectral truncation noise that may be clipped and renormalized into a probability transition. Larger negative mass makes `auto` fall back and makes explicit `spectral_matrix` fail unless `clip_negative=True`. |
| `gh_order` | strategy kwarg | `5` | Gauss-Hermite order for the local Lamperti transition. |
| `theta_cap` | strategy kwarg | `None` | Optional cap on the copula parameter after mapping from tau. Useful for very high positive dependence. |
| `clip_negative` | strategy kwarg | `False` | Clips negative entries in the truncated spectral matrix before row normalization. Use mainly for diagnostics. |
| `kappa_bounds` | strategy kwarg | `(1e-3, 100.0)` | Bounds for mean-reversion speed. |
| `xi_bounds` | strategy kwarg | `(1e-3, 5.0)` | Bounds for Jacobi volatility. |
| `stationary_shape_max` | strategy kwarg | `500.0` | Rejects extremely concentrated stationary beta shapes. |
| `memory_budget_bytes` | strategy kwarg | `1 GiB` | Conservative pre-allocation limit for basis, transition, gradient, `T x K` emission, and simultaneous Python/native fixed-draw boundary buffers. |
| `sampling_method` | strategy kwarg | `'tm_grid'` | Unconditional sampler: likelihood-consistent `'tm_grid'` or experimental continuous `'lamperti_euler'`. |
| `lamperti_substeps` | strategy kwarg | `8` | Euler substeps per observation interval for `lamperti_euler`. |
| `lamperti_boundary` | strategy kwarg | `'reflect'` | Boundary policy for `lamperti_euler`: `'reflect'` or diagnostic `'clip'`. |
| `lamperti_eps` | strategy kwarg | `1e-10` | Interior epsilon used only to evaluate the singular Lamperti drift. |
| `lamperti_engine` | strategy kwarg | `'native'` | Mandatory fixed-draw C++ Lamperti--Euler engine; legacy `'numba'`/`'python'` labels normalize to `'native'`. |
| `lamperti_chunk_observations` | strategy kwarg | `4096` | Maximum complete observation intervals per Gaussian-innovation chunk. |
| `tau_eps` | strategy kwarg | `1e-6` | Keeps tau away from the endpoints. |
| `smart_init` | strategy kwarg | `True` | Tries an MLE-derived tau initial point and falls back to the fixed initializer if validation fails. |

```python
result = fit(
    copula,
    u,
    method='scar-tm-jacobi',
    transition_method='auto',
    basis_order=32,
)
```

Jacobi orders are strict positive integers: booleans and fractional values
are rejected instead of being silently converted. Empty data, non-finite
options, non-positive `theta_cap`, and invalid physical
`alpha0=[kappa, m, xi]` are rejected before optimization. All Jacobi orders
have a safety cap of `2048`; the default adaptive candidate ladder spans
orders `48` through `768`. Before the native Gauss-Jacobi eigensolver or
quadratic transition arrays are allocated, the native domain core performs
checked arithmetic for the simultaneous float64 workspace. The estimate
includes the full Golub--Welsch eigenvector peak for both the Jacobi grid and
the configured Gauss-Hermite order, less any larger dense transition/gradient
workspace already included in the same conservative peak. If
`memory_budget_bytes` is too small, it raises `MemoryError` with the
required-byte estimate and guidance to reduce the grid/basis order. The same
1 GiB default guard applies to direct numerical Jacobi entry points. Jacobi
parameter transforms, stationary Beta shapes, Gauss-Jacobi/Gauss-Hermite
rules, normalized basis recurrence, and Lamperti transforms are C++17-owned;
SciPy is not used on these production paths.

#### Jacobi transfer methods

`transition_method='auto'` first tries `spectral_matrix`. If the truncated
spectral matrix has negative mass above `negative_mass_tol`, or if spectral
matrix construction raises a floating-point error, `auto` uses `local`.
Forcing `transition_method='spectral_matrix'` keeps those numerical failures
visible and does not fall back. Spectral negative mass within the configured
tolerance is treated as truncation noise: it is clipped, rows are renormalized,
and the cleanup magnitude is reported in transition diagnostics. A material
negative mass in an explicitly requested spectral matrix raises
`FloatingPointError` unless `clip_negative=True`.

`transition_method='local_fixed'` uses a parameter-independent tau grid and is
the fully analytical backend for `analytical_grad=True`. The `local` and
`spectral_matrix` backends use finite differences for setup-level arrays and
analytical differentiation for the filtering recursion, so their reported
`gradient_kind` is `semi_analytical`. For `auto`, diagnostics record the
backend selected at the fitted parameters. Finite-difference setup
perturbations use that same selected backend, so a single gradient evaluation
cannot mix spectral and local objectives. Analytical-gradient fits also
recompute the ordinary objective at the final point and report
`final_objective_consistent` in result diagnostics.
`transition_method='spectral_coeff'` uses coefficient-space filtering instead
of a transition matrix. It is available for diagnostic comparisons and does
not support the analytical-gradient option.

#### Unconditional Jacobi sampling

By default, `sample()` reproduces the same discrete Markov model used by the
matrix likelihood. It draws the first state from the stationary quadrature masses,
builds the transition with `dt = 1 / (n - 1)`, converts the probability-safe
transition to row-wise CDFs in place, and advances quadrature-grid indices.
The resulting tau atoms are mapped through `tau_to_param`, including the
fitted `theta_cap`, before the existing copula sampler generates observations.

The selected sampling backend follows the fitted backend. For
`spectral_coeff`, which has no probability transition matrix, sampling
explicitly uses `auto` with the fitted basis and quadrature orders. Explicit
`spectral_matrix` sampling fails if its transition contains material negative
mass; signed rows are never converted with absolute values.

`n` must be a non-negative integer. `n=0` returns an empty sample without
advancing the supplied generator, and `n=1` performs only a stationary grid
draw. Transition construction is `O(K^2 B)` for the spectral backend or
`O(KG)` for local construction, the in-place CDF is `O(K^2)`, and path
generation is `O(n log K)`. Peak memory is conservatively checked before
transition or RNG allocation, including simultaneously live Python, binding,
native, and returned path buffers. Hyphenated transition aliases are
normalized before this preflight and before any RNG draw. The same
parameter-path implementation is used by dynamic edges during C-vine and
R-vine sampling.

`sampling_method='lamperti_euler'` enables an experimental continuous-path
alternative. It starts from the exact stationary beta law, applies
Euler--Maruyama with `lamperti_substeps` in

$$
y=\frac{2}{\xi}\arcsin\sqrt{\tau},
$$

and maps back with $\tau=\sin^2(\xi y/2)$. The default boundary policy
reflects an overshoot in constant time; `'clip'` is retained only for
numerical comparison. A mutable dictionary passed as
`sampling_diagnostics=` receives the intervention count and rate. This
sampler approximates the continuous SDE but is not the transition backend
used by fitting and is not an exact Wright--Fisher sampler.

The fitted result persists the selected sampling method, Lamperti settings,
and a non-default memory budget. Call-time keyword arguments can override
them. Use `tools/validate_jacobi_sampling.py` to compare stationary
mean/variance, KS/TV error, conditional first-moment error, interventions,
and runtime over independent path ensembles.

The state evolution is sequential because each Euler update depends on the
preceding state. Random draws remain in the Python orchestration layer and are
passed to C++ in bounded chunks of complete observation intervals.
Consequently, changing `lamperti_chunk_observations` does not change the path
or RNG state. External applications may parallelize independent paths only
with separate explicitly managed random streams.

Lamperti--Euler evolution now has one mandatory fixed-draw C++17 path. Python
generates the stationary Beta draw and bounded normal chunks; C++ owns the
drift, substeps, boundary policy, tau reconstruction, diagnostics, and exact
draw-consumption counters. Chunk sizing includes both Python/C++ normal
buffers and both native/NumPy result buffers in the memory peak. `n=1` has no
Euler work.

Accuracy near singular boundaries remains a separate gate. A symmetric
stationary law with `a=b=0.4` remained reasonably stable, but the extreme
asymmetric case `a=0.04`, `b=0.16` showed severe reflection bias even at 64
substeps. Diagnostics therefore expose `stationary_boundary_singular` and the
boundary-intervention rate. Lamperti--Euler remains opt-in; native execution
does not make it a universally valid default.

### Sparse local Jacobi transitions

The moving-grid local transition has at most `2 * gh_order` active targets
per source node. It can be selected explicitly for likelihood, filtering,
prediction, and state distributions:

```python
result = copula.fit(
    u,
    method="scar-tm-jacobi",
    transition_method="local",
    transition_storage="sparse",
)
```

This backend stores and applies the local transition in `O(K * gh_order)`
space and time per filtering step. It does not materialize a dense `K x K`
matrix. The default remains `transition_storage="dense"`, and sparse storage
currently requires an explicit `transition_method="local"` or
`transition_method="local_fixed"`. The fixed-grid sparse backend also stores
the three transition derivative arrays in `O(K * gh_order)` space and
supports the existing fully analytical gradient. Spectral transitions
continue to use their dense representation.

Experimental stationarity corrections are available only on the same
end-to-end sparse path:

```python
result = copula.fit(
    u,
    method="scar-tm-jacobi",
    transition_method="local",
    transition_storage="sparse",
    stationarity_correction="mh",
)
```

The MH correction is available only with the moving-grid `local` backend. It
enforces the quadrature stationary weights and detailed
balance, but may materially increase stay probabilities and distort
short-horizon dynamics. It is therefore opt-in and is not selected by
`auto`. Use `tools/validate_jacobi_sampling.py` to compare transition memory,
filter timings, full-horizon stationarity, conditional moments, and lag-one
correlation.

`stationarity_correction="ipfp"` instead balances the stationary joint flux
on the existing sparse support. It does not silently add diagonal edges or
regularization. Consequently, IPFP fails explicitly when the proposal
support cannot deliver incoming mass to every stationary node. Even when it
is feasible, its conditional-moment and autocorrelation distortion must be
checked. Current validation does not support either MH or IPFP as a default;
increasing `K` preserved short-horizon dynamics better in the tested matrix.

### Experimental adaptive Jacobi order

For an uncorrected sparse moving-grid `local` backend,
`adaptive_quad_order=True` evaluates a strictly increasing candidate ladder
before optimization. It selects the first order satisfying deterministic
full-horizon stationarity, variance, conditional-mean, and lag-one
correlation gates:

```python
result = copula.fit(
    u,
    method="scar-tm-jacobi",
    transition_method="local",
    transition_storage="sparse",
    adaptive_quad_order=True,
    adaptive_quad_orders=(48, 80, 128, 192, 384),
)
```

The selected order is frozen for every fitting evaluation and persisted as
`spectral_quad_order`. A second diagnostic evaluates the same frozen order at
the fitted parameters; it does not silently increase the sampling grid.
Candidate records and both initial/final gate results are stored in
`LatentResult.diagnostics`. This mode remains experimental while its default
thresholds are calibrated over a wider parameter and data matrix.
The validation tool's `--adaptive-calibration` option runs predefined
baseline, high-kappa, symmetric-boundary, and asymmetric-boundary cases over
multiple observation counts.
The current matrix shows strongly non-monotone requirements across parameter
and observation-count regimes: some boundary cases do not pass through
`K=384`, while the baseline may select `K=80` for shorter series but require
`K=384` at `n=400`. Consequently, no stationary-shape-only order heuristic is
used.

The local method applies a Gaussian step in the Lamperti coordinate

$$
y = \frac{2}{\xi}\arcsin\sqrt{\tau},
$$

then maps the Gauss-Hermite nodes back to tau and interpolates on the Jacobi
quadrature grid. It produces a nonnegative row-normalized transition matrix
and is selected by `auto` when the one-step transition is too narrow for the
spectral path.

The spectral matrix method uses the Jacobi eigenbasis of the diffusion. It can
be useful as a diagnostic, but for high-frequency data the code uses

$$
dt = \frac{1}{T-1}.
$$

Large `T` therefore makes the one-step transition close to a delta kernel.
Representing such a narrow kernel with a truncated global Jacobi series can
produce oscillations, negative entries, or invalid row sums. Increasing
`basis_order` may reduce the truncation error in some parameter regions, but it
also raises cost sharply and can worsen conditioning. In that regime, matching
the high-order spectral likelihood and the local likelihood at a fitted point
isolates backend approximation error. Leave `transition_method='auto'` unless
performing that comparison or reproducing a fixed-backend result.

## Multivariate Native Paths

Multivariate Gaussian, Student, stochastic Student, and equicorrelation models
use compiled row, grid, and conditional kernels. Several optimizations are
automatic and do not require strategy options:

Static Gaussian and Student fitting uses the shared multivariate MLE
orchestrator with independent final-point objective and gradient validation.
The public `method="mle"` label means a static model; `corr_mode` determines
whether correlation is supplied/plug-in or enters the joint native objective.
`shrinkage` maps one analytical correlation score to a raw logit parameter,
while `cholesky` pulls the native lower-triangle score back through the full
SPD parameterization. The Cholesky mode is guarded at `d <= 10` by default.

`fixed` retains the fast path: Gaussian prepares normal scores once and uses
their sample correlation, while Student uses Kendall preprocessing and then
optimizes only `df`. Factor mode uses Woodbury operators with `O(d*k + k^2)`
stored state. Gaussian and two-stage Student estimate loadings outside the
joint optimizer; joint static Student uses the native loading score and an
identified loading parameterization. None of the compact likelihood,
sampling, bootstrap, or persistence paths needs a dense `d*d` correlation.

- conditional Gaussian/Student sampling reuses the Schur-complement Cholesky
  factor when one correlation matrix is shared by all output rows;
- Student density workspaces are reused inside GAS, static likelihood, Monte
  Carlo batches, and across all grid nodes within each SCAR emission row;
- dense static and GAS Student Rosenblatt transforms in the parity-validated
  domain (`df >= 0.1`, symmetric unit-diagonal SPD correlation with condition
  number at most `1e4`) send all rows and either a scalar or row-specific `df`
  trajectory to one native call, reuse one correlation Cholesky factor, and
  parallelize independent rows when useful;
- equicorrelation grids compute per-row normal-score sums once and reuse them
  for every latent-grid node; prepared static and SCAR-OU evaluators retain
  those statistics across repeated objective calls;
- conditional binding inputs avoid an additional C++ copy when they are
  C-contiguous `float64` arrays.

The zero-copy condition concerns the native boundary. Python adapters may still
normalize arbitrary user inputs with `np.ascontiguousarray`; non-contiguous
arrays and other dtypes therefore require a temporary conversion. Numeric
inputs are always checked for finite values, so zero-copy removes memory traffic
but not the linear validation pass.

Row-specific `(n,d,d)` correlation arrays cannot share one Cholesky factor and
retain per-row factorization cost. Near-singular Student conditional covariance
matrices also retain the per-row jitter path to preserve numerical semantics.
For dense Student Rosenblatt, the adapter rejects unsupported capability
combinations before entering the kernel. Production execution is always
native: invalid inputs, unsupported capability combinations, an unavailable
extension, and native runtime failures raise deterministic exceptions. The
preserved SciPy implementation is a test oracle only and is never selected by
production dispatch.

## Independent Fit Parallelism

Use process-level parallelism for independent datasets, bootstrap replicas,
starting points, hyperparameter variants, or different model prototypes:

```python
from pyscarcopula import StochasticStudentCopula
from pyscarcopula.contrib import fit_independent

batch = fit_independent(
    StochasticStudentCopula(d=u_bootstrap[0].shape[1]),
    u_bootstrap,
    method="scar-tm-ou",
    fit_kwargs={"maxiter": 100},
    n_jobs=4,
)

models = batch.models
print(batch.diagnostics)
```

Every task reconstructs its own model from constructor-level structural
parameters and creates its own prepared evaluator during fitting. Fitted
state and transient caches from the prototype are not copied. A list of model
prototypes and a list of `fit_kwargs` can be supplied to run different models
or initial points in the same batch.

For static Gaussian and Student prototypes, reconstruction preserves
`corr_mode`, constructor `R`/`corr_base`, shrinkage initialization, Cholesky
guard settings, and factor identification settings. JSON persistence also
retains the fitted estimator, raw correlation parameters, and compact
loadings. Parametric-bootstrap refits therefore use the same correlation
policy as the source model.

Avoid accidental CPU oversubscription. With `n_jobs > 1`, omitted `n_threads`
means one native thread per worker. Passing `n_threads=2` or more explicitly
enables nested process/thread parallelism; this affects performance, not task
ownership or correctness. The same strict opt-in rule applies with one outer
worker: if `n_threads` is omitted, exactly one native thread is used.

The same policy applies to rolling `risk_metrics`. Each result leaf contains a
`diagnostics` mapping with `n_jobs`, `n_threads`,
`multiprocessing_start_method`, `nested_parallelism`, and the worker ownership
contract. Per-window `SeedSequence` objects keep results independent of chunk
partitioning. For `n_jobs=1`, sequential windows may reuse the caller's model,
but each fit invalidates and rebuilds its transient prepared state before the
next window.

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

Multivariate Student models use separate GAS optimizer defaults,
so changing them does not affect bivariate GAS fits or vine edges:

```python
cfg = NumericalConfig(
    stochastic_student_gas_optimizer=LBFGSBConfig(ftol=1e-9),
)
```

## Generic VineCopula

`VineCopula.fit` has an explicit `config` argument and forwards strategy
options through the shared edge-fitting core. With `structure=None`, the
Dissmann selector builds the structure; with a fixed `RVineMatrix`, fitting
starts directly from decoded trees and skips MST selection.

```python
from pyscarcopula import VineCopula
from pyscarcopula import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=3000, maxiter=3000))

vine = VineCopula(
    truncation_level=2,
    truncation_fill='independent',
    threshold=0.02,
    min_edge_logL=5.0,
)

vine.fit(
    u,
    method='gas',
    config=cfg,
    gamma_bound=30.0,
    ftol=1e-12,
)
```

Strategy-specific optimizer and numerical options are forwarded to every
non-independent, non-truncated edge selected for dynamic fitting. R-vine
structure controls are:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | instance value / `None` | Tree levels $\ge$ `truncation_level` are truncated. |
| `truncation_fill` | `'independent'` | Truncated trees become independent edges or MLE-only edges (`'mle'`). |
| `threshold` | `0.0` | Edges with $|\text{Kendall tau}| < \texttt{threshold}$ are made independent before fitting. |
| `min_edge_logL` | `None` | Fitted weak edges below the threshold are replaced by independence. |
| `structure_search` | `'beam'` | Conditional-structure search mode when `given_vars` is used. |
| `beam_width` | `4` | Number of partial candidate structures retained by beam search. |
| `transform_type` | instance value / `'softplus'` | Parameter transform used for Archimedean candidate copulas. |

For every vine structure, automatic family selection is MLE-based. `gtol`,
`ftol`, `gamma_bound`, `K`, and similar strategy controls affect the dynamic
edge refit after a family has been selected. If `method='gas'`, a too-loose
`ftol` can make some edges stop early with `success=True`; set `ftol=1e-12` and increase
`maxfun` for difficult edges.

Fitted generic vines use sequential native hot paths where the model contract permits
them. GAS unconditional sampling executes the row recursion and causal score
updates in one native call while preserving RNG and edge-update order. Repeated
SCAR-TM-OU prediction against unchanged fitted history reuses
pseudo-observations and terminal posterior state. A new `fit`, a changed
explicit history, or an edge replacement invalidates the relevant transient
cache. These optimizations do not parallelize edge or sample execution.
The Python reference sampler and native GAS executor consume the same
model-independent R-vine traversal plan, so matrix order, conditioned nodes,
and edge orientation have one authoritative representation.

The regular-vine native capability matrix is deliberately narrower than the
public `VineCopula` API:

| Operation | Native capability | Production fallback |
|-----------|-------------------|---------------------|
| Static unconditional sampling | Fixed C-, D-, or R-vine; exact built-in Independent, Clayton, Gumbel, Joe, Frank, or bivariate Gaussian edges; scalar parameters | Unsupported exact-type combinations are rejected |
| Suffix/DAG conditional execution | The same exact built-in families, including supported rotations/orientations and scalar or row-specific parameter paths | Unsupported active edges are rejected |
| Row log-density and conditional MCMC | The same exact built-in families with scalar, row-specific, or mixed parameter storage; MCMC selects bounded incremental or full recomputation before consuming RNG | Unsupported active edges are rejected |
| R-vine Rosenblatt and GoF | Static scalar fitted edges from the exact built-in family set | Unsupported exact-type combinations are rejected |

Support is exact-type only. Unknown subclasses cannot opt in through class
flags or similarly named Python methods. Validation or numerical failures
after entering a supported native call are reported and never retried in
Python.

`gof_test(..., bootstrap=True)` also supports fitted `VineCopula` models. Each
replication simulates from the captured fitted vine, optionally refits an
independent worker-owned vine with the same structure and fitting contract,
then evaluates the R-vine Rosenblatt transform and Cramer-von Mises statistic.
Independent `SeedSequence` streams make the bootstrap statistics deterministic
across `n_jobs`; omitted native thread settings resolve to one thread per
worker.

Static `VineCopula` sampling bounds temporary vectorized workspace by processing at
most 8192 rows at a time. Use `sample(..., batch_rows=...)` to trade throughput
against peak memory. `memory_budget_bytes=` checks the estimated output and
workspace requirement before allocation. Dynamic edge trajectories are not
split because their row order is part of the fitted time-series model.

Arbitrary conditional MCMC checks its complete adapter, binding, state,
log-density, node-cache, contribution-cache, and replay-draw footprint against
the internal memory budget before allocation or random-number consumption.
Draws are validated and transferred in bounded chunks. Empty batches retain
the same diagnostics schema without undefined acceptance-rate arithmetic.

When benchmarking dynamic vines, report `vine.fit_result.actual_methods` and
`vine.fit_result.fallback_count`: unsuccessful dynamic edge fits are retained
as MLE fallbacks and otherwise make GAS or SCAR timings look artificially fast.
For prediction caches, measure cold and warm calls separately.
