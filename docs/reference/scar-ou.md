# SCAR-TM-OU Numerics

## SCAR-TM-OU

SCAR-TM-OU uses a deterministic transfer-matrix likelihood for an OU latent
state. Unlike Monte Carlo likelihoods, repeated evaluation at fixed inputs
does not introduce simulation noise.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | smart/MLE-based | Initial $[\kappa, \mu, \nu]$. |
| `gtol` | fit kwarg / `scar_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. Larger values are faster but less precise. |
| `maxfun` | fit kwarg / `scar_optimizer.maxfun` | `300` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `scar_optimizer.maxiter` | `100` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `scar_optimizer.maxls` | `100` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | Absolute step for numerical-gradient fits when `finite_diff_rel_step` is unset; scaled for physical OU coordinates. Inactive for native analytical gradients. |
| `finite_diff_rel_step` | fit kwarg / selected optimizer config | `None` | Relative step in optimizer coordinates for numerical-gradient fits. A non-None value takes precedence over `eps`; a non-None fit kwarg overrides the config. Inactive for native analytical gradients. |
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
| `corr_gradient_block_bytes` | strategy kwarg | `67108864` (64 MiB) | Budget for three active blocks in grid-based correlation gradients; larger blocks reduce repeated forward passes. |
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

The correlation-gradient budget covers active entries in the density,
state-derivative and forward-history blocks. It does not cap total process
memory: PPF tables, transition operators, checkpoint vectors and allocator
capacity are additional. The matrix and local backends use this setting;
ordinary OU-only and spectral gradients keep their existing storage policy.
`corr_gradient_block_bytes=25165824` (24 MiB) reproduces the former block size.
The budget must hold at least one row, or `24 * effective_K` bytes. With
multiple workers, budget each worker separately. This setting does not change
transition support, grid resolution, backend selection or optimizer tolerances.
The setting is saved in fit diagnostics and restored for bootstrap refits;
an explicit constructor override takes precedence over the saved budget.

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

The analytical `matrix` and `local` gradients stream emissions through the
specialized native batch kernels, in blocks of at most `B = floor(2^20 / K)`
rows, and keep rolling backward states and OU sensitivities. The ordinary
gradient uses `O(B*K + K)` workspace in addition to the transition operator
(`O(K^2)` for dense matrix, `O(K * gh_order)` for local).

Joint Student correlation gradients accumulate posterior scores, with one
forward-state block and balanced checkpoint recomputation instead of a full
history. Their additional workspace is `O(B*K + K*p + K*L)`, where `p` is the
number of correlation parameters and `L = ceil(log2(max(1, T/B)))`.
Checkpointing adds at most `O(T*L)` emission/transition steps; it does not add
one transition per correlation parameter. Histories that fit in one block
need no recomputation. These are working-block sizes, not hard limits on `T`.

### Compiled engine

The compiled engine implements the SCAR-TM-OU likelihood, analytical gradient,
grid forward quantities, and pointwise copula `h`/`h_inverse` kernels. No
backend argument is accepted.

SCAR-TM-OU likelihood and gradient support:

| Family | Rotations | Transform |
|--------|-----------|-----------|
| Clayton | 0, 90, 180, 270 | `softplus`, `xtanh`, `exp`, `logistic` |
| Gumbel | 0, 90, 180, 270 | `softplus`, `xtanh`, `exp`, `logistic` |
| Joe | 0, 90, 180, 270 | `softplus`, `xtanh`, `exp`, `logistic` |
| Frank | 0 | `softplus`, `xtanh`, `exp`, `logistic` |
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

### Transfer methods

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

With `auto`, selection is repeated for every optimizer objective evaluation,
including trials rejected by the line search. The current trial's
`kappa / (T - 1) < auto_small_kdt` selects `local`; equality and larger values
try `spectral`. A recoverable spectral numerical failure tries `matrix`,
followed by `local` if the matrix fails or its grid-resolution policy rejects
it. Prepared evaluators do not pin the backend chosen at the initial point.
This applies to analytical and numerical objectives and to joint correlation
optimization. Explicit `matrix`, `local`, and `spectral` requests remain fixed.

`auto_small_kdt=0.01` is a time-scale threshold. Auto routing checks numerical
validity and does not compare likelihood values between backends. There is
no cross-backend likelihood tolerance in this selection policy.
Different valid approximations can introduce a likelihood jump when the
optimizer crosses the routing threshold. Increasing `gh_order` alone does
not remove the local method's interpolation error on a fixed latent grid.

The forward quantities used for prediction, mixture h-functions, and
Rosenblatt GoF still need a grid posterior state. If `spectral` is selected for
the likelihood, those forward passes use the grid `auto` fallback internally.

### Matrix transfer likelihood

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

### Local Gauss-Hermite transition

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

### Spectral Hermite likelihood

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

### Optimizer and grid diagnostics

Optimizer trials in `(log kappa, mu, log stationary_sigma)` can overflow
when converted to physical OU parameters even though the trial coordinates
are finite. Such conversion failures receive the finite optimization penalty
and are counted as `invalid_parameter_trials`. Unsupported kernels and
invalid final parameter conversions still raise their original errors.

Sparse matrix transitions retain the historical five conditional Gaussian
standard deviations on either side of each transition center, covering about
99.99994% of the continuous standard-normal mass. Likelihood and analytical
gradient use the same support rule. The integer band can change with the OU
parameters; it is not widened by the optimizer. Transition support and
adaptive-grid resolution are separate settings.

Bootstrap refits retry once on the same simulated sample after an unsuccessful
fit. The retry starts from the failed candidate when its parameters are finite
and retains the original optimizer settings, including explicit `maxls`
overrides. The shared SCAR default is `maxls=100` from the first attempt,
both in ordinary fits and bootstrap refits. Convergence tolerances, transition
support and backend selection are unchanged. A failed retry still raises an error.

Native likelihood information reports `K_requested`, `K_effective`, and
`grid_was_capped` for grid backends. Fit diagnostics retain their final values
with the `last_` prefix and count `grid_capped_evaluations`. Spectral
evaluations have no grid size. A caller-selected matrix backend can retain
`max_K` points while the adaptive rule requests more; this does not increment
`matrix_capped`, which specifically counts fallback decisions.

Factor Student emissions keep exact quantiles and bounded tile storage.
Consecutive grid nodes that map to exactly the same floating-point degrees
of freedom share their density and degrees-of-freedom derivative, with the
latent-state derivative applied separately at every node. This avoids
repeated quantile solves in the saturated negative tail of the Softplus
transform without introducing an interpolated full-sample cache.
