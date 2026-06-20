# Performance Tuning

This page summarizes the knobs that affect fitting speed and optimizer
stability. The same strategy options are used by standalone bivariate copulas
and by pair-copula edges inside C-vines and R-vines.

For the statistical meaning of each method, see
[Estimation Methods](estimation-methods.md). This page focuses on runtime,
optimizer, and numerical-stability controls.

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

MLE estimates one constant copula parameter.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `alpha0` | fit kwarg | auto | Initial point in the copula's natural parameter space. |
| `gtol` | fit kwarg / `mle_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `maxls` | fit kwarg / `mle_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |

```python
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(mle_optimizer=LBFGSBConfig(gtol=1e-6))
result = fit(copula, u, method='mle', config=cfg)
```

MLE evaluates the likelihood directly in the natural copula parameter. For
example, `alpha0=[2.0]` means a Gumbel parameter of `2.0`, while a Student
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
| `ftol` | fit kwarg / `gas_optimizer.ftol` | `1e-12` | Relative objective decrease tolerance. Use a tight value to avoid premature FACTR convergence. |
| `maxfun` | fit kwarg / `gas_optimizer.maxfun` | `1000` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `gas_optimizer.maxiter` | `1000` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `gas_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `gas_optimizer.eps` | `1e-5` | L-BFGS-B finite-difference step. |
| `score_eps` | fit kwarg / `gas_score_eps` | `1e-4` | Finite-difference step for score calculations where needed. |
| `gamma_bound` | fit kwarg / `gas_gamma_bound` | `20.0` | Bounds score sensitivity to $[-\texttt{gamma\_bound}, \texttt{gamma\_bound}]$. |
| `beta_bound` | fit kwarg / `gas_beta_bound` | `0.999` | Bounds persistence to $[-\texttt{beta\_bound}, \texttt{beta\_bound}]$; must be in $(0, 1)$. |
| `scaling` | strategy kwarg | `'unit'` | Recommended score scaling mode. `'fisher'` is experimental and numerically sensitive. |

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

GAS evaluates deterministic numerical work in C++: likelihood, score
recursion, filtering, state updates, prediction, and Rosenblatt operations.
Python retains SciPy optimizer orchestration, RNG, and sampling. Official
wheels must contain `pyscarcopula._scar_cpp`; source builds require a C++17
compiler. Unsupported copulas fail immediately because GAS has no Python
numerical fallback.

The native score is the model score driving the GAS recursion. It is not an
analytical gradient of the complete likelihood with respect to
`omega`, `gamma`, and `beta`. SciPy L-BFGS-B currently obtains that optimizer
gradient by finite differences. `GASResult.diagnostics` records these as
`model_score='native'` and `optimizer_gradient='numerical'`.

Fisher scaling computes curvature by a second finite difference inside the GAS
recursion, while L-BFGS-B also differentiates the outer objective numerically.
Together with the Fisher floor and score clipping, this can produce a
piecewise, step-sensitive objective. Prefer `scaling='unit'` unless Fisher
behavior is specifically under study.

### SCAR-TM-OU

SCAR-TM-OU uses a deterministic transfer-matrix likelihood for an OU latent
state. It is usually the best stochastic model when reproducible likelihoods
and predictive paths are needed.

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
| `analytical_grad` | strategy kwarg | `True` | Uses analytical gradient and parameter rescaling. Usually much faster. |
| `smart_init` | strategy kwarg | `True` | Uses a heuristic initial point before falling back to MLE-based init. |

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

#### Native engine

The bundled pybind11 extension implements the SCAR-TM-OU likelihood,
analytical gradient, grid forward quantities, and pointwise copula
`h`/`h_inverse` kernels. It is the only SCAR-TM-OU production engine; no
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

Pointwise C++ `h`/`h_inverse` kernels are broader: they also support the
independence copula, xtanh transforms for Clayton/Gumbel/Joe/Frank where the
Python copula exposes them, Frank only with rotate=0, and Gaussian only with
rotate=0.

The native likelihood accepts `transition_method='auto'`, `'spectral'`,
`'matrix'`, and `'local'`. Forward/state calls are grid-based. When
`'spectral'` is requested, posterior quantities use an explicit native grid
reconstruction.

The C++ kernels impose only direct numerical-configuration limits:

- `K <= 100000` for local and sparse-grid paths;
- `K <= 10000` for the dense matrix path;
- `basis_order`, `quad_order`, and `gh_order` must not exceed `1024`.

Observation count, Student dimension, and derived allocation sizes do not have
administrative caps. Their memory and runtime cost are the caller's
responsibility. Size products are still checked for integer overflow before
allocation, and invalid shapes or non-finite inputs are still rejected.

The C++ Hermite-rule cache is a process-wide, mutex-protected LRU cache.
Compile-time defaults limit it to 16 rules and 8 MiB of retained node, weight,
basis, and weighted-basis vector storage. Rules larger than the byte budget
are computed for the current call but are not cached.

The native kernels use these status codes:

| Status | Name | Meaning |
|--------|------|---------|
| `0` | `ok` | The requested calculation completed successfully. |
| `1` | `null_pointer` | A required pointer was null. This is primarily relevant to direct native entry points. |
| `2` | `invalid_size` | A shape, order, grid size, or checked size product is invalid or cannot be represented safely. |
| `3` | `invalid_family` | The requested operation is not valid for the selected copula family. |
| `4` | `invalid_rotation` | The rotation value is invalid. |
| `5` | `invalid_transform` | The parameter transform is invalid or unsupported for the selected family. |
| `6` | `invalid_parameter` | An OU, copula, or numerical parameter is non-finite or outside its valid domain. |
| `7` | `numerical_failure` | A finite, normalizable numerical result could not be produced. |

The Python wrapper validates common shape and configuration errors before
entering C++ and raises `ValueError` for those failures. Unsupported backend
combinations raise `CppUnsupported`; non-zero native statuses raise
`CppError` with the numeric status and symbolic name. A native
`std::bad_alloc` is translated to Python `MemoryError`.

With `transition_method='auto'`, a `numerical_failure` from the spectral path
may be recovered by trying matrix and then local transition methods. Forced
transition methods expose the failure as `CppError`. Invalid configurations
and unsupported combinations are not treated as recoverable numerical
fallbacks.

Because observation count, Student dimension, and derived allocation sizes
are intentionally uncapped, a sufficiently large user-provided input may
exhaust memory or be terminated by the operating system before Python can
raise `MemoryError`. The direct `K` and quadrature-order limits above, integer
overflow checks, and input validation still apply.

For native safety testing, Linux builds can opt into strict compiler warnings
and AddressSanitizer/UndefinedBehaviorSanitizer:

```bash
PYSCA_CPP_STRICT=1 PYSCA_CPP_SANITIZE=1 \
  python -m pip install -e ".[test]"
LD_PRELOAD="$(gcc -print-file-name=libasan.so)" \
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  python -m pytest tests/test_cpp.py -m "not benchmark"
```

`PYSCA_CPP_SANITIZE` requires a GCC- or Clang-compatible platform. The
repository's `Native safety` GitHub Actions workflow also runs direct pybind,
limit-adjacent, forward/filter, fitting, and multivariate Student regression
tests under both sanitizers.

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
very small, $\rho$ is close to one, high modes decay slowly, and the grid local
path is usually safer. This is why the default `auto` mode sends the
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
| `spectral_basis_order` / `basis_order` | strategy kwarg | `32` | Number of Jacobi basis functions. |
| `spectral_quad_order` / `quad_order` | strategy kwarg | auto | Jacobi quadrature order; default is `max(2 * basis_order + 16, 48)`. |
| `analytical_grad` | strategy kwarg | `False` | Passes the Jacobi matrix-filter Jacobian to the optimizer. Fully analytical for `local_fixed`; semi-analytical for `local`, `spectral_matrix`, and `auto`. Not available with `spectral_coeff`. |
| `negative_mass_tol` | strategy kwarg | `1e-5` | Maximum accepted negative mass from the truncated spectral transition in `auto`. |
| `gh_order` | strategy kwarg | `5` | Gauss-Hermite order for the local Lamperti transition. |
| `theta_cap` | strategy kwarg | `None` | Optional cap on the copula parameter after mapping from tau. Useful for very high positive dependence. |
| `clip_negative` | strategy kwarg | `False` | Clips negative entries in the truncated spectral matrix before row normalization. Use mainly for diagnostics. |
| `kappa_bounds` | strategy kwarg | `(1e-3, 100.0)` | Bounds for mean-reversion speed. |
| `xi_bounds` | strategy kwarg | `(1e-3, 5.0)` | Bounds for Jacobi volatility. |
| `stationary_shape_max` | strategy kwarg | `500.0` | Rejects extremely concentrated stationary beta shapes. |
| `tau_eps` | strategy kwarg | `1e-6` | Keeps tau away from the endpoints. |
| `smart_init` | strategy kwarg | `True` | Uses an MLE-based tau initial point when possible. |

```python
result = fit(
    copula,
    u,
    method='scar-tm-jacobi',
    transition_method='auto',
    basis_order=32,
)
```

#### Jacobi transfer methods

`transition_method='auto'` first tries `spectral_matrix`. If the truncated
spectral matrix has negative mass above `negative_mass_tol`, or if spectral
matrix construction raises a floating-point error, `auto` uses `local`.
Forcing `transition_method='spectral_matrix'` keeps those numerical failures
visible and does not fall back.

`transition_method='local_fixed'` uses a parameter-independent tau grid and is
the fully analytical backend for `analytical_grad=True`. The `local` and
`spectral_matrix` backends use finite differences for setup-level arrays and
analytical differentiation for the filtering recursion, so their reported
`gradient_kind` is `semi_analytical`. For `auto`, diagnostics record the
backend selected at the fitted parameters.
`transition_method='spectral_coeff'` uses coefficient-space filtering instead
of a transition matrix; it is kept for comparisons and does not support the
analytical-gradient option.

The local method applies a Gaussian step in the Lamperti coordinate

$$
y = \frac{2}{\xi}\arcsin\sqrt{\tau},
$$

then maps the Gauss-Hermite nodes back to tau and interpolates on the Jacobi
quadrature grid. It produces a nonnegative row-normalized transition matrix
and is usually the stable choice when one-step transitions are very narrow.

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
is a good diagnostic; routine fitting should normally leave
`transition_method='auto'`.

### SCAR-MC

The Monte Carlo SCAR strategies are stochastic likelihood estimators.

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `n_tr` | strategy kwarg / `default_n_tr` | `500` | Number of Monte Carlo trajectories. |
| `M_iterations` | strategy kwarg | `3` | EIS iterations for `scar-m-ou`. |
| `stationary` | strategy kwarg | `True` | Initializes the OU process in stationarity. |
| `seed` / `dwt` | fit kwarg | random | Controls Wiener increments for reproducibility. |
| `gtol` | fit kwarg / `scar_optimizer.gtol` | `1e-3` | L-BFGS-B projected-gradient tolerance. |
| `maxfun` | fit kwarg / `scar_optimizer.maxfun` | `300` | Maximum function evaluations. |
| `maxiter` | fit kwarg / `scar_optimizer.maxiter` | `100` | Maximum optimizer iterations. |
| `maxls` | fit kwarg / `scar_optimizer.maxls` | `20` | Maximum L-BFGS-B line-search steps per iteration. |
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | L-BFGS-B finite-difference step. |

```python
result = fit(
    copula,
    u,
    method='scar-m-ou',
    n_tr=1000,
    M_iterations=5,
    seed=123,
)
```

## NumericalConfig

Use `NumericalConfig` when a setting should apply to many fits:

```python
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

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

## C-Vines

`CVineCopula.fit` selects a family for each edge using an MLE screening/refine
step, then optionally refits selected edges with the requested dynamic method.
All remaining keyword arguments are forwarded to the pair-copula strategy for
the dynamic refit.

```python
from pyscarcopula import CVineCopula
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=3000, maxiter=3000))

vine = CVineCopula()
vine.fit(
    u,
    method='gas',
    config=cfg,
    gamma_bound=30.0,
    ftol=1e-12,
    truncation_level=2,
    min_edge_logL=10.0,
)
```

For `method='mle'`, the edge stays at the MLE selection result. For dynamic
methods such as `'gas'` or `'scar-tm-ou'`, these controls are forwarded to each
dynamic edge fit:

- GAS: `gamma0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`, `eps`, `score_eps`,
  `gamma_bound`, `beta_bound`, `scaling`, `verbose`
- SCAR-TM-OU: `alpha0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`, `eps`, `K`, `grid_range`, `grid_method`, `adaptive`,
  `pts_per_sigma`, `transition_method`, `max_K`, `r_gh`, `gh_order`,
  `auto_small_kdt`,
  `spectral_basis_order`, `spectral_quad_order`, `analytical_grad`,
  `smart_init`, `verbose`
- SCAR-TM-JACOBI: `alpha0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`,
  `eps`, `transition_method`, `basis_order`, `quad_order`,
  `spectral_basis_order`, `spectral_quad_order`, `negative_mass_tol`,
  `gh_order`, `theta_cap`, `clip_negative`, `kappa_bounds`, `xi_bounds`,
  `stationary_shape_max`, `tau_eps`, `analytical_grad`, `smart_init`,
  `verbose`
- SCAR-MC: `alpha0`, `gtol`, `ftol`, `maxfun`, `maxiter`, `maxls`, `eps`, `n_tr`, `M_iterations`, `stationary`, `seed`,
  `dwt`, `smart_init`, `verbose`

Vine-level pruning controls reduce the number of dynamic edge fits:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `truncation_level` | `None` | Tree levels $\ge$ `truncation_level` stay MLE. Useful for high-dimensional vines. |
| `min_edge_logL` | `None` | Edges with MLE log-likelihood below the threshold stay MLE. |
| `transform_type` | `'softplus'` | Parameter transform used for Archimedean candidate copulas. |

In C-vines, `config` is accepted through `**kwargs` and is passed to the
dynamic edge refit. The initial MLE family-selection stage uses its own fast
MLE path.

## R-Vines

`RVineCopula.fit` has an explicit `config` argument and forwards strategy
options to edge fits through the Dissmann selector.

```python
from pyscarcopula import RVineCopula
from pyscarcopula._types import LBFGSBConfig, NumericalConfig

cfg = NumericalConfig(
    gas_optimizer=LBFGSBConfig(ftol=1e-12, maxfun=3000, maxiter=3000))

vine = RVineCopula(
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

The same strategy options listed for C-vines are forwarded to every
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

As with C-vines, automatic family selection is MLE-based. `gtol`, `ftol`,
`gamma_bound`, `K`, and similar strategy controls affect the dynamic edge refit
after a family has been selected. If `method='gas'`, a too-loose `ftol` can make
some edges stop early with `success=True`; set `ftol=1e-12` and increase
`maxfun` for difficult edges.

## BLAS Threads

`pyscarcopula` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` to `1` during package import to avoid oversubscription
in transfer-matrix workloads. This only affects BLAS libraries that have not
been initialized yet. If NumPy/SciPy/OpenBLAS was imported before
`pyscarcopula`, set the variables before starting Python:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python script.py
```

On Windows PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python script.py
```

If an application imports NumPy first and still needs to change BLAS threads at
runtime, use a runtime thread limiter such as `threadpoolctl`.
