# SCAR-TM-JACOBI Numerics

## SCAR-TM-JACOBI

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
| `eps` | fit kwarg / `scar_optimizer.eps` | `1e-4` | Absolute step in raw optimizer coordinates for numerical-gradient fits when `finite_diff_rel_step` is unset. Inactive for native analytical gradients. |
| `finite_diff_rel_step` | fit kwarg / `scar_optimizer.finite_diff_rel_step` | `None` | Relative step in raw optimizer coordinates for numerical-gradient fits. A non-None value takes precedence over `eps`; a non-None fit kwarg overrides the config. Inactive for native analytical gradients. |
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
| `analytical_grad` | strategy kwarg | `False` | Passes a model-provided Jacobian to the optimizer. Fully analytical for `local_fixed`; semi-analytical for `local`, `spectral_matrix`, and `auto`; native finite differences for `spectral_coeff`. |
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

### Jacobi transfer methods

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
Final fit validation checks the native parameter domain and evaluates the
objective and requested gradient without optimizer penalties. A domain or
numerical failure marks the fit unsuccessful regardless of `config.fail_value`,
so vine fitting can apply its configured fallback policy. The
`final_evaluation_status` diagnostic records `0` for successful evaluation,
`6` for invalid parameters, or `7` for a native numerical failure.
`transition_method='spectral_coeff'` uses coefficient-space filtering instead
of a transition matrix. It is available for diagnostic comparisons. With
`analytical_grad=True`, the native evaluator computes the complete objective
gradient by central finite differences and reports
`gradient_kind='native_finite_difference'`.

Fit and history-dependent evaluations need `T >= 2` to define
`dt = 1 / (T - 1)`. A prepared evaluator with one observation can still
condition an existing state. Native observation inputs must be finite and in
`[0, 1]`; out-of-range data are rejected before copula evaluation. State
conditioning and fixed-draw sampling require strictly increasing tau atoms in
`[0, 1]` and finite nonnegative masses with a finite positive total. Masses
need not sum to one: both operations normalize the measure without mutating
the caller's arrays. If conditioning has no finite likelihood at any
positive-mass atom, it retains the normalized prior.

Prepared-evaluator construction, state conditioning, and fixed-draw state
sampling reject complex inputs before conversion to `float64`, including
complex NumPy scalars stored in object arrays. This applies to both the Python
facade and direct native bindings. Real lists, integer/float arrays, and real
object arrays retain their supported conversion behavior.

### Unconditional Jacobi sampling

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

## Sparse local Jacobi transitions

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

## Experimental adaptive Jacobi order

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
