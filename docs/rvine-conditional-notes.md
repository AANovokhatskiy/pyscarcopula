# RVine Conditional Sampling Notes

This note summarizes the RVine conditional sampling work so future sessions can
restart without relying on chat context.

## Current Session Summary

Current implementation status:

- Arbitrary RVine conditional sampling already works through the existing
  matrix-order backends: `auto`, `grid`, and `exact`.
- The graph-based paths implement the conditional-sampling and
  conditioning-structure ideas from Cheng et al. (2025), "Vine Copulas as
  Differentiable Computational Graphs", arXiv:2506.13318. The differentiable
  PyTorch/autograd part of that paper is not implemented here.
- `conditional_method="graph"` is implemented as a production-safe fast path
  with a flexible inverse graph executor. It runs when either:
  - `conditional_plan(given)["posterior_dim"] == 0`, using the vectorized
    matrix-order column operation graph; or
  - `flexible_graph_plan(given)["sampleable"] == True`, using the experimental
    flexible graph executor. This path can sample higher-tree
    pseudo-observations and recover base variables through inverse h-function
    chains when the needed conditioning pseudo-observations are already known.
    It is currently enabled only for single-given patterns; multi-given
    patterns with `posterior_dim > 0` still need `grid` or `exact` because
    future fixed variables induce posterior reweighting.
- `structure_mode="conditional"` is available for fitted RVines when the future
  conditioning pattern is known. The default
  `conditional_structure_policy="min_posterior"` favors lowering
  `posterior_dim`; it falls back to a conditional C-vine when the priority
  structure is not encodable or is worse for the current matrix-order sampler.
- For the 6D crypto example with `given={0: 0.2, 3: 0.8}`, the conditional
  C-vine fallback reduces `posterior_dim` from `3` to `0`, making
  `conditional_method="graph"` feasible. The fitted log-likelihood is lower
  than standard Dissmann, so this is an explicit speed/sampling-cost tradeoff.

Files changed by this work:

- `pyscarcopula/vine/rvine.py`: public API, structure policy,
  `conditional_plan`, and `flexible_graph_plan`.
- `pyscarcopula/vine/_structure.py`: conditional-priority tree selection and
  priority Kruskal helpers.
- `pyscarcopula/vine/_conditional_rvine.py`: conditional plan metadata,
  `graph` dispatch, flexible graph reachability, and inverse graph executor.
- `pyscarcopula/vine/_conditional_grid.py`,
  `pyscarcopula/vine/_conditional_exact.py`, and
  `pyscarcopula/vine/_conditional_scalar.py`: grid/exact/scalar conditional
  backends from the earlier optimization work.
- `scripts/benchmark_rvine_conditional_structure.py`: compare standard
  Dissmann vs one conditional structure for a single `given`.
- `scripts/benchmark_rvine_conditioning_patterns.py`: scan many conditioning
  patterns and compare posterior workload/logL tradeoffs.
- `scripts/validate_rvine_conditional_gaussian.py`: validate conditional RVine
  samples against an analytic Gaussian-copula conditional oracle.
- `scripts/validate_rvine_conditional_dynamic_gaussian.py`: validate the
  sampler with externally supplied time-varying Gaussian pair-copula
  parameters, isolating conditional sampling from fitting / SCAR / GAS state
  estimation.
- `scripts/validate_rvine_conditional_dynamic_fit.py`: end-to-end diagnostic
  for fitting GAS / SCAR-TM-OU on a known dynamic Gaussian-copula DGP and
  comparing public `predict` output with the final-state Gaussian oracle.
- `scripts/report_rvine_dynamic_fit_methods.py`: compact MLE/GAS/SCAR
  comparison wrapper around the dynamic-fit validator. It reports conditional
  sampling errors and average predictive `r_t` calibration errors in one table.
- `pyscarcopula/vine/_selection.py`: Gaussian family selection now allows
  signed negative dependence for the non-rotatable Gaussian copula.
- `tests/test_vine.py`: focused coverage for conditional structures, graph
  method, plans, flexible graph executor, rotated higher-tree samples, and
  dynamic `r` arrays.

Best commands to resume:

```powershell
pytest tests\test_vine.py tests\test_vine_validation.py -q
```

Optional validation examples:

```powershell
# Static known Gaussian copula, analytic conditional oracle.
python scripts\validate_rvine_conditional_gaussian.py --corr toeplitz --given "{0: 0.2}" --conditional-method graph --check

# Dynamic Gaussian sampler oracle: known time-varying edge parameters, no fitting.
python scripts\validate_rvine_conditional_dynamic_gaussian.py --rho-process gas --given "{0: 0.2}" --conditional-method graph --check

# End-to-end fitted GAS/SCAR diagnostics against the final-state Gaussian oracle.
python scripts\validate_rvine_conditional_dynamic_fit.py --method gas --given "{0: 0.2}" --conditional-method graph --check
python scripts\validate_rvine_conditional_dynamic_fit.py --method scar-tm-ou --rho-process ou --given "{0: 0.2}" --conditional-method graph --diagnose-r

# Compact MLE/GAS/SCAR comparison table.
python scripts\report_rvine_dynamic_fit_methods.py --rho-process ou --fit-n 350 --sample-n 400
```

Observed local verification at the end of this session:

```text
pytest tests\test_vine.py -q
102 passed

pytest tests\test_vine.py tests\test_vine_validation.py -q
103 passed, 5 skipped

$env:PYSCA_RUN_VALIDATION='1'; pytest tests\test_vine_validation.py -q
5 passed
```

Only residual warning: pytest cannot create `.pytest_cache` in this workspace
because of Windows permissions. This is unrelated to the implementation.

Latest short dynamic-method report:

```powershell
python scripts\report_rvine_dynamic_fit_methods.py --rho-process ou --fit-n 180 --sample-n 180 --methods mle,gas,scar-tm-ou
```

```text
method=mle status=PASS mean_abs=0.2240 cov_abs=0.1779 corr_abs=0.0975 r_mae=0.0679 r_last_abs=0.1150
method=gas status=PASS mean_abs=0.0347 cov_abs=0.0692 corr_abs=0.0687 r_mae=0.0861 r_last_abs=0.0921
method=scar-tm-ou status=PASS mean_abs=0.1236 cov_abs=0.1929 corr_abs=0.2000 r_mae=0.0689 r_last_abs=0.1112
```

Interpretation: on this small OU run all methods pass the current diagnostic
thresholds. GAS gives the best conditional sample errors. SCAR has `r_mae`
close to MLE, but worse covariance/correlation sample errors, so this still
points to predictive-state/sample calibration as the next SCAR diagnostic
target rather than a failure of the conditional graph sampler itself.

Recommended next step:

Broaden distributional validation for the flexible graph executor. The
higher-tree inverse chain now has smoke coverage for rotated Archimedean
families and dynamic `r` arrays, but it should still be compared against
`exact` / high-order `grid` on small vines and multiple conditioning patterns.
The Gaussian-oracle validation script passes Toeplitz and block
Gaussian-copula checks. The block stress case exposed a family-selection bug:
non-rotatable Gaussian copulas were skipped for negative Kendall tau, causing
negative Gaussian partial correlations to be replaced by independence edges.
`pyscarcopula/vine/_selection.py` now keeps signed Gaussian tau and refines
negative Gaussian candidates correctly.

The dynamic Gaussian sampler-only validator passes single-given flexible graph
checks and multi-given grid checks. It also exposed that the flexible graph
executor is not valid for multi-given patterns with `posterior_dim > 0`: it can
sample variables before future fixed variables have been accounted for. The
runtime guard now keeps those patterns on `grid` / `exact`.

The dynamic fit validator has passing GAS and SCAR-TM-OU smoke checks on small
time-varying Gaussian-copula DGPs. GAS uses a GAS-like rho recursion; SCAR uses
an OU latent rho process matching the SCAR-TM-OU modeling assumption and the
same normalized time scale (`dt=1/(fit_n-1)`) used by `tm_loglik`. There is
also a small multi-given GAS/grid diagnostic. Use `--diagnose-r` to print true
edge-level Gaussian partial correlations versus fitted/predictive `r`; this
is the main tool for deciding whether SCAR sampling loss comes from state
calibration or from the conditional sampler.

Observed dynamic-fit diagnostics:

- On weak OU dynamics, MLE can still look competitive because the true
  correlation path is narrow. That does not indicate a sampler bug.
- On stronger OU dynamics, GAS and SCAR can improve some covariance/correlation
  errors versus MLE, but strict KS/mean thresholds may still fail. In those
  runs `--diagnose-r` usually shows the decisive issue: predictive SCAR partial
  correlations are biased or too diffuse relative to the analytic oracle, so
  the model's sampling advantage is lost at the state-calibration layer.
- `structure_mode="conditional"` and Dissmann can trade off sampler workload
  against state calibration. Dissmann may expose a higher-tree flexible graph
  path for single-given conditioning, while conditional structure often gives
  `posterior_dim == 0` for target variables but fits different edge state
  paths.

## User-Facing API

`RVineCopula.predict(...)` now accepts:

- `quad_order=10`
- `conditional_method='auto'`

Supported `conditional_method` values:

- `'auto'`: uses the graph/no-posterior path or joint-grid fast path when
  applicable.
- `'graph'`: requires `posterior_dim == 0` in `vine.conditional_plan(given)`;
  uses the vectorized column operation graph without posterior grid or exact
  quadrature.
- `'grid'`: requires the joint-grid fast path; raises if the grid would be too
  large.
- `'exact'`: forces recursive quadrature / exact-posterior path.

`RVineCopula(...)` can now select a structure for a known conditional sampling
pattern:

```python
vine = RVineCopula(
    structure_mode="conditional",
    conditional_vars={0, 3},
    conditional_structure_policy="min_posterior",
)
```

`structure_mode="conditional"` uses the same tree-by-tree fitting pipeline as
Dissmann selection, but its Kruskal step first considers candidate edges whose
conditioned pair is contained in `conditional_vars`. This prepares the fitted
vine for later calls such as `predict(..., given={0: 0.2, 3: 0.8})` while still
falling back to the existing `auto` / `grid` / `exact` conditional sampler.
If the selected conditional tree set cannot be encoded by the current
`RVineMatrix` converter, fitting emits a `RuntimeWarning` and falls back to a
conditional C-vine whose diagonal order places non-conditioning variables first
and `conditional_vars` last. In the current right-to-left sampler order this
puts the conditioning variables first and can reduce `posterior_dim` to zero.
The default `conditional_structure_policy="min_posterior"` also uses this
fallback when an encodable priority structure would have a larger
`posterior_dim` than the conditional C-vine. Use
`conditional_structure_policy="priority"` to keep any encodable priority
structure.

The helper `vine.is_conditioning_optimized_for(given)` returns whether the
given variables are a subset of the fitted `conditional_vars`.

`vine.conditional_plan(given, quad_order=10)` reports the current matrix-order
conditional sampling workload:

- `order_vars`: variable order used by the current sampler.
- `posterior_vars` / `posterior_dim`: latent Rosenblatt variables that must be
  sampled from a posterior because fixed variables appear later in `order_vars`.
- `joint_grid_points`: `quad_order ** posterior_dim` for the current grid path,
  or `0` when no posterior grid is needed.
- `graph_feasible`: whether `conditional_method="graph"` can run directly.
- `graph_steps`: matrix-order column actions used by the current no-posterior
  graph executor (`given` or `sample`, with a `posterior` marker).
- `optimized_for_given`: whether `given` is covered by fitted
  `conditional_vars`.
- `structure_status`: `priority`, `cvine_fallback`, or `None`.

`vine.flexible_graph_plan(given)` is an experimental introspection helper for
the flexible graph sampler. It reports pseudo-observation nodes reachable from
`given` through deterministic h-propagation, direct tree-0 sampling, and
higher-tree pseudo-observation sampling with inverse h-chain recovery. When
`sampleable=True`, `conditional_method="graph"` can use this executor even if
higher-tree copulas are not independent. `sampleable=True` currently also
requires a single given variable. `higher_tree_frontier` now means currently
blocked edge-level samples, not all higher-tree samples.

Recommended working call for non-prefix conditional RVine sampling:

```python
vine.predict(
    5000,
    u=u_6d,
    given={0: 0.2, 3: 0.8},
    horizon="next",
    quad_order=4,
    conditional_method="grid",
)
```

## Main Modules

- `pyscarcopula/vine/_conditional_rvine.py`
  Public orchestration for arbitrary RVine conditional sampling. Holds
  validation, column operation metadata, scalar column construction, static MLE
  grid cache, and `sample_rvine_conditional_with_r`.

- `pyscarcopula/vine/_conditional_scalar.py`
  Scalar fast calls for pair-copula `pdf`, `h`, and `h_inverse`. Dispatches to
  existing numba helpers for Clayton, Frank, Gumbel, Joe, Gaussian, and
  Independent copulas.

- `pyscarcopula/vine/_conditional_grid.py`
  Joint-grid posterior approximation and vectorized grid sampling. Handles
  dynamic SCAR/GAS `r` arrays in batches and vectorized `w` posterior draws.

- `pyscarcopula/vine/_conditional_exact.py`
  Recursive exact-posterior helpers: future likelihood recursion, tabulated
  inverse-CDF sampling, quadrature, and posterior index utilities.

## Algorithmic Changes

- Old path repeatedly integrated future likelihood during inverse-CDF sampling.
  This was too slow for non-prefix `given`.

- `_sample_w_posterior` now uses tabulated inverse-CDF sampling for the exact
  path, avoiding endpoints `0` and `1` to prevent Gumbel `h_inverse`
  singularities.

- Multi-condition non-prefix sampling has a joint-grid fast path:

  ```text
  grid_points = quad_order ** posterior_dim
  ```

  The grid path is approximate; the exact path remains available for small
  reference runs.

- Static MLE `r_all` is detected even when stored as length-`n` constant arrays,
  so the joint posterior CDF is computed once and reused for all samples.

- Dynamic SCAR/GAS `r` is handled by building joint posterior CDFs in vectorized
  batches.

- The final RVine sampling loop for the grid path is vectorized over `n`.

## SCAR/GAS Predict Optimizations

- `RVineCopula.predict` caches training pseudo-observations and SCAR predictive
  state distributions for repeated calls on the same data and horizon.

- `generate_r_for_predict` accepts an internal `state_cache` / `cache_key` for
  SCAR-TM state distributions.

- `tm_forward_mixture_h` can fill SCAR state-distribution cache during the
  forward pass already needed for pseudo-observation propagation.

- `TMGrid.forward_weights` now uses a direct loop instead of callback dispatch.

- Gaussian bivariate `h` and `h_inverse` have numba implementations:
  `_gauss_h_numba` and `_gauss_h_inv_numba`.

## Benchmarks Observed Locally

Example setup: 6D crypto data, `given={0: 0.2, 3: 0.8}`,
`quad_order=4`, `conditional_method='grid'`.

- MLE `n=5000`: about `0.26s`.
- MLE `n=5000`, `quad_order=10`: about `0.47s`.
- GAS `n=5000`: about `0.30s`.
- SCAR first `n=1000`: about `2.1s`.
- SCAR cached `n=5000`: about `2.6s`.

Timings are hardware-dependent; use them as regression smoke checks, not hard
performance guarantees.

## Verification Commands

Functional tests:

```powershell
pytest tests\test_vine.py tests\test_vine_validation.py -q
```

Optional validation suite:

```powershell
$env:PYSCA_RUN_VALIDATION='1'; pytest tests\test_vine_validation.py -q
```

Validation examples:

```powershell
python scripts\validate_rvine_conditional_gaussian.py --corr block --given "{0: 0.2}" --conditional-method graph --check
python scripts\validate_rvine_conditional_dynamic_gaussian.py --rho-process gas --given "{0: 0.2}" --conditional-method graph --check
python scripts\validate_rvine_conditional_dynamic_fit.py --method gas --given "{0: 0.2}" --conditional-method graph --check
python scripts\validate_rvine_conditional_dynamic_fit.py --method scar-tm-ou --rho-process ou --given "{0: 0.2}" --conditional-method graph --diagnose-r
python scripts\report_rvine_dynamic_fit_methods.py --rho-process ou --fit-n 350 --sample-n 400
```

## Known Limitations / Next Work

- `conditional_method='grid'` is approximate. Accuracy depends on `quad_order`
  and posterior dimension.

- Grid complexity is exponential in posterior dimension. The fast path is
  intentionally bounded by `quad_order ** posterior_dim <= 50000`.

- First SCAR `predict` still spends most time in transfer-matrix preprocessing,
  especially `tm_forward_mixture_h` and `TMGrid.rmatvec`.

- Cached SCAR `n=5000` is now mostly limited by numerical pair-copula kernels
  such as Gumbel/Joe `h` and `h_inverse`, not class dispatch.

- `RVineCopula.predict` cache keys use `id/shape/dtype` for `u`. If a user
  mutates the same `u` array in place, cached pseudo-observations may become
  stale.

- `structure_mode="conditional"` is currently matrix-order aware through
  diagnostics and the `min_posterior` policy. The policy compares the encodable
  priority structure against a conditional C-vine fallback; it does not yet
  search all matrix-encodable structures or optimize a full logL/sampling-cost
  objective.

- `conditional_method="graph"` uses the existing matrix-order operation graph
  when `posterior_dim == 0`, and otherwise can use the flexible graph executor
  when `flexible_graph_plan(...).sampleable` is true. The flexible executor is
  currently limited to single-given patterns. Patterns that are not
  graph-sampleable still need `auto`, `grid`, or `exact`.

- `flexible_graph_plan` now supports higher-tree pseudo-observation samples
  when the sampled node can be inverted back to the base variable through known
  h-function inputs. Patterns that cannot build such an inverse chain remain
  blocked and should fall back to `auto`, `grid`, or `exact`.
