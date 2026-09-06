# Multivariate Numerical Kernels

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
joint optimizer; joint static Student uses the native loading score and a
rotation-anchored loading parameterization with `d >= 2*k + 1`. None of the compact likelihood,
sampling, bootstrap, or persistence paths needs a dense `d*d` correlation.

Dense `GaussianCopula.sample` and `predict` check `memory_budget_bytes` against
the returned `n*d*8` bytes before consuming random draws. This output-only
contract also applies to dense conditional sampling; it does not bound all
temporary workspace. Batch iterators apply the check to each output block.
Factor Gaussian sampling instead checks its existing workspace-plus-output
estimate.

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
