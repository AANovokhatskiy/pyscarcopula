"""Named numerical constants shared by Python numerical paths."""

# Smallest positive density/log argument used by native pair-copula kernels.
PDF_FLOOR = 1e-300

# Boundary used before applying Gaussian or Student inverse CDFs to
# pseudo-observations. This is intentionally distinct from Rosenblatt output
# clipping, correlation regularization, Cholesky jitter, and log floors.
PSEUDO_OBS_EPS = 1e-10

# Boundary for bivariate h/inverse-h numerical outputs. It intentionally
# matches the native pair-copula guard and is not a pseudo-observation policy.
H_FUNCTION_EPS = 1e-6

# Boundary for the final bivariate/vine Rosenblatt transform. This is larger
# than ``PSEUDO_OBS_EPS`` because it protects the subsequent normal quantile
# used by goodness-of-fit statistics, rather than an internal copula kernel.
ROSENBLATT_OUTPUT_EPS = 1e-6

# Boundary for newly sampled free conditional coordinates. Fixed coordinates
# retain the exact valid values supplied through ``given``.
CONDITIONAL_SAMPLE_EPS = 1e-12
