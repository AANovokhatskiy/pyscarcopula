# Student Quantiles

## Student PPF table memory cap

Multivariate Student models normally precompute a quantile (inverse-CDF) table
of shape `(n_df_nodes, T, d)` (about 360 df nodes by default) to speed up
emission-density evaluations. The table costs `n_nodes × T × d × 8` bytes and
is bounded by `DEFAULT_MAX_TABLE_BYTES` (256 MiB) from
`pyscarcopula.copula.multivariate.student_ppf_cache`. The default nodes include
a dense geometric layer above `df = 2 + 1e-6`, where the quantile changes
rapidly, and extend through `df = 1000`.

When the estimated size exceeds the limit, the values table is not built.
Python-level table calls delegate to the native Student quantile evaluator,
including when the values table is absent. No SciPy quantile fallback is used.
Native dynamic-emission specs retain the df nodes even without the values table:
they use CDF inversion below `df=1000`, a sixth-order normal-limit expansion
from `df=1000` through the final node, and a third-order Cornish-Fisher
expansion above it. Static Student likelihoods carry no dynamic node metadata:
they use CDF inversion below `df=1000` and the sixth-order expansion above
that threshold, avoiding cancellation in large-`df` CDF inversion.

Student density normalization also avoids subtracting large log-gamma values.
At `df >= 32`, native kernels combine integer gamma recurrences in `log1p`
form with a half-step Stirling expansion through `df^-11`. The derivative uses
the same expansion. Density scores evaluate the small remainder
`log1p(x) - x/(1+x)` by a series near zero, avoiding cancellation and products
of `df^2` in the normal limit. Dense and factor kernels share this implementation;
the finite-df copula is retained rather than replaced by a hard Gaussian cutoff.

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

Prepared Student quantile tables keep their observation snapshot, nodes, and
cached values read-only. Reuse compares observation contents; modifying the
source observations creates a new cache. Row-block coordinates must be
non-negative integers with `start <= stop` inside the available rows. An empty
block at the end is valid.

With `transition_method='auto'`, a `numerical_failure` from the spectral path
may be recovered by trying matrix and then local transition methods. Forced
transition methods raise an error instead of trying other transition methods.
Invalid configurations and unsupported combinations are not treated as
recoverable numerical fallbacks.
