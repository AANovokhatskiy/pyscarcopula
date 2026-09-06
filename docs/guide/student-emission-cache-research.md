# Opt-in Student full-emission cache (research)

The native prepared OU evaluator exposes `configure_student_emission_cache`,
`student_emission_cache_info`, and `clear_student_emission_cache`. This is an
experimental approximation for dense Student models with fixed observations
and correlation. No strategy or default objective enables it. Spectral, matrix,
and local OU transition implementations and the optimizer remain unchanged.

For each observation row, the table reconstructs the complete log-density
`g_t(xi) = log c(u_t; offset + exp(xi), R)`. Cubic Hermite interpolation uses
direct Student kernel values and first derivatives at both interval endpoints.
The returned df derivative is the derivative of this same polynomial divided
by `df - offset`. Existing native state transformations supply the remaining
chain rule. Samples bypass the PPF interpolation table and its dynamic
large-df approximation; the direct Student kernel retains its existing
normal-limit implementation at large df. The opt-in sampling/fallback path
uses the existing refined quantile solver with prepared per-df normalization
constants. Its implicit derivative is evaluated at that same refined quantile;
the density formula is shared with the original kernel. Default native
point-density and PPF interpolation paths are unchanged.

This distinction is necessary in addition to speed: the original unprepared
Student quantile solver can stall near `p=.5` and return its remaining bracket
midpoint. A reproduced case at `p=1261/2516`, `df=523.287105996051` had quantile
error -0.00149512908675446. Adaptive reconstruction correctly refused the
resulting nonsmooth density. The refined sampler fixes this opt-in path without
loosening tolerances; it does not claim to fix the existing default scalar
quantile implementation. Independent SciPy tests cover this point, neighboring
df values, interpolation and direct fallback, plus likelihood-gradient finite
differences. Consequently the original uncached native kernel alone cannot be
used as a universal accuracy oracle.

Adaptive bisection checks the value and the xi derivative at the quarter,
midpoint, and three-quarter locations for every row. The maximum accepted
residuals must satisfy separate absolute tolerances. These checks are empirical:
they do not bound the error between sample locations and do not certify the
integrated likelihood or its gradient. Tighter settings must be compared with
direct Student emissions and independently refined OU integration.
When an interval is split, its already evaluated quarter samples are reused as
child midpoint samples when their floating-point coordinates match exactly.
This preserves the table and accepted residuals bit for bit; `reused_samples`
counts avoided coordinate evaluations. Algebraically equal coordinates that
differ by rounding are evaluated separately.

The builder also deduplicates exactly equal input probabilities once, using a
sorted value/index array and an inverse index map. At each coordinate it
computes the refined quantile and its derivative once per distinct probability,
then gathers each observation row into the shared density kernel. Ordinal
pseudo-observation columns often share the same rank grid, making this reuse
especially useful. There is no rounding, approximate equality, or symmetry
folding. `unique_probabilities` and `observation_entries` report the reduction.
Construction memory checks include the index map, sort scratch, unique
probabilities, quantile/derivative arrays, and dense row workspaces before
allocation. These construction-only arrays are released after preparation.

The default coordinate interval is [-24, 6.9], with 32 initial intervals,
value tolerance 1e-7, score tolerance 1e-6, 1025 maximum knots, depth 12,
and a 128 MiB workspace budget. Finite configuration bounds, checked size
arithmetic, depth and knot limits bound memory and termination. The budget
covers new table storage and a conservative interpolation-build scratch bound;
existing prepared observations and the PPF table, if present, are additional.
Reconfiguring an active evaluator temporarily retains its old table as well,
so `max_bytes` is a candidate-build budget, not a process-wide peak-memory cap.
`table_bytes` reports occupied arrays and `reserved_bytes` the conservative
new-table/build allocation bound. Construction exceeding a refinement limit
throws explicitly and preserves an already configured objective.

Outside the interval, evaluation uses the direct Student density and score.
When floating point state transformation yields exactly `df == offset`, the
cache reuses directly calculated endpoint values and df derivatives. This is
an exact repeated-value cache, not a constant tail approximation. The supported
offset must exceed 2. Diagnostics distinguish interpolated evaluations, exact
endpoint reuse, and direct fallbacks.
The optional minimum coordinate can reach -36. At that boundary `df - offset`
has few significant bits; the direct density is already constant to nearly
machine precision, but coordinate-relative derivative accuracy is not promised.
Use the default -24 unless tail profiling justifies changing it.

The prepared evaluator owns observations, correlation and table storage. Calls
and mutations share its existing mutex; readers within an emission pass share
immutable tables. Atomic diagnostic counters do not alter likelihood results.
Every successful correlation update and every emission refresh clears the table.
Invalid correlation updates and failed cache builds leave the existing state
unchanged. Full and directional correlation-gradient calls reject an active
table, since their current exact correlation scores would not differentiate
the interpolated full-emission objective. Clear/rebuild is explicit.

Building a direct table can cost substantially more than preparing the existing
PPF interpolant. Report both setup-inclusive fit time and warmed evaluation
time, and calculate break-even evaluation counts. A scoped research factory
can skip the redundant PPF table before constructing the native evaluator;
that factory must keep the existing validation and owned-observation boundary.
Current cache block evaluation is serial even when the original backend offers
multiple emission threads, so single-thread benchmarks do not establish a
multicore speedup.
