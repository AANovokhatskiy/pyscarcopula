#pragma once

namespace scar {

/// Iterative inverse-h controls for conditional pair sampling.
/// Gumbel tests a relative transformed-equation residual; Joe tests a
/// log(-log(h)) residual or a certified representable root bracket.
/// The tolerance is an upper bound; kernels may use a tighter tail criterion.
struct HInverseOptions {
    double tolerance = 1e-10;
    int max_iterations = 60;
};

}  // namespace scar
