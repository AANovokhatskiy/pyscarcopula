#pragma once

namespace scar {

/// Iterative inverse-h controls for conditional pair sampling.
/// Gumbel tests a log-equation residual; Joe tests the h-value residual.
struct HInverseOptions {
    double tolerance = 1e-10;
    int max_iterations = 60;
};

}  // namespace scar
