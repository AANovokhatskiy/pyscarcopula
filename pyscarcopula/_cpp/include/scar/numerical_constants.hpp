#pragma once

namespace scar {
namespace numerical {

// Smallest positive density/log argument used by pair-copula kernels.
inline constexpr double kPdfFloor = 1e-300;

// Boundary used by pair-copula h and inverse-h numerical formulas. Internal
// vine recursion must retain tail probabilities below the final Rosenblatt
// output boundary.
inline constexpr double kHFunctionEps = 1e-10;

// Boundary applied only before Gaussian/Student inverse CDF evaluation.
// Rosenblatt clipping, matrix regularization, and log floors are separate.
inline constexpr double kPseudoObservationEps = 1e-10;

// Boundary applied to the completed Rosenblatt transform before GoF normal
// quantiles. This is intentionally wider than the internal h-function guard.
inline constexpr double kRosenblattOutputEps = 1e-6;

}  // namespace numerical
}  // namespace scar
