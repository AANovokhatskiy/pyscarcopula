#pragma once

namespace scar {
namespace numerical {

// Smallest positive density/log argument used by pair-copula kernels.
inline constexpr double kPdfFloor = 1e-300;

// Boundary used by pair-copula h and inverse-h numerical formulas.
inline constexpr double kHFunctionEps = 1e-6;

// Boundary applied only before Gaussian/Student inverse CDF evaluation.
// Rosenblatt clipping, matrix regularization, and log floors are separate.
inline constexpr double kPseudoObservationEps = 1e-10;

}  // namespace numerical
}  // namespace scar
