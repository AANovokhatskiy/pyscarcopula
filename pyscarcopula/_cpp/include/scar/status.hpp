#pragma once

namespace scar {

/// Status codes returned by numerical kernels and exposed through pybind11.
/// A non-zero value indicates that output fields may be partial or invalid.
inline constexpr int SCAR_OK = 0;
/// A required raw pointer was null.
inline constexpr int SCAR_NULL_POINTER = 1;
/// An array size, dimension, or row count was invalid.
inline constexpr int SCAR_INVALID_SIZE = 2;
/// The requested copula family is unknown or unsupported by the kernel.
inline constexpr int SCAR_INVALID_FAMILY = 3;
/// The requested bivariate rotation is invalid.
inline constexpr int SCAR_INVALID_ROTATION = 4;
/// The requested latent-to-parameter transform is invalid.
inline constexpr int SCAR_INVALID_TRANSFORM = 5;
/// A model parameter lies outside its valid domain.
inline constexpr int SCAR_INVALID_PARAMETER = 6;
/// Floating-point evaluation failed or produced a non-finite result.
inline constexpr int SCAR_NUMERICAL_FAILURE = 7;

/// No transfer-matrix fallback was required.
inline constexpr int SCAR_FALLBACK_NONE = 0;
/// The preferred transition method failed numerically.
inline constexpr int SCAR_FALLBACK_FAILED = 1;
/// The preferred grid size exceeded its configured cap.
inline constexpr int SCAR_FALLBACK_CAPPED = 2;

}  // namespace scar
