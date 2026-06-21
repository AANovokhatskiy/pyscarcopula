#pragma once

#include "scar/numerical_constants.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace scar_internal {

inline constexpr double kOffset = 0.0001;
inline constexpr double kPdfEps = scar::numerical::kPdfFloor;
inline constexpr double kHEps = scar::numerical::kHFunctionEps;
inline constexpr double kPseudoObsEps =
    scar::numerical::kPseudoObservationEps;
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr std::size_t kMaxGridSize = 100000;
inline constexpr std::size_t kMaxDenseGridSize = 10000;
inline constexpr std::size_t kMaxSpectralOrder = 1024;
// The process-wide Hermite cache is bounded independently by entry count and
// by the vector storage retained for nodes, weights, and basis matrices.
inline constexpr std::size_t kHermiteRuleCacheMaxEntries = 16;
inline constexpr std::size_t kHermiteRuleCacheMaxBytes =
    8 * 1024 * 1024;

inline double clip_pseudo_observation(double value) noexcept {
    return std::clamp(value, kPseudoObsEps, 1.0 - kPseudoObsEps);
}

inline bool checked_size_mul(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t& result) noexcept {

    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool checked_size_add(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t& result) noexcept {

    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

inline bool checked_nonnegative_size(
    std::int64_t value,
    std::size_t& result) noexcept {

    if (value < 0
        || static_cast<std::uint64_t>(value)
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    result = static_cast<std::size_t>(value);
    return true;
}

inline bool checked_positive_int_size(
    int value,
    std::size_t limit,
    std::size_t& result) noexcept {

    if (value <= 0) {
        return false;
    }
    result = static_cast<std::size_t>(value);
    return result <= limit;
}

inline bool checked_row_offset(
    std::size_t row,
    std::size_t width,
    std::size_t total,
    std::size_t& result) noexcept {

    return checked_size_mul(row, width, result) && result <= total;
}

inline bool valid_student_dimension(int dim, std::size_t& square) noexcept {
    if (dim <= 0) {
        return false;
    }
    const std::size_t dim_size = static_cast<std::size_t>(dim);
    return checked_size_mul(dim_size, dim_size, square);
}

inline bool valid_student_correlation_count(
    int dim,
    std::size_t& count) noexcept {

    if (dim < 2) {
        return false;
    }
    const std::size_t dim_size = static_cast<std::size_t>(dim);
    std::size_t product = 0;
    if (!checked_size_mul(dim_size, dim_size - 1, product)) {
        return false;
    }
    count = product / 2;
    return true;
}

inline bool valid_spectral_dimensions(
    int quad_order,
    int basis_order,
    std::size_t& basis_elements) noexcept {

    std::size_t quad_size = 0;
    std::size_t basis_size = 0;
    return checked_positive_int_size(
               quad_order, kMaxSpectralOrder, quad_size)
        && checked_positive_int_size(
               basis_order, kMaxSpectralOrder, basis_size)
        && quad_order >= basis_order
        && checked_size_mul(quad_size, basis_size, basis_elements);
}

}  // namespace scar_internal
