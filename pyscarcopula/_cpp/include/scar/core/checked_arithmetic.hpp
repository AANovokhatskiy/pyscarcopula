#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace scar::core {

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

inline bool checked_uint64_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t& result) noexcept {

    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

/// Compute the number of elements in a two-dimensional shape.
inline bool checked_shape_size(
    std::size_t rows,
    std::size_t columns,
    std::size_t& result) noexcept {

    return checked_size_mul(rows, columns, result);
}

/// Convert an element count to bytes without overflowing the output type.
template <typename Element>
inline bool checked_byte_count(
    std::size_t count,
    std::uint64_t& result) noexcept {

    static_assert(!std::is_void<Element>::value, "Element must have a size");
    static_assert(
        sizeof(std::size_t) <= sizeof(std::uint64_t),
        "byte accounting requires size_t to fit in uint64_t");
    constexpr std::uint64_t item_size = sizeof(Element);
    const std::uint64_t count_u64 = static_cast<std::uint64_t>(count);
    if (count_u64
        > std::numeric_limits<std::uint64_t>::max() / item_size) {
        return false;
    }
    result = count_u64 * item_size;
    return true;
}

}  // namespace scar::core

namespace scar_internal {

using scar::core::checked_byte_count;
using scar::core::checked_shape_size;
using scar::core::checked_size_add;
using scar::core::checked_size_mul;
using scar::core::checked_uint64_add;

}  // namespace scar_internal
