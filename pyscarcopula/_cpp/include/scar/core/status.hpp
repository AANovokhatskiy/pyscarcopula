#pragma once

namespace scar {

/// Stable status values returned by numerical kernels.
enum class Status : int {
    Ok = 0,
    NullPointer = 1,
    InvalidSize = 2,
    InvalidFamily = 3,
    InvalidRotation = 4,
    InvalidTransform = 5,
    InvalidParameter = 6,
    NumericalFailure = 7,
};

inline constexpr int SCAR_OK = 0;
inline constexpr int SCAR_NULL_POINTER = 1;
inline constexpr int SCAR_INVALID_SIZE = 2;
inline constexpr int SCAR_INVALID_FAMILY = 3;
inline constexpr int SCAR_INVALID_ROTATION = 4;
inline constexpr int SCAR_INVALID_TRANSFORM = 5;
inline constexpr int SCAR_INVALID_PARAMETER = 6;
inline constexpr int SCAR_NUMERICAL_FAILURE = 7;

static_assert(SCAR_OK == static_cast<int>(Status::Ok));
static_assert(SCAR_NULL_POINTER == static_cast<int>(Status::NullPointer));
static_assert(SCAR_INVALID_SIZE == static_cast<int>(Status::InvalidSize));
static_assert(SCAR_INVALID_FAMILY == static_cast<int>(Status::InvalidFamily));
static_assert(
    SCAR_INVALID_ROTATION == static_cast<int>(Status::InvalidRotation));
static_assert(
    SCAR_INVALID_TRANSFORM == static_cast<int>(Status::InvalidTransform));
static_assert(
    SCAR_INVALID_PARAMETER == static_cast<int>(Status::InvalidParameter));
static_assert(
    SCAR_NUMERICAL_FAILURE == static_cast<int>(Status::NumericalFailure));

/// Stable fallback diagnostics used by the existing SCAR-OU result DTOs.
inline constexpr int SCAR_FALLBACK_NONE = 0;
inline constexpr int SCAR_FALLBACK_FAILED = 1;
inline constexpr int SCAR_FALLBACK_CAPPED = 2;

inline constexpr bool ok(Status status) noexcept {
    return status == Status::Ok;
}

inline constexpr Status status_from_int(int status) noexcept {
    return static_cast<Status>(status);
}

}  // namespace scar
