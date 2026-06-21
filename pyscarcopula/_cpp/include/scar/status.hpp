#pragma once

namespace scar {

inline constexpr int SCAR_OK = 0;
inline constexpr int SCAR_NULL_POINTER = 1;
inline constexpr int SCAR_INVALID_SIZE = 2;
inline constexpr int SCAR_INVALID_FAMILY = 3;
inline constexpr int SCAR_INVALID_ROTATION = 4;
inline constexpr int SCAR_INVALID_TRANSFORM = 5;
inline constexpr int SCAR_INVALID_PARAMETER = 6;
inline constexpr int SCAR_NUMERICAL_FAILURE = 7;

inline constexpr int SCAR_FALLBACK_NONE = 0;
inline constexpr int SCAR_FALLBACK_FAILED = 1;
inline constexpr int SCAR_FALLBACK_CAPPED = 2;

}  // namespace scar
