#pragma once

#include "scar/core/status.hpp"

#include <cstdint>
#include <utility>

namespace scar {

/// Model-independent location and fallback context for a kernel failure.
struct FailureContext {
    std::int64_t index = -1;
    std::int64_t row = -1;
    int coordinate = -1;
    int edge = -1;
    int operation = -1;
    int fallback_from = -1;
};

/// Common value/status envelope for new typed native contracts.
template <typename T>
struct Result {
    T value{};
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

template <typename T>
Result<typename std::decay<T>::type> success(T&& value) {
    return {std::forward<T>(value), Status::Ok, {}};
}

}  // namespace scar
