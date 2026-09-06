#pragma once

#include <cstddef>

namespace scar {

/// Minimal C++17 non-owning view over contiguous storage.
template <typename T>
struct Span {
    T* values = nullptr;
    std::size_t count = 0;

    std::size_t size() const noexcept {
        return count;
    }

    bool empty() const noexcept {
        return count == 0;
    }

    T* data() const noexcept {
        return values;
    }

    T& operator[](std::size_t index) const noexcept {
        return values[index];
    }
};

using DoubleView = Span<const double>;

}  // namespace scar
