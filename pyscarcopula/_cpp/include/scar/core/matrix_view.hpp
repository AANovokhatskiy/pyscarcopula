#pragma once

#include "scar/core/span.hpp"

#include <cstddef>

namespace scar {

/// Minimal C++17 non-owning row-major matrix view.
///
/// `n_obs` and `dim` retain the names used by the current native ABI.  The
/// type itself is model-independent and can represent any contiguous matrix.
template <typename T>
struct MatrixView {
    T* values = nullptr;
    std::size_t n_obs = 0;
    int dim = 0;

    std::size_t size() const noexcept {
        return n_obs;
    }

    bool empty() const noexcept {
        return n_obs == 0;
    }

    T* data() const noexcept {
        return values;
    }

    T* row(std::size_t index) const noexcept {
        return values + index * static_cast<std::size_t>(dim);
    }
};

using DoubleMatrixView = MatrixView<const double>;

}  // namespace scar
