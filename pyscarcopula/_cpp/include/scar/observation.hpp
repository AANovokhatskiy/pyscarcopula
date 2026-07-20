#pragma once

#include <cstddef>

namespace scar {

/// Non-owning, row-major view over an `(n_obs, dim)` observation matrix.
///
/// The caller owns `values` and must keep it alive for the lifetime of the
/// view. Kernels treat the pointed-to memory as immutable.
struct ObservationView {
    const double* values = nullptr;  ///< First element of row-major storage.
    std::size_t n_obs = 0;           ///< Number of observation rows.
    int dim = 0;                     ///< Number of columns per row.

    std::size_t size() const noexcept {
        return n_obs;
    }

    bool empty() const noexcept {
        return n_obs == 0;
    }

    const double* data() const noexcept {
        return values;
    }
};

}  // namespace scar
