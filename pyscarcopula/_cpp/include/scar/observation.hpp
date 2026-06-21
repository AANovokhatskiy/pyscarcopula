#pragma once

#include <cstddef>

namespace scar {

struct ObservationView {
    const double* values = nullptr;
    std::size_t n_obs = 0;
    int dim = 0;

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
