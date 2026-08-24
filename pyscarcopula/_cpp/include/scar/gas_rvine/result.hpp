#pragma once

#include "scar/core/result.hpp"

#include <cstdint>
#include <vector>

namespace scar {

struct GasRvineSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
