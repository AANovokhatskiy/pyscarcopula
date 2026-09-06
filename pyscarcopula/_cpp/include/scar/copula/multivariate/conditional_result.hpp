#pragma once

#include "scar/core/result.hpp"

#include <cstdint>
#include <vector>

namespace scar {

struct ConditionalSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    std::int64_t n_free = 0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_blocks = 0;
    std::uint64_t correlation_factorizations = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
