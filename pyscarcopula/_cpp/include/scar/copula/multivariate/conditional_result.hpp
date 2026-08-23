#pragma once

#include <cstdint>
#include <vector>

namespace scar {

struct ConditionalSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    std::int64_t n_free = 0;
    int status = 0;
    std::int64_t failure_index = -1;
    int n_threads_requested = 1;
    int parallel_blocks = 0;
    std::uint64_t correlation_factorizations = 0;
};

}  // namespace scar
