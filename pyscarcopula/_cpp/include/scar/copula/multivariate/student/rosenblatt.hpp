#pragma once

#include "scar/core/span.hpp"
#include "scar/core/result.hpp"
#include "scar/observation.hpp"

#include <cstdint>
#include <vector>

namespace scar {

struct DenseStudentRosenblattResult {
    std::vector<double> residuals;
    std::int64_t n_rows = 0;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_blocks = 0;
    std::uint64_t correlation_factorizations = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

DenseStudentRosenblattResult student_rosenblatt_dense(
    DoubleView correlation,
    int dimension,
    ObservationView u,
    DoubleView df,
    int n_threads = 1);

}  // namespace scar
