#pragma once

#include "scar/core/result.hpp"

#include <vector>

namespace scar {

/// Static negative log-likelihood objective and requested gradients.
struct StaticObjectiveResult {
    double negative_log_likelihood = 0.0;
    double negative_gradient = 0.0;
    std::vector<double> negative_correlation_gradient;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_blocks = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
