#pragma once

#include "scar/core/result.hpp"
#include "scar/core/span.hpp"
#include "scar/observation.hpp"

#include <cstdint>
#include <vector>

namespace scar {

class FactorCorrelationOperator;

struct GaussianScoreCorrelationResult {
    std::vector<double> correlation;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct MultivariateRosenblattResult {
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

struct RadialSummaryResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_blocks = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

GaussianScoreCorrelationResult gaussian_score_correlation(
    ObservationView u);

MultivariateRosenblattResult gaussian_rosenblatt_dense(
    DoubleView correlation,
    int dimension,
    ObservationView u,
    int n_threads = 1);

MultivariateRosenblattResult gaussian_rosenblatt_equicorrelation(
    DoubleView rho,
    ObservationView u,
    int n_threads = 1);

MultivariateRosenblattResult gaussian_rosenblatt_factor(
    const FactorCorrelationOperator& correlation,
    ObservationView u,
    int n_threads = 1);

MultivariateRosenblattResult student_rosenblatt_factor(
    const FactorCorrelationOperator& correlation,
    ObservationView u,
    DoubleView df,
    int n_threads = 1);

RadialSummaryResult radial_uniform_summary(
    ObservationView residuals,
    int n_threads = 1);

}  // namespace scar
