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

#ifdef SCAR_PARALLEL_TESTING
namespace rosenblatt_testing {
enum class Operation { Preparation, Rows, Update };
using FaultHook = void (*)(
    std::size_t coordinate, Operation operation, std::size_t block,
    std::int64_t row);
using OutputObserver = void (*)(const MultivariateRosenblattResult&) noexcept;
using RecordedObserver = void (*)(
    std::size_t coordinate, Operation operation, std::size_t block,
    std::size_t registration) noexcept;
// Captured from caller TLS before execution; absent from extension builds.
void set_hooks(
    FaultHook fault, OutputObserver observer,
    RecordedObserver recorded = nullptr) noexcept;
}  // namespace rosenblatt_testing
#endif

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
