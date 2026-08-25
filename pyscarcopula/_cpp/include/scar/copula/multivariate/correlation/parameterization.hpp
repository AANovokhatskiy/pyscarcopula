#pragma once

#include "scar/core/matrix_view.hpp"
#include "scar/core/result.hpp"
#include "scar/core/span.hpp"
#include "scar/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

enum class DenseCorrelationMode : int {
    Shrinkage = 0,
    Cholesky = 1,
};

struct CorrelationPreprocessingResult {
    std::vector<double> correlation;
    std::vector<double> input_correlation;
    double min_eigenvalue_before = 0.0;
    double min_eigenvalue_after = 0.0;
    bool projection_applied = false;
    std::vector<std::int64_t> nonfinite_kendall_pairs;
    std::size_t dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct DenseCorrelationPreparationResult {
    std::vector<double> inverse_cholesky;
    double log_determinant = 0.0;
    std::size_t dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

Result<std::vector<double>> logistic_transform(DoubleView values);
Result<std::vector<double>> logit_transform(DoubleView values);

Result<bool> validate_correlation(
    DoubleView matrix,
    std::size_t dimension,
    double tolerance);

CorrelationPreprocessingResult preprocess_correlation(
    DoubleView matrix,
    std::size_t dimension,
    double eigenvalue_floor);

DenseCorrelationPreparationResult prepare_dense_correlation(
    DoubleView matrix,
    std::size_t dimension);

CorrelationPreprocessingResult estimate_kendall_correlation(
    ObservationView observations,
    double eigenvalue_floor);

Result<std::vector<double>> make_shrinkage_correlation(
    double raw_parameter,
    DoubleView base,
    std::size_t dimension);

Result<std::vector<double>> pack_cholesky_correlation(
    DoubleView correlation,
    std::size_t dimension,
    double eigenvalue_floor);

Result<std::vector<double>> unpack_cholesky_correlation(
    DoubleView parameters,
    std::size_t dimension);

Result<std::vector<double>> correlation_gradient_to_raw(
    DenseCorrelationMode mode,
    DoubleView parameters,
    DoubleView correlation,
    DoubleView correlation_gradient,
    DoubleView base,
    std::size_t dimension);

Result<std::vector<double>> shrinkage_raw_correlation_direction(
    DoubleView parameters,
    DoubleView base,
    std::size_t dimension);

}  // namespace scar
