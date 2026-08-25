#pragma once

#include "scar/core/matrix_view.hpp"
#include "scar/core/result.hpp"
#include "scar/core/span.hpp"
#include "scar/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

struct FactorCorrelationTransformResult {
    std::vector<double> loadings;
    std::vector<double> uniqueness;
    std::size_t dimension = 0;
    std::size_t rank = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct FactorLoadingParameterizationResult {
    std::vector<std::int64_t> anchors;
    std::vector<std::int64_t> free_rows;
    std::vector<std::int64_t> free_columns;
    std::vector<std::uint8_t> diagonal_entries;
    std::vector<double> parameters;
    double max_norm = 0.0;
    std::size_t dimension = 0;
    std::size_t rank = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct FactorInitializationResult {
    std::vector<double> loadings;
    std::vector<double> leading_eigenvalues;
    std::size_t dimension = 0;
    std::size_t rank = 0;
    std::size_t subspace_size = 0;
    std::size_t score_tile = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

FactorCorrelationTransformResult factor_correlation_from_loadings(
    DoubleView loadings,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min);

FactorCorrelationTransformResult factor_correlation_from_unconstrained(
    DoubleView values,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min);

Result<std::vector<double>> factor_correlation_to_dense(
    DoubleView loadings,
    DoubleView uniqueness,
    std::size_t dimension,
    std::size_t rank);

FactorLoadingParameterizationResult factor_parameterization_from_loadings(
    DoubleView loadings,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min);

Result<std::vector<double>> factor_parameterization_loadings(
    DoubleView parameters,
    DoubleView free_rows,
    DoubleView free_columns,
    DoubleView diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    double max_norm);

Result<std::vector<double>> factor_parameterization_pullback(
    DoubleView parameters,
    DoubleView loading_gradient,
    DoubleView free_rows,
    DoubleView free_columns,
    DoubleView diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    double max_norm);

FactorInitializationResult estimate_factor_loadings(
    ObservationView observations,
    std::size_t rank,
    double uniqueness_min,
    std::size_t configured_dimension_tile,
    DoubleMatrixView random_projection);

}  // namespace scar
