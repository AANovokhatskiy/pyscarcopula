#pragma once

#include "scar/core/result.hpp"
#include "scar/core/span.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scar {

enum class NumericalValidationCode : int {
    Ok = 0,
    InvalidEpsilon,
    NonFinite,
    OutsideUnitInterval,
    ConstantColumn,
    DuplicateColumn,
    NegativeValue,
    CauchyBound,
};

struct NumericalValidationResult {
    Status status = Status::Ok;
    FailureContext failure{};
    NumericalValidationCode code = NumericalValidationCode::Ok;
    std::int64_t row = -1;
    int column = -1;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct ClipResult {
    std::vector<double> values;
    Status status = Status::Ok;
    FailureContext failure{};
    NumericalValidationCode code = NumericalValidationCode::Ok;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

ClipResult clip_open_unit(DoubleView values, double epsilon);
/// Column-wise ordinal ranks / (rows + 1), with ties in row order and NaNs last.
std::vector<double> pseudo_observations(
    DoubleView values, std::size_t rows, std::size_t columns);
std::vector<double> pseudo_observations(
    Span<const std::int64_t> values, std::size_t rows, std::size_t columns);
std::vector<double> pseudo_observations(
    Span<const std::uint64_t> values, std::size_t rows, std::size_t columns);
Result<bool> open_unit_clip_required(DoubleView values, double epsilon);
bool objective_is_invalid(double value) noexcept;

/// Range validation without fit-specific identifiability restrictions.
NumericalValidationResult validate_pseudo_observations(
    DoubleView values) noexcept;

NumericalValidationResult validate_fit_observations(
    DoubleView values,
    std::size_t rows,
    std::size_t columns) noexcept;

NumericalValidationResult validate_equicorr_prepared_statistics(
    DoubleView sum_z,
    DoubleView sum_z2,
    int dimension,
    double clipping_epsilon) noexcept;

struct FinalFitValidation {
    std::vector<std::string> reasons;
    double optimizer_abs_tolerance = 0.0;
    double optimizer_rel_tolerance = 0.0;
    bool has_projected_gradient = false;
    double projected_gradient_norm = 0.0;
    double projected_gradient_tolerance = 0.0;
    std::vector<unsigned char> boundary_flags;
    bool has_ou_diagnostics = false;
    double kappa_dt = 0.0;
    double rho = 0.0;
    double stationary_std = 0.0;
    double conditional_std = 0.0;
    bool has_parameter_growth = false;
    std::vector<double> parameter_growth;
};

bool valid_ou_final_parameters(DoubleView values) noexcept;

FinalFitValidation validate_ou_final_fit(
    DoubleView final_parameters,
    DoubleView initial_parameters,
    DoubleView lower,
    DoubleView upper,
    double optimizer_value,
    double selected_value,
    DoubleView selected_gradient,
    bool selected_evaluation_succeeded,
    const std::string& selected_engine,
    const std::string& selected_error,
    std::int64_t n_obs,
    bool strict_gradient_policy,
    bool has_explicit_gradient_tolerance,
    double explicit_gradient_tolerance,
    double optimizer_gtol,
    double rho_tolerance,
    double growth_limit);

struct BackendAgreementValidation {
    std::vector<std::string> reasons;
    double value = 0.0;
    double difference = 0.0;
    double tolerance = 0.0;
};

BackendAgreementValidation validate_backend_agreement(
    bool enabled,
    bool evaluation_succeeded,
    const std::string& engine,
    const std::string& error,
    double validation_value,
    double selected_value,
    std::int64_t n_obs,
    double abs_per_observation,
    double relative_tolerance);

std::vector<std::string> validate_correlation_fit_state(
    DoubleView raw_parameters,
    std::size_t expected_parameter_count,
    DoubleView correlation,
    std::size_t dimension,
    DoubleView inverse_factor,
    double log_determinant,
    double tolerance);

}  // namespace scar
