#include "scar/numerical_validation.hpp"
#include "scar/core/checked_arithmetic.hpp"

#include "scar/copula/multivariate/correlation/parameterization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace scar {
namespace {

constexpr double kInvalidObjectiveThreshold = 1e9;
constexpr double kBoundaryTolerance = 1e-10;

bool close(double left, double right, double rtol, double atol) noexcept {
    return std::isfinite(left)
        && std::isfinite(right)
        && std::abs(left - right) <= atol + rtol * std::abs(right);
}

bool finite_view(DoubleView values) noexcept {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

double quiet_nan() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

template <typename T>
std::vector<double> rank_observations(
    Span<const T> values, std::size_t rows, std::size_t columns) {
    std::size_t size = 0;
    if (!core::checked_size_mul(rows, columns, size)
        || size != values.size() || (size != 0 && values.data() == nullptr)) {
        throw std::invalid_argument("data size does not match its shape");
    }
    std::vector<double> output(size);
    if (size == 0) {
        return output;
    }
    std::vector<std::size_t> order(rows);
    const double denominator = static_cast<double>(rows) + 1.0;
    for (std::size_t column = 0; column < columns; ++column) {
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const T left = values[a * columns + column];
            const T right = values[b * columns + column];
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(left) != std::isnan(right)) {
                    return !std::isnan(left);
                }
            }
            if (left < right) return true;
            if (right < left) return false;
            return a < b;
        });
        for (std::size_t rank = 0; rank < rows; ++rank) {
            output[order[rank] * columns + column] =
                (static_cast<double>(rank) + 1.0) / denominator;
        }
    }
    return output;
}

}  // namespace

std::vector<double> pseudo_observations(
    DoubleView values, std::size_t rows, std::size_t columns) {
    return rank_observations(values, rows, columns);
}

std::vector<double> pseudo_observations(
    Span<const std::int64_t> values, std::size_t rows, std::size_t columns) {
    return rank_observations(values, rows, columns);
}

std::vector<double> pseudo_observations(
    Span<const std::uint64_t> values, std::size_t rows, std::size_t columns) {
    return rank_observations(values, rows, columns);
}

bool objective_is_invalid(double value) noexcept {
    return !std::isfinite(value) || value >= kInvalidObjectiveThreshold;
}

ClipResult clip_open_unit(DoubleView values, double epsilon) {
    ClipResult result;
    if (!std::isfinite(epsilon) || !(epsilon > 0.0) || !(epsilon < 0.5)) {
        result.status = Status::InvalidParameter;
        result.code = NumericalValidationCode::InvalidEpsilon;
        return result;
    }
    const double upper = 1.0 - epsilon;
    result.values.resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        result.values[index] = std::isnan(value)
            ? value
            : std::clamp(value, epsilon, upper);
    }
    return result;
}

Result<bool> open_unit_clip_required(DoubleView values, double epsilon) {
    if (!std::isfinite(epsilon) || !(epsilon > 0.0) || !(epsilon < 0.5)) {
        return {false, Status::InvalidParameter, {}};
    }
    const double upper = 1.0 - epsilon;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        if (!(value > epsilon) || !(value < upper)) {
            return success(true);
        }
    }
    return success(false);
}

NumericalValidationResult validate_pseudo_observations(
    DoubleView values) noexcept {

    NumericalValidationResult result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        if (!std::isfinite(value)) {
            result.status = Status::NumericalFailure;
            result.code = NumericalValidationCode::NonFinite;
        } else if (value < 0.0 || value > 1.0) {
            result.status = Status::InvalidParameter;
            result.code = NumericalValidationCode::OutsideUnitInterval;
        } else {
            continue;
        }
        result.failure.index = static_cast<std::int64_t>(index);
        return result;
    }
    return result;
}

NumericalValidationResult validate_fit_observations(
    DoubleView values,
    std::size_t rows,
    std::size_t columns) noexcept {

    NumericalValidationResult result;
    if (columns == 0 || rows > values.size() / columns
        || rows * columns != values.size()) {
        result.status = Status::InvalidSize;
        result.code = NumericalValidationCode::NonFinite;
        return result;
    }
    result = validate_pseudo_observations(values);
    if (!result.is_ok()) {
        const auto index = static_cast<std::size_t>(result.failure.index);
        result.row = static_cast<std::int64_t>(index / columns);
        result.column = static_cast<int>(index % columns);
        result.failure.index = -1;
        result.failure.row = result.row;
        result.failure.coordinate = result.column;
        return result;
    }
    for (std::size_t column = 0; column < columns; ++column) {
        const double first = values[column];
        bool constant = true;
        for (std::size_t row = 1; row < rows; ++row) {
            if (values[row * columns + column] != first) {
                constant = false;
                break;
            }
        }
        if (constant) {
            result.status = Status::InvalidParameter;
            result.code = NumericalValidationCode::ConstantColumn;
            result.column = static_cast<int>(column);
            result.failure.coordinate = result.column;
            return result;
        }
    }
    for (std::size_t right = 1; right < columns; ++right) {
        for (std::size_t left = 0; left < right; ++left) {
            bool duplicate = true;
            for (std::size_t row = 0; row < rows; ++row) {
                if (values[row * columns + left]
                    != values[row * columns + right]) {
                    duplicate = false;
                    break;
                }
            }
            if (duplicate) {
                result.status = Status::InvalidParameter;
                result.code = NumericalValidationCode::DuplicateColumn;
                result.column = static_cast<int>(right);
                result.failure.coordinate = result.column;
                return result;
            }
        }
    }
    return result;
}

NumericalValidationResult validate_equicorr_prepared_statistics(
    DoubleView sum_z,
    DoubleView sum_z2,
    int dimension,
    double clipping_epsilon) noexcept {

    NumericalValidationResult result;
    if (sum_z.size() != sum_z2.size() || dimension < 2) {
        result.status = Status::InvalidSize;
        result.code = NumericalValidationCode::InvalidEpsilon;
        return result;
    }
    if (!std::isfinite(clipping_epsilon)
        || !(clipping_epsilon > 0.0)
        || !(clipping_epsilon < 0.5)) {
        result.status = Status::InvalidParameter;
        result.code = NumericalValidationCode::InvalidEpsilon;
        return result;
    }
    const double dimension_root = std::sqrt(static_cast<double>(dimension));
    const double scale = 64.0 * std::numeric_limits<double>::epsilon();
    for (std::size_t row = 0; row < sum_z.size(); ++row) {
        if (!std::isfinite(sum_z[row]) || !std::isfinite(sum_z2[row])) {
            result.status = Status::NumericalFailure;
            result.code = NumericalValidationCode::NonFinite;
            result.row = static_cast<std::int64_t>(row);
            result.failure.row = result.row;
            return result;
        }
        if (sum_z2[row] < 0.0) {
            result.status = Status::InvalidParameter;
            result.code = NumericalValidationCode::NegativeValue;
            result.row = static_cast<std::int64_t>(row);
            result.failure.row = result.row;
            return result;
        }
        const double absolute_sum = std::abs(sum_z[row]);
        const double bound = dimension_root * std::sqrt(sum_z2[row]);
        const double tolerance = scale * std::max({1.0, absolute_sum, bound});
        if (absolute_sum > bound + tolerance) {
            result.status = Status::InvalidParameter;
            result.code = NumericalValidationCode::CauchyBound;
            result.row = static_cast<std::int64_t>(row);
            result.failure.row = result.row;
            return result;
        }
    }
    return result;
}

bool valid_ou_final_parameters(DoubleView values) noexcept {
    return values.size() >= 3
        && finite_view(values)
        && values[0] > 0.0
        && values[2] > 0.0;
}

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
    double growth_limit) {

    FinalFitValidation out;
    if (final_parameters.size() < 3 || !finite_view(final_parameters)) {
        out.reasons.emplace_back("final parameters are not finite");
    } else if (final_parameters[0] <= 0.0 || final_parameters[2] <= 0.0) {
        out.reasons.emplace_back("final kappa and nu must be positive");
    }
    if (!selected_evaluation_succeeded) {
        out.reasons.push_back(
            selected_engine + " final evaluation failed: " + selected_error);
    }
    if (objective_is_invalid(optimizer_value)) {
        out.reasons.emplace_back("optimizer returned an invalid objective value");
    }
    if (objective_is_invalid(selected_value)) {
        out.reasons.emplace_back("invalid objective value from selected backend");
    }
    if (selected_gradient.size() != final_parameters.size()
        || !finite_view(selected_gradient)) {
        out.reasons.emplace_back("final gradient is not finite");
    }

    const double observations = static_cast<double>(std::max<std::int64_t>(n_obs, 1));
    out.optimizer_abs_tolerance = std::max(1e-7, observations * 1e-8);
    out.optimizer_rel_tolerance = 1e-8;
    if (!objective_is_invalid(selected_value)
        && !objective_is_invalid(optimizer_value)
        && !close(
            selected_value,
            optimizer_value,
            out.optimizer_rel_tolerance,
            out.optimizer_abs_tolerance)) {
        out.reasons.emplace_back(
            "optimizer and selected-backend objectives disagree");
    }

    if (strict_gradient_policy) {
        out.has_projected_gradient = true;
        out.projected_gradient_norm = std::numeric_limits<double>::infinity();
        out.projected_gradient_tolerance = has_explicit_gradient_tolerance
            ? explicit_gradient_tolerance
            : std::max(1e-2, 10.0 * optimizer_gtol);
        if (selected_gradient.size() == final_parameters.size()
            && lower.size() == final_parameters.size()
            && upper.size() == final_parameters.size()
            && finite_view(selected_gradient)) {
            double maximum = 0.0;
            for (std::size_t index = 0;
                 index < final_parameters.size();
                 ++index) {
                double gradient = selected_gradient[index];
                const bool at_lower = std::isfinite(lower[index])
                    && std::abs(final_parameters[index] - lower[index])
                        <= kBoundaryTolerance;
                const bool at_upper = std::isfinite(upper[index])
                    && std::abs(final_parameters[index] - upper[index])
                        <= kBoundaryTolerance;
                if ((at_lower && gradient > 0.0)
                    || (at_upper && gradient < 0.0)) {
                    gradient = 0.0;
                }
                maximum = std::max(maximum, std::abs(gradient));
            }
            out.projected_gradient_norm = maximum;
            if (maximum > out.projected_gradient_tolerance) {
                out.reasons.emplace_back(
                    "projected gradient exceeds validation tolerance");
            }
        }
    }

    out.boundary_flags.assign(final_parameters.size(), 0);
    if (lower.size() == final_parameters.size()
        && upper.size() == final_parameters.size()) {
        for (std::size_t index = 0; index < final_parameters.size(); ++index) {
            const bool at_lower = std::isfinite(lower[index])
                && std::abs(final_parameters[index] - lower[index])
                    <= kBoundaryTolerance;
            const bool at_upper = std::isfinite(upper[index])
                && std::abs(final_parameters[index] - upper[index])
                    <= kBoundaryTolerance;
            out.boundary_flags[index] = static_cast<unsigned char>(
                at_lower || at_upper);
        }
    }

    if (final_parameters.size() >= 3
        && std::isfinite(final_parameters[0])
        && std::isfinite(final_parameters[1])
        && std::isfinite(final_parameters[2])) {
        out.has_ou_diagnostics = true;
        const double kappa = final_parameters[0];
        const double nu = final_parameters[2];
        const double dt = 1.0 / static_cast<double>(std::max<std::int64_t>(n_obs - 1, 1));
        out.kappa_dt = kappa * dt;
        out.rho = out.kappa_dt < 746.0 ? std::exp(-out.kappa_dt) : 0.0;
        out.stationary_std = kappa > 0.0 && nu > 0.0
            ? nu / std::sqrt(2.0 * kappa)
            : quiet_nan();
        const double conditional_variance = out.kappa_dt >= 0.0
            ? -std::expm1(-2.0 * out.kappa_dt)
            : quiet_nan();
        out.conditional_std = std::isfinite(out.stationary_std)
                && std::isfinite(conditional_variance)
                && conditional_variance >= 0.0
            ? out.stationary_std * std::sqrt(conditional_variance)
            : quiet_nan();
        if (!std::isfinite(out.stationary_std)
            || !(out.stationary_std > 0.0)
            || !std::isfinite(out.conditional_std)
            || !(out.conditional_std > 0.0)) {
            out.reasons.emplace_back("final OU variance is degenerate");
        }
        if (out.rho <= rho_tolerance
            || 1.0 - out.rho <= rho_tolerance) {
            out.reasons.emplace_back(
                "final one-step autocorrelation is degenerate");
        }

        if (initial_parameters.size() >= 3
            && std::isfinite(initial_parameters[0])
            && std::isfinite(initial_parameters[1])
            && std::isfinite(initial_parameters[2])) {
            out.has_parameter_growth = true;
            out.parameter_growth.resize(3);
            bool excessive = false;
            for (std::size_t index = 0; index < 3; ++index) {
                const double baseline = std::max(
                    std::abs(initial_parameters[index]), 1.0);
                out.parameter_growth[index] =
                    std::abs(final_parameters[index]) / baseline;
                excessive = excessive
                    || out.parameter_growth[index] > growth_limit;
            }
            if (excessive) {
                out.reasons.push_back(
                    "final OU parameters exceed initialization scale by more than "
                    + std::to_string(growth_limit));
            }
        }
    }
    return out;
}

BackendAgreementValidation validate_backend_agreement(
    bool enabled,
    bool evaluation_succeeded,
    const std::string& engine,
    const std::string& error,
    double validation_value,
    double selected_value,
    std::int64_t n_obs,
    double abs_per_observation,
    double relative_tolerance) {

    BackendAgreementValidation out;
    out.value = quiet_nan();
    out.difference = quiet_nan();
    out.tolerance = quiet_nan();
    if (!enabled) {
        return out;
    }
    if (!evaluation_succeeded) {
        out.reasons.push_back(engine + " validation failed: " + error);
        return out;
    }
    out.value = validation_value;
    if (objective_is_invalid(validation_value)) {
        out.reasons.push_back(
            engine + " validation returned an invalid objective");
        return out;
    }
    out.difference = std::abs(validation_value - selected_value);
    out.tolerance = std::max({
        1e-5,
        static_cast<double>(std::max<std::int64_t>(n_obs, 1))
            * abs_per_observation,
        relative_tolerance
            * std::max({
                std::abs(validation_value),
                std::abs(selected_value),
                1.0,
            }),
    });
    if (out.difference > out.tolerance) {
        out.reasons.emplace_back("selected and validation backends disagree");
    }
    return out;
}

std::vector<std::string> validate_correlation_fit_state(
    DoubleView raw_parameters,
    std::size_t expected_parameter_count,
    DoubleView correlation,
    std::size_t dimension,
    DoubleView inverse_factor,
    double log_determinant,
    double tolerance) {

    std::vector<std::string> reasons;
    if (raw_parameters.size() != expected_parameter_count
        || !finite_view(raw_parameters)) {
        reasons.emplace_back("final correlation parameters are not finite");
        return reasons;
    }
    if (dimension < 2 || correlation.size() != dimension * dimension
        || !finite_view(correlation)) {
        reasons.emplace_back("final correlation matrix is invalid");
        return reasons;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        if (std::abs(correlation[row * dimension + row] - 1.0)
            > tolerance) {
            reasons.emplace_back("final correlation diagonal is not one");
            break;
        }
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        bool failed = false;
        for (std::size_t column = 0; column < row; ++column) {
            if (std::abs(
                    correlation[row * dimension + column]
                    - correlation[column * dimension + row]) > tolerance) {
                reasons.emplace_back(
                    "final correlation matrix is not symmetric");
                failed = true;
                break;
            }
        }
        if (failed) {
            break;
        }
    }
    const Result<bool> valid = validate_correlation(
        correlation, dimension, tolerance);
    if (!valid.is_ok() && reasons.empty()) {
        reasons.emplace_back(
            "final correlation matrix is not positive definite");
    }
    if (inverse_factor.size() != dimension * dimension
        || !finite_view(inverse_factor)
        || !std::isfinite(log_determinant)) {
        reasons.emplace_back(
            "final correlation factorization is not finite");
    }
    return reasons;
}

}  // namespace scar
