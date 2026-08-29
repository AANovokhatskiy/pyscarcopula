#include "scar/jacobi.hpp"

#include "scar/copula/transforms.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/math/beta.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace scar {
namespace {

template <typename ResultType>
ResultType failure(Status status) {
    ResultType result;
    result.status = status;
    return result;
}

bool finite_positive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

double logit(double value) noexcept {
    return std::log(value / (1.0 - value));
}

bool add_product(
    std::size_t factor,
    std::size_t lhs,
    std::size_t rhs,
    std::size_t& total) noexcept {

    std::size_t product = 0;
    std::size_t term = 0;
    return core::checked_size_mul(lhs, rhs, product)
        && core::checked_size_mul(factor, product, term)
        && core::checked_size_add(total, term, total);
}

bool add_multiple(
    std::size_t factor,
    std::size_t value,
    std::size_t& total) noexcept {

    std::size_t term = 0;
    return core::checked_size_mul(factor, value, term)
        && core::checked_size_add(total, term, total);
}

bool add_quadrature_peak(
    std::size_t jacobi_order,
    std::size_t hermite_order,
    std::size_t dense_matrix_allowance,
    std::size_t& total) noexcept {

    std::size_t jacobi_eigenvectors = 0;
    std::size_t hermite_eigenvectors = 0;
    std::size_t hermite_linear = 0;
    std::size_t hermite_workspace = 0;
    if (!core::checked_size_mul(
            jacobi_order, jacobi_order, jacobi_eigenvectors)
        || !core::checked_size_mul(
            hermite_order, hermite_order, hermite_eigenvectors)
        || !core::checked_size_mul(5, hermite_order, hermite_linear)
        || !core::checked_size_add(
            hermite_eigenvectors,
            hermite_linear,
            hermite_workspace)) {
        return false;
    }
    const std::size_t quadrature_peak =
        std::max(jacobi_eigenvectors, hermite_workspace);
    if (quadrature_peak <= dense_matrix_allowance) {
        return true;
    }
    return core::checked_size_add(
        total, quadrature_peak - dense_matrix_allowance, total);
}

JacobiMemoryResult finish_memory_estimate(
    std::size_t elements,
    std::uint64_t budget) noexcept {

    JacobiMemoryResult result;
    result.value.elements = elements;
    result.value.budget_bytes = budget;
    if (!core::checked_byte_count<double>(elements, result.value.bytes)) {
        result.status = Status::InvalidSize;
        return result;
    }
    result.value.within_budget = result.value.bytes <= budget;
    if (!result.value.within_budget) {
        result.status = Status::InvalidSize;
    }
    return result;
}

}  // namespace

Status validate_jacobi_params(
    const JacobiParams& params,
    double stationary_shape_max) noexcept {

    if (!finite_positive(params.kappa)
        || !std::isfinite(params.m)
        || !(params.m > 0.0 && params.m < 1.0)
        || !finite_positive(params.xi)) {
        return Status::InvalidParameter;
    }
    if (!(stationary_shape_max > 0.0)
        || std::isnan(stationary_shape_max)) {
        return Status::InvalidParameter;
    }
    const double xi_squared = params.xi * params.xi;
    const double alpha = 2.0 * params.kappa * params.m / xi_squared;
    const double beta =
        2.0 * params.kappa * (1.0 - params.m) / xi_squared;
    if (!finite_positive(alpha) || !finite_positive(beta)
        || alpha > stationary_shape_max
        || beta > stationary_shape_max) {
        return Status::InvalidParameter;
    }
    return Status::Ok;
}

Status validate_jacobi_bounds(
    const JacobiParameterBounds& bounds) noexcept {

    if (!finite_positive(bounds.kappa_lower)
        || !finite_positive(bounds.kappa_upper)
        || !(bounds.kappa_lower < bounds.kappa_upper)
        || !finite_positive(bounds.xi_lower)
        || !finite_positive(bounds.xi_upper)
        || !(bounds.xi_lower < bounds.xi_upper)
        || !std::isfinite(bounds.tau_eps)
        || !(bounds.tau_eps > 0.0 && bounds.tau_eps < 0.5)) {
        return Status::InvalidParameter;
    }
    return Status::Ok;
}

Status validate_jacobi_config(
    const JacobiNumericalConfig& config) noexcept {

    if (config.quad_order <= 0
        || config.quad_order > kMaxJacobiOrder
        || config.basis_order <= 0
        || config.basis_order > kMaxJacobiOrder
        || config.basis_order > config.quad_order
        || config.gh_order <= 0
        || config.gh_order > kMaxJacobiOrder
        || config.n_obs < 0) {
        return Status::InvalidSize;
    }
    if (!std::isfinite(config.tau_eps)
        || !(config.tau_eps > 0.0 && config.tau_eps < 0.5)
        || (!std::isnan(config.theta_cap)
            && !finite_positive(config.theta_cap))
        || !(config.stationary_shape_max > 0.0)
        || std::isnan(config.stationary_shape_max)
        || !std::isfinite(config.lamperti_eps)
        || !(config.lamperti_eps > 0.0 && config.lamperti_eps < 0.5)
        || (config.boundary != JacobiBoundaryPolicy::Reflect
            && config.boundary != JacobiBoundaryPolicy::Clip)) {
        return Status::InvalidParameter;
    }
    return Status::Ok;
}

JacobiParamsResult jacobi_raw_to_physical(
    const std::array<double, 3>& raw) noexcept {

    if (!std::all_of(
            raw.begin(), raw.end(),
            [](double value) { return std::isfinite(value); })) {
        return failure<JacobiParamsResult>(Status::InvalidParameter);
    }
    const double raw_kappa = std::clamp(
        raw[0], -kJacobiRawClip, kJacobiRawClip);
    const double raw_m = std::clamp(
        raw[1], -kJacobiRawClip, kJacobiRawClip);
    const double raw_xi = std::clamp(
        raw[2], -kJacobiRawClip, kJacobiRawClip);
    JacobiParamsResult result;
    result.value = {
        std::exp(raw_kappa),
        copula::logistic_unit(raw_m),
        std::exp(raw_xi),
    };
    return result;
}

JacobiRawParamsResult jacobi_physical_to_raw(
    const JacobiParams& params,
    double tau_eps) noexcept {

    if (!finite_positive(params.kappa)
        || !std::isfinite(params.m)
        || !finite_positive(params.xi)
        || !std::isfinite(tau_eps)
        || !(tau_eps > 0.0 && tau_eps < 0.5)) {
        return failure<JacobiRawParamsResult>(Status::InvalidParameter);
    }
    JacobiRawParamsResult result;
    result.value = {
        std::log(std::max(params.kappa, 1e-300)),
        logit(std::clamp(params.m, tau_eps, 1.0 - tau_eps)),
        std::log(std::max(params.xi, 1e-300)),
    };
    return result;
}

JacobiRawParamsResult jacobi_gradient_to_raw(
    const JacobiParams& params,
    const std::array<double, 3>& physical_gradient) noexcept {

    if (!finite_positive(params.kappa)
        || !std::isfinite(params.m)
        || !(params.m > 0.0 && params.m < 1.0)
        || !finite_positive(params.xi)
        || !std::all_of(
            physical_gradient.begin(), physical_gradient.end(),
            [](double value) { return std::isfinite(value); })) {
        return failure<JacobiRawParamsResult>(Status::InvalidParameter);
    }
    JacobiRawParamsResult result;
    result.value = {
        physical_gradient[0] * params.kappa,
        physical_gradient[1] * params.m * (1.0 - params.m),
        physical_gradient[2] * params.xi,
    };
    if (!std::all_of(
            result.value.begin(), result.value.end(),
            [](double value) { return std::isfinite(value); })) {
        return failure<JacobiRawParamsResult>(Status::NumericalFailure);
    }
    return result;
}

JacobiRawBoundsResult jacobi_raw_bounds(
    const JacobiParameterBounds& bounds) noexcept {

    if (!ok(validate_jacobi_bounds(bounds))) {
        return failure<JacobiRawBoundsResult>(Status::InvalidParameter);
    }
    JacobiRawBoundsResult result;
    result.value.lower = {
        std::log(bounds.kappa_lower),
        logit(bounds.tau_eps),
        std::log(bounds.xi_lower),
    };
    result.value.upper = {
        std::log(bounds.kappa_upper),
        logit(1.0 - bounds.tau_eps),
        std::log(bounds.xi_upper),
    };
    return result;
}

JacobiParamsResult jacobi_initial_point(
    double tau, bool has_tau, double tau_eps) noexcept {
    JacobiParamsResult result;
    if (!std::isfinite(tau_eps) || tau_eps <= 0.0 || tau_eps >= 0.5
        || (has_tau && !std::isfinite(tau))) {
        result.status = Status::InvalidParameter;
        return result;
    }
    result.value = JacobiParams{
        1.0,
        has_tau ? std::clamp(tau, tau_eps, 1.0 - tau_eps) : 0.5,
        0.2};
    return result;
}

JacobiShapeResult jacobi_stationary_shape(
    const JacobiParams& params) noexcept {

    if (!ok(validate_jacobi_params(
            params, std::numeric_limits<double>::infinity()))) {
        return failure<JacobiShapeResult>(Status::InvalidParameter);
    }
    const double xi_squared = params.xi * params.xi;
    JacobiShapeResult result;
    result.value.alpha =
        2.0 * params.kappa * params.m / xi_squared;
    result.value.beta =
        2.0 * params.kappa * (1.0 - params.m) / xi_squared;
    result.value.dalpha = {
        2.0 * params.m / xi_squared,
        2.0 * params.kappa / xi_squared,
        -4.0 * params.kappa * params.m
            / (xi_squared * params.xi),
    };
    result.value.dbeta = {
        2.0 * (1.0 - params.m) / xi_squared,
        -2.0 * params.kappa / xi_squared,
        -4.0 * params.kappa * (1.0 - params.m)
            / (xi_squared * params.xi),
    };
    if (!finite_positive(result.value.alpha)
        || !finite_positive(result.value.beta)) {
        return failure<JacobiShapeResult>(Status::InvalidParameter);
    }
    return result;
}

JacobiVectorResult sample_jacobi_stationary(
    const JacobiParams& params,
    const std::vector<double>& uniforms) {

    const JacobiShapeResult shape = jacobi_stationary_shape(params);
    if (!shape.is_ok()) {
        return failure<JacobiVectorResult>(shape.status);
    }
    JacobiVectorResult result;
    try {
        result.value.resize(uniforms.size());
        for (std::size_t index = 0; index < uniforms.size(); ++index) {
            const double draw = uniforms[index];
            if (!std::isfinite(draw) || draw < 0.0 || draw >= 1.0) {
                result.status = Status::InvalidParameter;
                result.failure.index = static_cast<std::int64_t>(index);
                result.value.clear();
                return result;
            }
            const double tau = math::beta_quantile(
                draw, shape.value.alpha, shape.value.beta);
            if (!std::isfinite(tau) || tau < 0.0 || tau > 1.0) {
                result.status = Status::NumericalFailure;
                result.failure.index = static_cast<std::int64_t>(index);
                result.value.clear();
                return result;
            }
            result.value[index] = tau;
        }
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
    return result;
}

JacobiScalarResult jacobi_resolve_dt(std::int64_t n_obs) noexcept {
    if (n_obs < 2) {
        return failure<JacobiScalarResult>(Status::InvalidSize);
    }
    JacobiScalarResult result;
    result.value = 1.0 / static_cast<double>(n_obs - 1);
    return result;
}

JacobiMemoryResult estimate_jacobi_workspace(
    const JacobiNumericalConfig& config) noexcept {

    if (!ok(validate_jacobi_config(config))) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    const std::size_t k = static_cast<std::size_t>(config.quad_order);
    const std::size_t b = static_cast<std::size_t>(config.basis_order);
    const std::size_t g = static_cast<std::size_t>(config.gh_order);
    const std::size_t n = static_cast<std::size_t>(config.n_obs);
    std::size_t elements = 0;
    if (!add_product(2, k, b, elements)
        || !add_product(1, b, b, elements)
        || !add_multiple(12, k, elements)
        || !add_multiple(4, b, elements)) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    if (config.matrix
        && (!add_product(3, k, k, elements)
            || !add_product(5, n, k, elements))) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    if (config.matrix && config.gradient
        && (!add_product(4, k, k, elements)
            || !add_product(4, n, k, elements))) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    if (!config.matrix && !add_product(5, n, k, elements)) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    std::size_t dense_matrix_allowance = 0;
    if (config.matrix) {
        const std::size_t factor = config.gradient ? 7 : 3;
        if (!core::checked_size_mul(k, k, dense_matrix_allowance)
            || !core::checked_size_mul(
                factor, dense_matrix_allowance, dense_matrix_allowance)) {
            return failure<JacobiMemoryResult>(Status::InvalidSize);
        }
    }
    // Golub--Welsch retains a full eigenvector matrix while extracting the
    // quadrature weights.  Jacobi and Hermite solvers run sequentially, so
    // charge their maximum peak after subtracting dense KxK arrays that are
    // already included in the conservative transition/gradient allowance.
    if (!add_quadrature_peak(k, g, dense_matrix_allowance, elements)) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    return finish_memory_estimate(elements, config.memory_budget_bytes);
}

JacobiMemoryResult estimate_jacobi_sampling_workspace(
    std::int64_t n,
    const JacobiNumericalConfig& config) noexcept {

    if (n < 0 || !ok(validate_jacobi_config(config))) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    const std::size_t count = static_cast<std::size_t>(n);
    const std::size_t k = static_cast<std::size_t>(config.quad_order);
    const std::size_t b = static_cast<std::size_t>(config.basis_order);
    const std::size_t g = static_cast<std::size_t>(config.gh_order);
    std::size_t elements = 0;
    if ((count > 1 && !add_product(3, k, k, elements))
        || !add_product(2, k, b, elements)
        || !add_product(1, b, b, elements)
        || !add_multiple(12, k, elements)
        || !add_multiple(4, b, elements)
        || !core::checked_size_add(elements, count, elements)) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    std::size_t transition_allowance = 0;
    if (count > 1
        && (!core::checked_size_mul(k, k, transition_allowance)
            || !core::checked_size_mul(
                3, transition_allowance, transition_allowance))) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    if (!add_quadrature_peak(k, g, transition_allowance, elements)) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    return finish_memory_estimate(elements, config.memory_budget_bytes);
}

JacobiScalarResult jacobi_lamperti(
    double tau,
    double xi) noexcept {

    if (!std::isfinite(tau) || !finite_positive(xi)) {
        return failure<JacobiScalarResult>(Status::InvalidParameter);
    }
    JacobiScalarResult result;
    const double bounded_tau = std::clamp(tau, 0.0, 1.0);
    result.value = 2.0 * std::asin(std::sqrt(bounded_tau)) / xi;
    return result;
}

JacobiVectorResult jacobi_lamperti_values(
    const std::vector<double>& tau,
    double xi) {

    if (!finite_positive(xi)) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        JacobiVectorResult result;
        result.value.resize(tau.size());
        for (std::size_t index = 0; index < tau.size(); ++index) {
            const JacobiScalarResult scalar = jacobi_lamperti(tau[index], xi);
            if (!scalar.is_ok()) {
                result.status = scalar.status;
                result.failure.index = static_cast<std::int64_t>(index);
                result.value.clear();
                return result;
            }
            result.value[index] = scalar.value;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiScalarResult jacobi_inverse_lamperti(
    double value,
    double xi) noexcept {

    if (!std::isfinite(value) || !finite_positive(xi)) {
        return failure<JacobiScalarResult>(Status::InvalidParameter);
    }
    const double pi = std::acos(-1.0);
    const double bounded_value = std::clamp(value, 0.0, pi / xi);
    const double sine = std::sin(0.5 * xi * bounded_value);
    JacobiScalarResult result;
    result.value = sine * sine;
    return result;
}

JacobiVectorResult jacobi_inverse_lamperti_values(
    const std::vector<double>& values,
    double xi) {

    if (!finite_positive(xi)) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        JacobiVectorResult result;
        result.value.resize(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            const JacobiScalarResult scalar =
                jacobi_inverse_lamperti(values[index], xi);
            if (!scalar.is_ok()) {
                result.status = scalar.status;
                result.failure.index = static_cast<std::int64_t>(index);
                result.value.clear();
                return result;
            }
            result.value[index] = scalar.value;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiScalarResult jacobi_lamperti_drift(
    const JacobiParams& params,
    double tau,
    double interior_eps) noexcept {

    if (!ok(validate_jacobi_params(
            params, std::numeric_limits<double>::infinity()))
        || !std::isfinite(tau)
        || tau < 0.0 || tau > 1.0
        || !std::isfinite(interior_eps)
        || interior_eps < 0.0 || interior_eps >= 0.5) {
        return failure<JacobiScalarResult>(Status::InvalidParameter);
    }
    const double bounded_tau = interior_eps > 0.0
        ? std::clamp(tau, interior_eps, 1.0 - interior_eps)
        : tau;
    const double denominator = std::sqrt(std::max(
        bounded_tau * (1.0 - bounded_tau),
        kJacobiDriftDenominatorFloor));
    JacobiScalarResult result;
    result.value =
        params.kappa * (params.m - bounded_tau)
            / (params.xi * denominator)
        - params.xi * (1.0 - 2.0 * bounded_tau)
            / (4.0 * denominator);
    if (!std::isfinite(result.value)) {
        result.status = Status::NumericalFailure;
    }
    return result;
}

JacobiVectorResult jacobi_lamperti_drift_values(
    const JacobiParams& params,
    const std::vector<double>& tau,
    double interior_eps) {

    if (!ok(validate_jacobi_params(
            params, std::numeric_limits<double>::infinity()))) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        JacobiVectorResult result;
        result.value.resize(tau.size());
        for (std::size_t index = 0; index < tau.size(); ++index) {
            const JacobiScalarResult scalar = jacobi_lamperti_drift(
                params, tau[index], interior_eps);
            if (!scalar.is_ok()) {
                result.status = scalar.status;
                result.failure.index = static_cast<std::int64_t>(index);
                result.value.clear();
                return result;
            }
            result.value[index] = scalar.value;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiBoundaryResult apply_jacobi_boundary(
    double value,
    double upper,
    JacobiBoundaryPolicy policy) noexcept {

    if (!std::isfinite(value) || !finite_positive(upper)
        || (policy != JacobiBoundaryPolicy::Reflect
            && policy != JacobiBoundaryPolicy::Clip)) {
        return failure<JacobiBoundaryResult>(Status::InvalidParameter);
    }
    JacobiBoundaryResult result;
    result.value.intervened = value < 0.0 || value > upper;
    if (!result.value.intervened) {
        result.value.value = value;
        return result;
    }
    if (policy == JacobiBoundaryPolicy::Clip) {
        result.value.value = std::clamp(value, 0.0, upper);
        return result;
    }
    const double period = 2.0 * upper;
    double reflected = std::fmod(value, period);
    if (reflected < 0.0) {
        reflected += period;
    }
    if (reflected > upper) {
        reflected = period - reflected;
    }
    result.value.value = reflected;
    return result;
}

JacobiScalarResult jacobi_log_beta(
    double alpha,
    double beta) noexcept {

    if (!finite_positive(alpha) || !finite_positive(beta)) {
        return failure<JacobiScalarResult>(Status::InvalidParameter);
    }
    JacobiScalarResult result;
    result.value =
        std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
    if (!std::isfinite(result.value)) {
        result.status = Status::NumericalFailure;
    }
    return result;
}

JacobiScalarResult jacobi_digamma(double value) noexcept {
    if (!finite_positive(value)) {
        return failure<JacobiScalarResult>(Status::InvalidParameter);
    }
    double shifted = value;
    double result_value = 0.0;
    while (shifted < 8.0) {
        result_value -= 1.0 / shifted;
        shifted += 1.0;
    }
    const double inverse = 1.0 / shifted;
    const double inverse_squared = inverse * inverse;
    result_value += std::log(shifted) - 0.5 * inverse
        - inverse_squared * (
            1.0 / 12.0
            - inverse_squared * (
                1.0 / 120.0
                - inverse_squared * (
                    1.0 / 252.0
                    - inverse_squared * (
                        1.0 / 240.0
                        - inverse_squared * 5.0 / 660.0))));
    JacobiScalarResult result;
    result.value = result_value;
    return result;
}

JacobiScalarResult jacobi_trigamma(double value) noexcept {
    if (!finite_positive(value)) {
        return failure<JacobiScalarResult>(Status::InvalidParameter);
    }
    double shifted = value;
    double result_value = 0.0;
    while (shifted < 8.0) {
        result_value += 1.0 / (shifted * shifted);
        shifted += 1.0;
    }
    const double inverse = 1.0 / shifted;
    const double inverse_squared = inverse * inverse;
    result_value += inverse + 0.5 * inverse_squared
        + inverse * inverse_squared * (
            1.0 / 6.0
            - inverse_squared * (
                1.0 / 30.0
                - inverse_squared * (
                    1.0 / 42.0
                    - inverse_squared * (
                        1.0 / 30.0
                        - inverse_squared * 5.0 / 66.0))));
    JacobiScalarResult result;
    result.value = result_value;
    return result;
}

}  // namespace scar
