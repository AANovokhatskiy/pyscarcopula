#include "scar/jacobi.hpp"

#include "scar/detail/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace scar {
namespace {

template <typename ResultType>
ResultType failure(Status status) {
    ResultType result;
    result.status = status;
    return result;
}

bool valid_shape(double alpha, double beta) noexcept {
    return std::isfinite(alpha) && alpha > 0.0
        && std::isfinite(beta) && beta > 0.0;
}

bool valid_order(int order) noexcept {
    return order > 0 && order <= kMaxJacobiOrder;
}

bool quadrature_workspace_within_budget(
    int jacobi_order,
    int hermite_order,
    std::uint64_t memory_budget_bytes) noexcept {

    JacobiNumericalConfig config;
    config.quad_order = jacobi_order;
    config.basis_order = 1;
    config.gh_order = hermite_order;
    config.matrix = false;
    config.memory_budget_bytes = memory_budget_bytes;
    return estimate_jacobi_workspace(config).is_ok();
}

bool jacobi_recurrence_coefficients(
    double alpha,
    double beta,
    int order,
    std::vector<double>& diagonal,
    std::vector<double>& off_diagonal) {

    if (!valid_shape(alpha, beta) || !valid_order(order)) {
        return false;
    }
    // Parameters of scipy.special.roots_jacobi for tau=(x+1)/2.
    const double a = beta - 1.0;
    const double b = alpha - 1.0;
    const double sum = a + b;
    const std::size_t count = static_cast<std::size_t>(order);
    diagonal.assign(count, 0.0);
    off_diagonal.assign(count, 0.0);
    diagonal[0] = (b - a) / (sum + 2.0);

    for (int index = 1; index < order; ++index) {
        const double n = static_cast<double>(index);
        const double two_n_sum = 2.0 * n + sum;
        diagonal[static_cast<std::size_t>(index)] =
            (b * b - a * a)
            / (two_n_sum * (two_n_sum + 2.0));

        double square = 0.0;
        if (index == 1) {
            square =
                4.0 * (1.0 + a) * (1.0 + b)
                / ((sum + 2.0) * (sum + 2.0) * (sum + 3.0));
        } else {
            square =
                4.0 * n * (n + a) * (n + b) * (n + sum)
                / (two_n_sum * two_n_sum
                    * (two_n_sum + 1.0)
                    * (two_n_sum - 1.0));
        }
        if (!(square > 0.0) || !std::isfinite(square)) {
            return false;
        }
        off_diagonal[static_cast<std::size_t>(index - 1)] =
            std::sqrt(square);
    }
    return std::all_of(
        diagonal.begin(), diagonal.end(),
        [](double value) { return std::isfinite(value); });
}

JacobiQuadratureResult sorted_rule_from_tridiagonal(
    std::vector<double> diagonal,
    std::vector<double> off_diagonal) {

    const std::size_t order = diagonal.size();
    std::vector<double> eigenvectors;
    if (!scar_internal::linalg::symmetric_tridiagonal_eigen_ql(
            diagonal, off_diagonal, eigenvectors)) {
        return failure<JacobiQuadratureResult>(Status::NumericalFailure);
    }
    std::vector<std::size_t> permutation(order);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::stable_sort(
        permutation.begin(), permutation.end(),
        [&diagonal](std::size_t lhs, std::size_t rhs) {
            return diagonal[lhs] < diagonal[rhs];
        });

    JacobiQuadratureResult result;
    result.value.nodes.resize(order);
    result.value.weights.resize(order);
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < order; ++index) {
        const std::size_t source = permutation[index];
        const double first_component = eigenvectors[source];
        const double weight = first_component * first_component;
        result.value.nodes[index] = diagonal[source];
        result.value.weights[index] = weight;
        weight_sum += weight;
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        return failure<JacobiQuadratureResult>(Status::NumericalFailure);
    }
    for (double& weight : result.value.weights) {
        weight /= weight_sum;
    }
    return result;
}

bool fill_orthonormal_basis(
    const std::vector<double>& tau,
    double alpha,
    double beta,
    int basis_order,
    std::vector<double>& basis,
    std::vector<double>& derivative) {

    std::vector<double> diagonal;
    std::vector<double> off_diagonal;
    if (!jacobi_recurrence_coefficients(
            alpha, beta, basis_order, diagonal, off_diagonal)) {
        return false;
    }
    const std::size_t rows = tau.size();
    const std::size_t columns = static_cast<std::size_t>(basis_order);
    basis.assign(rows * columns, 0.0);
    derivative.assign(rows * columns, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        const double x = 2.0 * tau[row] - 1.0;
        basis[row * columns] = 1.0;
        if (basis_order == 1) {
            continue;
        }
        if (!(off_diagonal[0] > 0.0)) {
            return false;
        }
        basis[row * columns + 1] =
            (x - diagonal[0]) / off_diagonal[0];
        // The returned derivative is with respect to tau, hence dx/dtau=2.
        derivative[row * columns + 1] = 2.0 / off_diagonal[0];
        for (int degree = 1; degree + 1 < basis_order; ++degree) {
            const std::size_t current = static_cast<std::size_t>(degree);
            const std::size_t next = current + 1;
            const double divisor = off_diagonal[current];
            if (!(divisor > 0.0)) {
                return false;
            }
            basis[row * columns + next] = (
                (x - diagonal[current]) * basis[row * columns + current]
                - off_diagonal[current - 1]
                    * basis[row * columns + current - 1]
            ) / divisor;
            derivative[row * columns + next] = (
                2.0 * basis[row * columns + current]
                + (x - diagonal[current])
                    * derivative[row * columns + current]
                - off_diagonal[current - 1]
                    * derivative[row * columns + current - 1]
            ) / divisor;
        }
    }
    return std::all_of(
        basis.begin(), basis.end(),
        [](double value) { return std::isfinite(value); })
        && std::all_of(
            derivative.begin(), derivative.end(),
            [](double value) { return std::isfinite(value); });
}

}  // namespace

JacobiQuadratureResult gauss_hermite_probability_rule(
    int order,
    std::uint64_t memory_budget_bytes) {

    if (!valid_order(order)) {
        return failure<JacobiQuadratureResult>(Status::InvalidSize);
    }
    if (!quadrature_workspace_within_budget(
            1, order, memory_budget_bytes)) {
        JacobiQuadratureResult result =
            failure<JacobiQuadratureResult>(Status::InvalidSize);
        result.failure.operation = 1;
        return result;
    }
    try {
        const std::size_t count = static_cast<std::size_t>(order);
        std::vector<double> diagonal(count, 0.0);
        std::vector<double> off_diagonal(count, 0.0);
        for (std::size_t index = 0; index + 1 < count; ++index) {
            off_diagonal[index] =
                std::sqrt((static_cast<double>(index) + 1.0) / 2.0);
        }
        return sorted_rule_from_tridiagonal(
            std::move(diagonal), std::move(off_diagonal));
    } catch (const std::bad_alloc&) {
        return failure<JacobiQuadratureResult>(Status::InvalidSize);
    }
}

JacobiQuadratureResult gauss_jacobi_probability_rule(
    double alpha,
    double beta,
    int order,
    std::uint64_t memory_budget_bytes) {

    if (!valid_shape(alpha, beta)) {
        return failure<JacobiQuadratureResult>(Status::InvalidParameter);
    }
    if (!valid_order(order)) {
        return failure<JacobiQuadratureResult>(Status::InvalidSize);
    }
    if (!quadrature_workspace_within_budget(
            order, 1, memory_budget_bytes)) {
        JacobiQuadratureResult result =
            failure<JacobiQuadratureResult>(Status::InvalidSize);
        result.failure.operation = 1;
        return result;
    }
    try {
        std::vector<double> diagonal;
        std::vector<double> off_diagonal;
        if (!jacobi_recurrence_coefficients(
                alpha, beta, order, diagonal, off_diagonal)) {
            return failure<JacobiQuadratureResult>(Status::NumericalFailure);
        }
        JacobiQuadratureResult result = sorted_rule_from_tridiagonal(
            std::move(diagonal), std::move(off_diagonal));
        if (!result.is_ok()) {
            return result;
        }
        for (double& node : result.value.nodes) {
            node = 0.5 * (node + 1.0);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiQuadratureResult>(Status::InvalidSize);
    }
}

JacobiBasisResult build_jacobi_rule(
    double alpha,
    double beta,
    int quad_order,
    int basis_order,
    std::uint64_t memory_budget_bytes) {

    if (!valid_shape(alpha, beta)) {
        return failure<JacobiBasisResult>(Status::InvalidParameter);
    }
    if (!valid_order(quad_order)
        || !valid_order(basis_order)
        || basis_order > quad_order) {
        return failure<JacobiBasisResult>(Status::InvalidSize);
    }
    JacobiNumericalConfig config;
    config.quad_order = quad_order;
    config.basis_order = basis_order;
    config.gh_order = 1;
    config.matrix = false;
    config.memory_budget_bytes = memory_budget_bytes;
    const JacobiMemoryResult memory = estimate_jacobi_workspace(config);
    if (!memory.is_ok()) {
        JacobiBasisResult result =
            failure<JacobiBasisResult>(Status::InvalidSize);
        result.failure.operation = 1;
        return result;
    }

    JacobiQuadratureResult quadrature =
        gauss_jacobi_probability_rule(
            alpha, beta, quad_order, memory_budget_bytes);
    if (!quadrature.is_ok()) {
        return failure<JacobiBasisResult>(quadrature.status);
    }
    try {
        JacobiBasisResult result;
        result.value.tau = std::move(quadrature.value.nodes);
        result.value.weights = std::move(quadrature.value.weights);
        result.value.quad_order = quad_order;
        result.value.basis_order = basis_order;
        if (!fill_orthonormal_basis(
                result.value.tau,
                alpha,
                beta,
                basis_order,
                result.value.basis,
                result.value.basis_derivative)) {
            return failure<JacobiBasisResult>(Status::NumericalFailure);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiBasisResult>(Status::InvalidSize);
    }
}

JacobiFixedRuleResult build_fixed_jacobi_rule(
    const JacobiParams& params,
    int quad_order,
    std::uint64_t memory_budget_bytes) {

    if (!valid_order(quad_order)) {
        return failure<JacobiFixedRuleResult>(Status::InvalidSize);
    }
    const JacobiShapeResult shape = jacobi_stationary_shape(params);
    if (!shape.is_ok()) {
        return failure<JacobiFixedRuleResult>(shape.status);
    }
    JacobiNumericalConfig config;
    config.quad_order = quad_order;
    config.basis_order = 1;
    config.gh_order = 1;
    config.matrix = false;
    config.memory_budget_bytes = memory_budget_bytes;
    const JacobiMemoryResult memory = estimate_jacobi_workspace(config);
    if (!memory.is_ok()) {
        JacobiFixedRuleResult result =
            failure<JacobiFixedRuleResult>(Status::InvalidSize);
        result.failure.operation = 1;
        return result;
    }
    const JacobiScalarResult log_beta = jacobi_log_beta(
        shape.value.alpha, shape.value.beta);
    const JacobiScalarResult psi_alpha =
        jacobi_digamma(shape.value.alpha);
    const JacobiScalarResult psi_beta =
        jacobi_digamma(shape.value.beta);
    const JacobiScalarResult psi_sum =
        jacobi_digamma(shape.value.alpha + shape.value.beta);
    if (!log_beta.is_ok() || !psi_alpha.is_ok()
        || !psi_beta.is_ok() || !psi_sum.is_ok()) {
        return failure<JacobiFixedRuleResult>(Status::NumericalFailure);
    }

    try {
        const std::size_t count = static_cast<std::size_t>(quad_order);
        JacobiFixedRuleResult result;
        result.value.quad_order = quad_order;
        result.value.tau.resize(count);
        result.value.weights.resize(count);
        result.value.weight_derivatives.assign(3 * count, 0.0);
        const double epsilon = 0.5 / (static_cast<double>(quad_order) + 1.0);
        const double step = quad_order > 1
            ? (1.0 - 2.0 * epsilon)
                / static_cast<double>(quad_order - 1)
            : 0.0;
        double maximum_log_mass = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < count; ++index) {
            const double tau = quad_order == 1
                ? 0.5
                : epsilon + step * static_cast<double>(index);
            result.value.tau[index] = tau;
            const double width = quad_order == 1 ? 1.0 : step;
            const double log_mass =
                (shape.value.alpha - 1.0) * std::log(tau)
                + (shape.value.beta - 1.0) * std::log1p(-tau)
                - log_beta.value
                + std::log(width);
            result.value.weights[index] = log_mass;
            maximum_log_mass = std::max(maximum_log_mass, log_mass);
        }
        double total = 0.0;
        for (double& weight : result.value.weights) {
            weight = std::exp(weight - maximum_log_mass);
            total += weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            return failure<JacobiFixedRuleResult>(Status::NumericalFailure);
        }
        for (double& weight : result.value.weights) {
            weight /= total;
        }

        std::vector<double> score(count, 0.0);
        for (std::size_t parameter = 0; parameter < 3; ++parameter) {
            double mean_score = 0.0;
            for (std::size_t index = 0; index < count; ++index) {
                const double tau = result.value.tau[index];
                const double dlog_alpha =
                    std::log(tau) - psi_alpha.value + psi_sum.value;
                const double dlog_beta =
                    std::log1p(-tau) - psi_beta.value + psi_sum.value;
                score[index] =
                    shape.value.dalpha[parameter] * dlog_alpha
                    + shape.value.dbeta[parameter] * dlog_beta;
                mean_score += result.value.weights[index] * score[index];
            }
            for (std::size_t index = 0; index < count; ++index) {
                result.value.weight_derivatives[parameter * count + index] =
                    result.value.weights[index]
                    * (score[index] - mean_score);
            }
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiFixedRuleResult>(Status::InvalidSize);
    }
}

JacobiVectorResult evaluate_jacobi_polynomials(
    double x,
    double alpha,
    double beta,
    int order,
    bool derivative) {

    if (!std::isfinite(x) || !valid_shape(alpha, beta)) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    if (!valid_order(order)) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
    try {
        std::vector<double> diagonal;
        std::vector<double> off_diagonal;
        if (!jacobi_recurrence_coefficients(
                alpha, beta, order, diagonal, off_diagonal)) {
            return failure<JacobiVectorResult>(Status::NumericalFailure);
        }
        JacobiVectorResult result;
        result.value.assign(static_cast<std::size_t>(order), 0.0);
        std::vector<double> values(static_cast<std::size_t>(order), 0.0);
        values[0] = 1.0;
        if (order > 1) {
            values[1] = (x - diagonal[0]) / off_diagonal[0];
            result.value[1] = 1.0 / off_diagonal[0];
        }
        for (int degree = 1; degree + 1 < order; ++degree) {
            const std::size_t current = static_cast<std::size_t>(degree);
            const std::size_t next = current + 1;
            values[next] = (
                (x - diagonal[current]) * values[current]
                - off_diagonal[current - 1] * values[current - 1]
            ) / off_diagonal[current];
            result.value[next] = (
                values[current]
                + (x - diagonal[current]) * result.value[current]
                - off_diagonal[current - 1] * result.value[current - 1]
            ) / off_diagonal[current];
        }
        if (!derivative) {
            result.value = std::move(values);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

}  // namespace scar
