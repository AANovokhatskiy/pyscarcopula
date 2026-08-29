#include "scar/jacobi.hpp"

#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"
#include "scar/numerical_constants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace scar {
namespace {

constexpr double kProbabilityNegativeTolerance = 1e-12;

template <typename ResultType>
ResultType failed(Status status, int operation = -1) {
    ResultType result;
    result.status = status;
    result.failure.operation = operation;
    return result;
}

template <typename ResultType>
ResultType failed_with_diagnostics(
    Status status,
    int operation,
    const JacobiFilterDiagnostics& diagnostics) {

    ResultType result = failed<ResultType>(status, operation);
    result.value.diagnostics = diagnostics;
    return result;
}

bool same_params(const JacobiParams& lhs, const JacobiParams& rhs) noexcept {
    return lhs.kappa == rhs.kappa && lhs.m == rhs.m && lhs.xi == rhs.xi;
}

double sum_values(const double* values, std::size_t size) noexcept {
    double total = 0.0;
    for (std::size_t index = 0; index < size; ++index) {
        total += values[index];
    }
    return total;
}

bool normalize_probability(
    std::vector<double>& values,
    double negative_tolerance = kProbabilityNegativeTolerance) {

    double total = 0.0;
    for (double& value : values) {
        if (!std::isfinite(value) || value < -negative_tolerance) {
            return false;
        }
        value = value > 0.0 ? value : 0.0;
        total += value;
    }
    if (!std::isfinite(total) || !(total > 0.0)) {
        return false;
    }
    const double inverse = 1.0 / total;
    for (double& value : values) {
        value *= inverse;
    }
    return true;
}

double mass_error(const double* values, std::size_t size) noexcept {
    return std::abs(sum_values(values, size) - 1.0);
}

void dense_left_multiply(
    const std::vector<double>& matrix,
    int order,
    const double* probability,
    std::vector<double>& output) {

    const std::size_t k = static_cast<std::size_t>(order);
    output.assign(k, 0.0);
    for (std::size_t row = 0; row < k; ++row) {
        const double source = probability[row];
        const std::size_t offset = row * k;
        for (std::size_t column = 0; column < k; ++column) {
            output[column] += source * matrix[offset + column];
        }
    }
}

void sparse_left_multiply(
    const JacobiSparseTransition& transition,
    const double* probability,
    std::vector<double>& output) {

    const std::size_t k = static_cast<std::size_t>(transition.order);
    const std::size_t width = static_cast<std::size_t>(transition.max_width);
    output.assign(k, 0.0);
    for (std::size_t row = 0; row < k; ++row) {
        const double source = probability[row];
        const std::size_t count = static_cast<std::size_t>(
            transition.counts[row]);
        for (std::size_t slot = 0; slot < count; ++slot) {
            const std::size_t offset = row * width + slot;
            const std::int64_t target = transition.indices[offset];
            if (target >= 0 && target < transition.order) {
                output[static_cast<std::size_t>(target)] +=
                    source * transition.probabilities[offset];
            }
        }
    }
}

std::vector<double> dense_from_sparse(
    const JacobiSparseTransition& transition) {

    const std::size_t k = static_cast<std::size_t>(transition.order);
    const std::size_t width = static_cast<std::size_t>(transition.max_width);
    std::vector<double> dense(k * k, 0.0);
    for (std::size_t row = 0; row < k; ++row) {
        const std::size_t count = static_cast<std::size_t>(
            transition.counts[row]);
        for (std::size_t slot = 0; slot < count; ++slot) {
            const std::size_t offset = row * width + slot;
            const std::int64_t target = transition.indices[offset];
            if (target >= 0 && target < transition.order) {
                dense[row * k + static_cast<std::size_t>(target)] =
                    transition.probabilities[offset];
            }
        }
    }
    return dense;
}

std::vector<double> derivative_dense_from_sparse(
    const JacobiSparseTransition& transition) {

    const std::size_t k = static_cast<std::size_t>(transition.order);
    const std::size_t width = static_cast<std::size_t>(transition.max_width);
    std::vector<double> dense(3 * k * k, 0.0);
    if (transition.derivatives.size() != 3 * k * width) {
        return {};
    }
    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
        for (std::size_t row = 0; row < k; ++row) {
            const std::size_t count = static_cast<std::size_t>(
                transition.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t sparse_offset = row * width + slot;
                const std::int64_t target = transition.indices[sparse_offset];
                if (target >= 0 && target < transition.order) {
                    dense[parameter * k * k
                        + row * k + static_cast<std::size_t>(target)] =
                        transition.derivatives[
                            parameter * k * width + sparse_offset];
                }
            }
        }
    }
    return dense;
}

void repair_and_clip_h(std::vector<double>& row) {
    constexpr double epsilon = numerical::kHFunctionEps;
    std::vector<std::size_t> finite;
    finite.reserve(row.size());
    for (std::size_t index = 0; index < row.size(); ++index) {
        if (std::isfinite(row[index])) {
            finite.push_back(index);
        }
    }
    if (finite.empty()) {
        std::fill(row.begin(), row.end(), 0.5);
        return;
    }
    for (std::size_t index = 0; index < row.size(); ++index) {
        if (std::isfinite(row[index])) {
            row[index] = std::clamp(row[index], epsilon, 1.0 - epsilon);
            continue;
        }
        const auto upper = std::lower_bound(finite.begin(), finite.end(), index);
        if (upper == finite.begin()) {
            row[index] = row[*upper];
        } else if (upper == finite.end()) {
            row[index] = row[finite.back()];
        } else {
            const std::size_t right = *upper;
            const std::size_t left = *(upper - 1);
            const double fraction = static_cast<double>(index - left)
                / static_cast<double>(right - left);
            row[index] = row[left] + fraction * (row[right] - row[left]);
        }
        row[index] = std::clamp(row[index], epsilon, 1.0 - epsilon);
    }
}

struct EvaluatorSetup {
    JacobiParams params{};
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> theta;
    std::vector<double> emissions;
    JacobiDenseTransition dense{};
    JacobiSparseTransition sparse{};
    JacobiCoefficientTransition coefficient{};
    JacobiTransitionDiagnostics diagnostics{};
    int order = 0;
    bool is_sparse = false;
    bool is_coefficient = false;
    bool has_derivatives = false;
    std::uint64_t generation = 0;
};

std::vector<double> setup_dense_transition(const EvaluatorSetup& setup) {
    if (setup.is_sparse) {
        return dense_from_sparse(setup.sparse);
    }
    if (!setup.is_coefficient) {
        return setup.dense.probabilities;
    }
    const std::size_t k = static_cast<std::size_t>(setup.order);
    const std::size_t b = static_cast<std::size_t>(
        setup.coefficient.basis_order);
    std::vector<double> transition(k * k, 0.0);
    for (std::size_t row = 0; row < k; ++row) {
        for (std::size_t column = 0; column < k; ++column) {
            double kernel = 0.0;
            for (std::size_t basis = 0; basis < b; ++basis) {
                kernel += setup.coefficient.basis[row * b + basis]
                    * setup.coefficient.spectral_powers[basis]
                    * setup.coefficient.basis[column * b + basis];
            }
            transition[row * k + column] = setup.weights[column] * kernel;
        }
    }
    return transition;
}

void setup_left_multiply(
    const EvaluatorSetup& setup,
    const double* probability,
    std::vector<double>& output) {

    if (setup.is_sparse) {
        sparse_left_multiply(setup.sparse, probability, output);
    } else {
        dense_left_multiply(
            setup.dense.probabilities, setup.order, probability, output);
    }
}

std::vector<double> coefficient_probability(
    const EvaluatorSetup& setup,
    const std::vector<double>& coefficients,
    bool clip_negative) {

    const std::size_t k = static_cast<std::size_t>(setup.order);
    const std::size_t b = static_cast<std::size_t>(
        setup.coefficient.basis_order);
    std::vector<double> probability(k, 0.0);
    for (std::size_t row = 0; row < k; ++row) {
        double density_ratio = 0.0;
        for (std::size_t basis = 0; basis < b; ++basis) {
            density_ratio += setup.coefficient.basis[row * b + basis]
                * coefficients[basis];
        }
        double value = setup.weights[row] * density_ratio;
        if (clip_negative && (!std::isfinite(value) || value <= 0.0)) {
            value = 0.0;
        }
        probability[row] = value;
    }
    if (clip_negative && !normalize_probability(probability, 0.0)) {
        std::fill(
            probability.begin(), probability.end(),
            1.0 / static_cast<double>(k));
    }
    return probability;
}

bool compute_filter(
    const EvaluatorSetup& setup,
    JacobiFilterState& state) {

    const std::size_t k = static_cast<std::size_t>(setup.order);
    const std::size_t n = setup.emissions.size() / k;
    state.tau = setup.tau;
    state.theta = setup.theta;
    state.emissions = setup.emissions;
    state.n_obs = static_cast<std::int64_t>(n);
    state.order = setup.order;
    state.predicted.assign(n * k, 0.0);
    state.filtered.assign(n * k, 0.0);
    state.smoothed.assign(n * k, 0.0);
    state.scales.assign(n, 0.0);
    state.diagnostics.transition = setup.diagnostics;
    state.diagnostics.n_obs = state.n_obs;
    state.diagnostics.order = setup.order;
    state.diagnostics.minimum_scale = std::numeric_limits<double>::infinity();
    state.diagnostics.maximum_scale = 0.0;
    state.diagnostics.preparation_generation = setup.generation;

    double log_likelihood = 0.0;
    if (setup.is_coefficient) {
        const std::size_t b = static_cast<std::size_t>(
            setup.coefficient.basis_order);
        std::vector<double> coefficients(b, 0.0);
        coefficients[0] = 1.0;
        std::vector<double> predicted_coefficients(b, 0.0);
        for (std::size_t observation = 0; observation < n; ++observation) {
            for (std::size_t basis = 0; basis < b; ++basis) {
                predicted_coefficients[basis] = coefficients[basis]
                    * setup.coefficient.spectral_powers[basis];
            }
            std::vector<double> predicted_probability = coefficient_probability(
                setup, predicted_coefficients, false);
            std::copy(
                predicted_probability.begin(), predicted_probability.end(),
                state.predicted.begin() + observation * k);

            std::vector<double> raw(b, 0.0);
            for (std::size_t basis = 0; basis < b; ++basis) {
                double value = 0.0;
                for (std::size_t row = 0; row < k; ++row) {
                    double density_ratio = 0.0;
                    for (std::size_t inner = 0; inner < b; ++inner) {
                        density_ratio +=
                            setup.coefficient.basis[row * b + inner]
                            * predicted_coefficients[inner];
                    }
                    value += setup.coefficient.basis[row * b + basis]
                        * setup.weights[row]
                        * setup.emissions[observation * k + row]
                        * density_ratio;
                }
                raw[basis] = value;
            }
            const double scale = raw[0];
            if (!std::isfinite(scale) || !(scale > 0.0)) {
                return false;
            }
            for (std::size_t basis = 0; basis < b; ++basis) {
                coefficients[basis] = raw[basis] / scale;
            }
            std::vector<double> filtered_probability = coefficient_probability(
                setup, coefficients, true);
            std::copy(
                filtered_probability.begin(), filtered_probability.end(),
                state.filtered.begin() + observation * k);
            state.scales[observation] = scale;
            log_likelihood += std::log(scale);
            state.diagnostics.minimum_scale = std::min(
                state.diagnostics.minimum_scale, scale);
            state.diagnostics.maximum_scale = std::max(
                state.diagnostics.maximum_scale, scale);
            state.diagnostics.max_predictive_mass_error = std::max(
                state.diagnostics.max_predictive_mass_error,
                mass_error(state.predicted.data() + observation * k, k));
            state.diagnostics.max_filtered_mass_error = std::max(
                state.diagnostics.max_filtered_mass_error,
                mass_error(state.filtered.data() + observation * k, k));
        }
        state.current_probability = coefficient_probability(
            setup, coefficients, true);
        for (std::size_t basis = 0; basis < b; ++basis) {
            coefficients[basis] *= setup.coefficient.spectral_powers[basis];
        }
        state.next_probability = coefficient_probability(
            setup, coefficients, true);
    } else {
        std::vector<double> predicted = setup.weights;
        std::vector<double> posterior(k, 0.0);
        for (std::size_t observation = 0; observation < n; ++observation) {
            std::copy(
                predicted.begin(), predicted.end(),
                state.predicted.begin() + observation * k);
            double scale = 0.0;
            for (std::size_t node = 0; node < k; ++node) {
                posterior[node] = predicted[node]
                    * setup.emissions[observation * k + node];
                scale += posterior[node];
            }
            if (!std::isfinite(scale) || !(scale > 0.0)) {
                return false;
            }
            const double inverse_scale = 1.0 / scale;
            for (double& value : posterior) {
                value *= inverse_scale;
            }
            if (!normalize_probability(posterior)) {
                return false;
            }
            std::copy(
                posterior.begin(), posterior.end(),
                state.filtered.begin() + observation * k);
            state.scales[observation] = scale;
            log_likelihood += std::log(scale);
            state.diagnostics.minimum_scale = std::min(
                state.diagnostics.minimum_scale, scale);
            state.diagnostics.maximum_scale = std::max(
                state.diagnostics.maximum_scale, scale);
            state.diagnostics.max_predictive_mass_error = std::max(
                state.diagnostics.max_predictive_mass_error,
                mass_error(predicted.data(), k));
            state.diagnostics.max_filtered_mass_error = std::max(
                state.diagnostics.max_filtered_mass_error,
                mass_error(posterior.data(), k));
            if (observation + 1 < n) {
                setup_left_multiply(setup, posterior.data(), predicted);
                if (!normalize_probability(predicted)) {
                    return false;
                }
            }
        }
        state.current_probability = posterior;
        setup_left_multiply(setup, posterior.data(), state.next_probability);
        if (!normalize_probability(state.next_probability)) {
            return false;
        }
    }

    state.diagnostics.log_likelihood = log_likelihood;
    if (!std::isfinite(log_likelihood)) {
        return false;
    }

    // Backward smoothing uses the same transition representation as the
    // forward recursion.  The coefficient backend materializes its spectral
    // node transition only for this backward state operation.
    const std::vector<double> coefficient_dense = setup.is_coefficient
        ? setup_dense_transition(setup) : std::vector<double>{};
    std::copy(
        state.filtered.end() - static_cast<std::ptrdiff_t>(k),
        state.filtered.end(),
        state.smoothed.end() - static_cast<std::ptrdiff_t>(k));
    for (std::size_t observation = n - 1; observation-- > 0;) {
        const double* next_smoothed =
            state.smoothed.data() + (observation + 1) * k;
        const double* next_predicted =
            state.predicted.data() + (observation + 1) * k;
        std::vector<double> ratio(k, 0.0);
        for (std::size_t node = 0; node < k; ++node) {
            if (next_predicted[node] > 0.0) {
                ratio[node] = next_smoothed[node] / next_predicted[node];
            }
        }
        std::vector<double> backward(k, 0.0);
        if (setup.is_sparse) {
            const std::size_t width = static_cast<std::size_t>(
                setup.sparse.max_width);
            for (std::size_t row = 0; row < k; ++row) {
                const std::size_t count = static_cast<std::size_t>(
                    setup.sparse.counts[row]);
                for (std::size_t slot = 0; slot < count; ++slot) {
                    const std::size_t offset = row * width + slot;
                    const std::int64_t target = setup.sparse.indices[offset];
                    if (target >= 0 && target < setup.order) {
                        backward[row] += setup.sparse.probabilities[offset]
                            * ratio[static_cast<std::size_t>(target)];
                    }
                }
            }
        } else {
            const std::vector<double>& matrix = setup.is_coefficient
                ? coefficient_dense : setup.dense.probabilities;
            for (std::size_t row = 0; row < k; ++row) {
                for (std::size_t column = 0; column < k; ++column) {
                    backward[row] += matrix[row * k + column] * ratio[column];
                }
            }
        }
        std::vector<double> smoothed(k, 0.0);
        const double* filtered = state.filtered.data() + observation * k;
        for (std::size_t node = 0; node < k; ++node) {
            smoothed[node] = filtered[node] * backward[node];
        }
        if (!normalize_probability(smoothed)) {
            // Signed spectral coefficient smoothing is not part of the
            // likelihood recursion; retain the valid filtered distribution
            // if its node representation cannot support a backward update.
            if (!setup.is_coefficient) {
                return false;
            }
            std::copy(filtered, filtered + k, smoothed.begin());
        }
        std::copy(
            smoothed.begin(), smoothed.end(),
            state.smoothed.begin() + observation * k);
    }
    for (std::size_t observation = 0; observation < n; ++observation) {
        state.diagnostics.max_smoothed_mass_error = std::max(
            state.diagnostics.max_smoothed_mass_error,
            mass_error(state.smoothed.data() + observation * k, k));
    }
    return true;
}

bool filter_gradient(
    const EvaluatorSetup& setup,
    const std::vector<double>& dweights,
    const std::vector<double>& dtransition,
    const std::vector<double>& demissions,
    JacobiObjectiveGradient& output) {

    const std::size_t k = static_cast<std::size_t>(setup.order);
    const std::size_t n = setup.emissions.size() / k;
    const std::vector<double> transition = setup_dense_transition(setup);
    if (dweights.size() != 3 * k
        || dtransition.size() != 3 * k * k
        || demissions.size() != 3 * n * k
        || transition.size() != k * k) {
        return false;
    }
    std::vector<double> predicted = setup.weights;
    std::vector<double> dpredicted = dweights;
    std::vector<double> posterior(k, 0.0);
    std::vector<double> dposterior(3 * k, 0.0);
    double log_likelihood = 0.0;
    std::array<double, 3> gradient{};

    for (std::size_t observation = 0; observation < n; ++observation) {
        double scale = 0.0;
        std::array<double, 3> dscale{};
        for (std::size_t node = 0; node < k; ++node) {
            const double emission = setup.emissions[observation * k + node];
            posterior[node] = predicted[node] * emission;
            scale += posterior[node];
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                const double value =
                    dpredicted[parameter * k + node] * emission
                    + predicted[node]
                    * demissions[(parameter * n + observation) * k + node];
                dposterior[parameter * k + node] = value;
                dscale[parameter] += value;
            }
        }
        if (!std::isfinite(scale) || !(scale > 0.0)) {
            return false;
        }
        for (std::size_t node = 0; node < k; ++node) {
            const double raw = posterior[node];
            posterior[node] = raw / scale;
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                dposterior[parameter * k + node] =
                    (dposterior[parameter * k + node] * scale
                     - raw * dscale[parameter])
                    / (scale * scale);
            }
        }
        log_likelihood += std::log(scale);
        for (std::size_t parameter = 0; parameter < 3; ++parameter) {
            gradient[parameter] += dscale[parameter] / scale;
        }
        if (observation + 1 < n) {
            std::vector<double> next(k, 0.0);
            std::vector<double> dnext(3 * k, 0.0);
            for (std::size_t row = 0; row < k; ++row) {
                for (std::size_t column = 0; column < k; ++column) {
                    const double probability = transition[row * k + column];
                    next[column] += posterior[row] * probability;
                    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                        dnext[parameter * k + column] +=
                            dposterior[parameter * k + row] * probability
                            + posterior[row]
                            * dtransition[parameter * k * k + row * k + column];
                    }
                }
            }
            const double total = sum_values(next.data(), k);
            if (!std::isfinite(total) || !(total > 0.0)) {
                return false;
            }
            std::array<double, 3> dtotal{};
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                dtotal[parameter] = sum_values(
                    dnext.data() + parameter * k, k);
            }
            for (std::size_t node = 0; node < k; ++node) {
                const double raw = next[node];
                predicted[node] = raw / total;
                for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                    dpredicted[parameter * k + node] =
                        (dnext[parameter * k + node] * total
                         - raw * dtotal[parameter])
                        / (total * total);
                }
            }
        }
    }
    if (!std::isfinite(log_likelihood)
        || !std::all_of(
            gradient.begin(), gradient.end(),
            [](double value) { return std::isfinite(value); })) {
        return false;
    }
    output.objective = -log_likelihood;
    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
        output.gradient[parameter] = -gradient[parameter];
    }
    return true;
}

}  // namespace

struct PreparedScarJacobiEvaluator::Impl {
    Impl(
        CopulaSpec source_copula,
        std::vector<double> source_observations,
        std::int64_t source_n_obs,
        int source_dim,
        JacobiEvaluatorConfig source_config)
        : copula(std::move(source_copula)),
          pair(copula),
          observations(std::move(source_observations)),
          n_obs(source_n_obs),
          dim(source_dim),
          config(source_config) {

        if (!pair.is_registered() || !pair.is_supported()
            || pair.family() == CopulaFamily::Independent) {
            throw std::invalid_argument(
                "copula is not supported by SCAR-TM-Jacobi");
        }
        if (n_obs < 1 || dim != 2 || copula.dim != 2) {
            throw std::invalid_argument("u must have shape (n, 2), n >= 1");
        }
        std::size_t rows = 0;
        std::size_t expected = 0;
        if (!scar_internal::checked_nonnegative_size(n_obs, rows)
            || !scar_internal::checked_size_mul(rows, 2, expected)
            || observations.size() != expected) {
            throw std::invalid_argument("u shape is not representable");
        }
        if (!std::all_of(
                observations.begin(), observations.end(),
                [](double value) { return std::isfinite(value); })) {
            throw std::invalid_argument("u must contain only finite values");
        }
        if (!ok(validate_jacobi_config(config.transition.numerical))) {
            throw std::invalid_argument("invalid Jacobi evaluator config");
        }
        if (config.transition.numerical.n_obs != n_obs) {
            throw std::invalid_argument(
                "Jacobi evaluator n_obs must match observations");
        }
        if (!std::isfinite(config.finite_difference_relative_step)
            || !(config.finite_difference_relative_step > 0.0)) {
            throw std::invalid_argument(
                "finite_difference_relative_step must be positive");
        }
    }

    Result<EvaluatorSetup> build_setup(
        const JacobiParams& params,
        const JacobiTransitionConfig& requested_config,
        std::uint64_t generation) const {

        Result<EvaluatorSetup> result;
        const Status parameter_status = validate_jacobi_params(
            params, requested_config.numerical.stationary_shape_max);
        if (!ok(parameter_status)) {
            result.status = parameter_status;
            return result;
        }
        if (n_obs < 2) {
            result.status = Status::InvalidSize;
            return result;
        }
        EvaluatorSetup& setup = result.value;
        setup.params = params;
        setup.generation = generation;
        const JacobiMemoryResult memory = estimate_jacobi_workspace(
            requested_config.numerical);
        setup.diagnostics.estimated_workspace_bytes = memory.value.bytes;
        setup.diagnostics.memory_budget_bytes =
            requested_config.numerical.memory_budget_bytes;
        setup.diagnostics.method_requested = requested_config.method;
        setup.diagnostics.method_used = requested_config.method;
        setup.diagnostics.storage = requested_config.storage;
        setup.diagnostics.correction = requested_config.correction;
        if (!memory.is_ok() || !memory.value.within_budget) {
            result.status = memory.is_ok() ? Status::InvalidSize : memory.status;
            result.failure.operation = 13;
            return result;
        }

        if (requested_config.method == JacobiTransitionMethod::SpectralCoeff) {
            const JacobiCoefficientTransitionResult transition =
                build_jacobi_coefficient_transition(params, requested_config);
            if (!transition.is_ok()) {
                result.status = transition.status;
                result.failure = transition.failure;
                return result;
            }
            setup.coefficient = transition.value;
            setup.tau = setup.coefficient.tau;
            setup.weights = setup.coefficient.weights;
            setup.order = setup.coefficient.quad_order;
            setup.diagnostics = setup.coefficient.diagnostics;
            setup.is_coefficient = true;
        } else if (requested_config.storage == JacobiTransitionStorage::Sparse) {
            const JacobiSparseTransitionResult transition =
                build_jacobi_sparse_transition(params, requested_config);
            if (!transition.is_ok()) {
                result.status = transition.status;
                result.failure = transition.failure;
                return result;
            }
            setup.sparse = transition.value;
            setup.tau = setup.sparse.tau;
            setup.weights = setup.sparse.weights;
            setup.order = setup.sparse.order;
            setup.diagnostics = setup.sparse.diagnostics;
            setup.is_sparse = true;
            setup.has_derivatives = !setup.sparse.derivatives.empty();
        } else {
            const JacobiDenseTransitionResult transition =
                build_jacobi_dense_transition(params, requested_config);
            if (!transition.is_ok()) {
                result.status = transition.status;
                result.failure = transition.failure;
                return result;
            }
            setup.dense = transition.value;
            setup.tau = setup.dense.tau;
            setup.weights = setup.dense.weights;
            setup.order = setup.dense.order;
            setup.diagnostics = setup.dense.diagnostics;
            setup.has_derivatives = !setup.dense.derivatives.empty();
        }

        const std::size_t k = static_cast<std::size_t>(setup.order);
        const std::size_t n = static_cast<std::size_t>(n_obs);
        setup.theta.resize(k);
        for (std::size_t node = 0; node < k; ++node) {
            double theta = pair.tau_to_parameter(setup.tau[node]);
            if (!std::isnan(requested_config.numerical.theta_cap)) {
                theta = std::min(
                    theta, requested_config.numerical.theta_cap);
            }
            if (!std::isfinite(theta)) {
                result.status = Status::NumericalFailure;
                result.failure.index = static_cast<std::int64_t>(node);
                result.failure.operation = 10;
                return result;
            }
            setup.theta[node] = theta;
        }
        setup.emissions.resize(n * k);
        for (std::size_t observation = 0; observation < n; ++observation) {
            pair.fill_grid_row(
                observations[2 * observation],
                observations[2 * observation + 1],
                setup.theta,
                setup.emissions.data() + observation * k);
            for (std::size_t node = 0; node < k; ++node) {
                const double value = setup.emissions[observation * k + node];
                if (!std::isfinite(value) || value < 0.0) {
                    result.status = Status::NumericalFailure;
                    result.failure.row = static_cast<std::int64_t>(observation);
                    result.failure.index = static_cast<std::int64_t>(node);
                    result.failure.operation = 11;
                    return result;
                }
            }
        }
        return result;
    }

    const Result<EvaluatorSetup>& setup_for(
        const JacobiParams& params,
        bool derivatives) const {

        if (setup_cache.is_ok()
            && same_params(setup_cache.value.params, params)
            && (!derivatives || setup_cache.value.has_derivatives
                || config.transition.method
                    != JacobiTransitionMethod::LocalFixed)) {
            return setup_cache;
        }
        JacobiTransitionConfig requested = config.transition;
        requested.derivatives = derivatives
            && requested.method == JacobiTransitionMethod::LocalFixed;
        requested.numerical.gradient = derivatives;
        ++preparation_generation;
        setup_cache = build_setup(params, requested, preparation_generation);
        filter_cache = {};
        return setup_cache;
    }

    const JacobiFilterResult& filter_for(const JacobiParams& params) const {
        const Result<EvaluatorSetup>& setup_result = setup_for(params, false);
        if (!setup_result.is_ok()) {
            filter_cache = failed<JacobiFilterResult>(
                setup_result.status, setup_result.failure.operation);
            filter_cache.failure = setup_result.failure;
            filter_cache.value.n_obs = n_obs;
            filter_cache.value.order = setup_result.value.order;
            filter_cache.value.diagnostics.transition =
                setup_result.value.diagnostics;
            filter_cache.value.diagnostics.n_obs = n_obs;
            filter_cache.value.diagnostics.order = setup_result.value.order;
            filter_cache.value.diagnostics.preparation_generation =
                setup_result.value.generation;
            return filter_cache;
        }
        if (filter_cache.is_ok()
            && filter_cache.value.diagnostics.preparation_generation
                == setup_result.value.generation) {
            return filter_cache;
        }
        filter_cache = {};
        try {
            if (!compute_filter(setup_result.value, filter_cache.value)) {
                filter_cache.status = Status::NumericalFailure;
                filter_cache.failure.operation = 12;
            }
        } catch (const std::bad_alloc&) {
            filter_cache.status = Status::InvalidSize;
            filter_cache.failure.operation = 13;
        }
        return filter_cache;
    }

    JacobiFilterDiagnostics diagnostics_for(
        const EvaluatorSetup& setup) const {
        JacobiFilterDiagnostics diagnostics;
        diagnostics.transition = setup.diagnostics;
        diagnostics.n_obs = n_obs;
        diagnostics.order = setup.order;
        diagnostics.preparation_generation = setup.generation;
        return diagnostics;
    }

    mutable std::mutex mutex;
    CopulaSpec copula;
    PreparedPairKernel pair;
    std::vector<double> observations;
    std::int64_t n_obs = 0;
    int dim = 2;
    JacobiEvaluatorConfig config{};
    mutable Result<EvaluatorSetup> setup_cache{
        {}, Status::InvalidParameter, {}};
    mutable JacobiFilterResult filter_cache{
        {}, Status::InvalidParameter, {}};
    mutable std::uint64_t preparation_generation = 0;
};

PreparedScarJacobiEvaluator::PreparedScarJacobiEvaluator(
    CopulaSpec copula,
    std::vector<double> observations,
    std::int64_t n_obs,
    int dim,
    JacobiEvaluatorConfig config)
    : impl_(std::make_unique<Impl>(
        std::move(copula),
        std::move(observations),
        n_obs,
        dim,
        config)) {}

PreparedScarJacobiEvaluator::~PreparedScarJacobiEvaluator() = default;
PreparedScarJacobiEvaluator::PreparedScarJacobiEvaluator(
    PreparedScarJacobiEvaluator&&) noexcept = default;
PreparedScarJacobiEvaluator& PreparedScarJacobiEvaluator::operator=(
    PreparedScarJacobiEvaluator&&) noexcept = default;

JacobiFilterResult PreparedScarJacobiEvaluator::filter(
    const JacobiParams& params) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->filter_for(params);
}

JacobiObjectiveResult PreparedScarJacobiEvaluator::loglik(
    const JacobiParams& params) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!ok(validate_jacobi_params(
            params,
            impl_->config.transition.numerical.stationary_shape_max))) {
        JacobiObjectiveResult result;
        result.value.log_likelihood =
            -std::numeric_limits<double>::infinity();
        result.value.objective = std::numeric_limits<double>::infinity();
        return result;
    }
    const JacobiFilterResult& filtered = impl_->filter_for(params);
    if (!filtered.is_ok()) {
        JacobiObjectiveResult result = failed<JacobiObjectiveResult>(
            filtered.status, filtered.failure.operation);
        result.failure = filtered.failure;
        result.value.diagnostics = filtered.value.diagnostics;
        return result;
    }
    JacobiObjectiveResult result;
    result.value.log_likelihood = filtered.value.diagnostics.log_likelihood;
    result.value.objective = -result.value.log_likelihood;
    result.value.diagnostics = filtered.value.diagnostics;
    return result;
}

JacobiGradientResult PreparedScarJacobiEvaluator::neg_loglik_with_grad(
    const JacobiParams& params) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    JacobiFilterDiagnostics failure_diagnostics;
    try {
        const bool fixed = impl_->config.transition.method
            == JacobiTransitionMethod::LocalFixed;
        const Result<EvaluatorSetup>& base_result =
            impl_->setup_for(params, fixed);
        failure_diagnostics = impl_->diagnostics_for(base_result.value);
        if (!base_result.is_ok()) {
            JacobiGradientResult result =
                failed_with_diagnostics<JacobiGradientResult>(
                    base_result.status,
                    base_result.failure.operation,
                    failure_diagnostics);
            result.failure = base_result.failure;
            return result;
        }
        const EvaluatorSetup base = base_result.value;

        // The coefficient likelihood projects every emission back into the
        // truncated basis, so differentiating an equivalent node transition
        // would not preserve that recursion.  Its frozen contract is a
        // physical central-difference gradient; keep the complete setup and
        // filter evaluation in C++ and reuse the prepared base objective.
        if (base.is_coefficient) {
            JacobiGradientResult result;
            JacobiFilterState base_filter;
            if (!compute_filter(base, base_filter)) {
                return failed_with_diagnostics<JacobiGradientResult>(
                    Status::NumericalFailure, 14, failure_diagnostics);
            }
            result.value.objective =
                -base_filter.diagnostics.log_likelihood;
            result.value.diagnostics = base_filter.diagnostics;
            JacobiTransitionConfig derivative_config =
                impl_->config.transition;
            derivative_config.derivatives = false;
            derivative_config.numerical.gradient = false;
            const std::array<double, 3> values{
                params.kappa, params.m, params.xi};
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                double step = impl_->config.finite_difference_relative_step
                    * std::max(std::abs(values[parameter]), 1.0);
                if (parameter == 1) {
                    step = std::min(
                        {step, 0.49 * values[parameter],
                         0.49 * (1.0 - values[parameter])});
                } else {
                    step = std::min(step, 0.49 * values[parameter]);
                }
                if (!std::isfinite(step) || !(step > 0.0)) {
                    return failed_with_diagnostics<JacobiGradientResult>(
                        Status::InvalidParameter, 14, failure_diagnostics);
                }
                JacobiParams plus = params;
                JacobiParams minus = params;
                double* plus_values[3]{&plus.kappa, &plus.m, &plus.xi};
                double* minus_values[3]{&minus.kappa, &minus.m, &minus.xi};
                *plus_values[parameter] += step;
                *minus_values[parameter] -= step;
                const Result<EvaluatorSetup> upper = impl_->build_setup(
                    plus, derivative_config, base.generation);
                const Result<EvaluatorSetup> lower = impl_->build_setup(
                    minus, derivative_config, base.generation);
                JacobiFilterState upper_filter;
                JacobiFilterState lower_filter;
                if (!upper.is_ok() || !lower.is_ok()
                    || !compute_filter(upper.value, upper_filter)
                    || !compute_filter(lower.value, lower_filter)) {
                    return failed_with_diagnostics<JacobiGradientResult>(
                        !upper.is_ok() ? upper.status
                            : (!lower.is_ok() ? lower.status
                                             : Status::NumericalFailure),
                        14,
                        failure_diagnostics);
                }
                result.value.gradient[parameter] = -(
                    upper_filter.diagnostics.log_likelihood
                    - lower_filter.diagnostics.log_likelihood)
                    / (2.0 * step);
            }
            return result;
        }

        const std::size_t k = static_cast<std::size_t>(base.order);
        const std::size_t n = static_cast<std::size_t>(impl_->n_obs);
        std::vector<double> dweights(3 * k, 0.0);
        std::vector<double> dtransition(3 * k * k, 0.0);
        std::vector<double> demissions(3 * n * k, 0.0);

        if (fixed) {
            const JacobiFixedRuleResult rule = build_fixed_jacobi_rule(
                params,
                base.order,
                impl_->config.transition.numerical.memory_budget_bytes);
            if (!rule.is_ok() || rule.value.weight_derivatives.size() != 3 * k) {
                return failed_with_diagnostics<JacobiGradientResult>(
                    rule.is_ok() ? Status::NumericalFailure : rule.status,
                    15,
                    failure_diagnostics);
            }
            dweights = rule.value.weight_derivatives;
            dtransition = base.is_sparse
                ? derivative_dense_from_sparse(base.sparse)
                : base.dense.derivatives;
            if (dtransition.size() != 3 * k * k) {
                return failed_with_diagnostics<JacobiGradientResult>(
                    Status::InvalidSize, 15, failure_diagnostics);
            }
        } else {
            JacobiTransitionConfig derivative_config = impl_->config.transition;
            derivative_config.method = base.diagnostics.method_used;
            derivative_config.derivatives = false;
            derivative_config.numerical.gradient = false;
            const std::array<double, 3> values{
                params.kappa, params.m, params.xi};
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                double step = impl_->config.finite_difference_relative_step
                    * std::max(std::abs(values[parameter]), 1.0);
                if (parameter == 1) {
                    step = std::min(
                        {step, 0.49 * values[parameter],
                         0.49 * (1.0 - values[parameter])});
                } else {
                    step = std::min(step, 0.49 * values[parameter]);
                }
                if (!std::isfinite(step) || !(step > 0.0)) {
                    return failed_with_diagnostics<JacobiGradientResult>(
                        Status::InvalidParameter, 16, failure_diagnostics);
                }
                JacobiParams plus = params;
                JacobiParams minus = params;
                double* plus_values[3]{&plus.kappa, &plus.m, &plus.xi};
                double* minus_values[3]{&minus.kappa, &minus.m, &minus.xi};
                *plus_values[parameter] += step;
                *minus_values[parameter] -= step;
                const Result<EvaluatorSetup> upper = impl_->build_setup(
                    plus, derivative_config, base.generation);
                const Result<EvaluatorSetup> lower = impl_->build_setup(
                    minus, derivative_config, base.generation);
                if (!upper.is_ok() || !lower.is_ok()
                    || upper.value.order != base.order
                    || lower.value.order != base.order) {
                    return failed_with_diagnostics<JacobiGradientResult>(
                        !upper.is_ok() ? upper.status
                            : (!lower.is_ok() ? lower.status
                                             : Status::InvalidSize),
                        16,
                        failure_diagnostics);
                }
                const std::vector<double> upper_transition =
                    setup_dense_transition(upper.value);
                const std::vector<double> lower_transition =
                    setup_dense_transition(lower.value);
                const double inverse_denominator = 1.0 / (2.0 * step);
                for (std::size_t node = 0; node < k; ++node) {
                    dweights[parameter * k + node] =
                        (upper.value.weights[node] - lower.value.weights[node])
                        * inverse_denominator;
                }
                for (std::size_t index = 0; index < k * k; ++index) {
                    dtransition[parameter * k * k + index] =
                        (upper_transition[index] - lower_transition[index])
                        * inverse_denominator;
                }
                for (std::size_t index = 0; index < n * k; ++index) {
                    demissions[parameter * n * k + index] =
                        (upper.value.emissions[index] - lower.value.emissions[index])
                        * inverse_denominator;
                }
            }
        }

        JacobiGradientResult result;
        if (!filter_gradient(
                base, dweights, dtransition, demissions, result.value)) {
            return failed_with_diagnostics<JacobiGradientResult>(
                Status::NumericalFailure, 17, failure_diagnostics);
        }
        JacobiFilterState filter_state;
        if (!compute_filter(base, filter_state)) {
            return failed_with_diagnostics<JacobiGradientResult>(
                Status::NumericalFailure, 17, failure_diagnostics);
        }
        result.value.diagnostics = filter_state.diagnostics;
        return result;
    } catch (const std::bad_alloc&) {
        return failed_with_diagnostics<JacobiGradientResult>(
            Status::InvalidSize, 13, failure_diagnostics);
    }
}

JacobiEvaluatorVectorResult PreparedScarJacobiEvaluator::predictive_mean(
    const JacobiParams& params) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const JacobiFilterResult& filtered = impl_->filter_for(params);
    if (!filtered.is_ok()) {
        JacobiEvaluatorVectorResult result =
            failed<JacobiEvaluatorVectorResult>(filtered.status, 18);
        result.failure = filtered.failure;
        result.value.diagnostics = filtered.value.diagnostics;
        return result;
    }
    JacobiEvaluatorVectorResult result;
    const std::size_t n = static_cast<std::size_t>(filtered.value.n_obs);
    const std::size_t k = static_cast<std::size_t>(filtered.value.order);
    result.value.values.assign(n, 0.0);
    for (std::size_t observation = 0; observation < n; ++observation) {
        for (std::size_t node = 0; node < k; ++node) {
            result.value.values[observation] +=
                filtered.value.predicted[observation * k + node]
                * filtered.value.theta[node];
        }
    }
    result.value.diagnostics = filtered.value.diagnostics;
    return result;
}

JacobiEvaluatorPairResult PreparedScarJacobiEvaluator::mixture_h_pair(
    const JacobiParams& params) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const JacobiFilterResult& filtered = impl_->filter_for(params);
    if (!filtered.is_ok()) {
        JacobiEvaluatorPairResult result =
            failed<JacobiEvaluatorPairResult>(filtered.status, 19);
        result.failure = filtered.failure;
        result.value.diagnostics = filtered.value.diagnostics;
        return result;
    }
    JacobiEvaluatorPairResult result;
    const std::size_t n = static_cast<std::size_t>(filtered.value.n_obs);
    const std::size_t k = static_cast<std::size_t>(filtered.value.order);
    result.value.first.assign(n, 0.0);
    result.value.second.assign(n, 0.0);
    std::vector<double> first_row(k, 0.0);
    std::vector<double> second_row(k, 0.0);
    for (std::size_t observation = 0; observation < n; ++observation) {
        for (std::size_t node = 0; node < k; ++node) {
            double first_given_second = 0.0;
            double second_given_first = 0.0;
            impl_->pair.h_pair(
                impl_->observations[2 * observation],
                impl_->observations[2 * observation + 1],
                filtered.value.theta[node],
                first_given_second,
                second_given_first);
            first_row[node] = second_given_first;
            second_row[node] = first_given_second;
        }
        repair_and_clip_h(first_row);
        repair_and_clip_h(second_row);
        for (std::size_t node = 0; node < k; ++node) {
            const double probability =
                filtered.value.predicted[observation * k + node];
            result.value.first[observation] += probability * first_row[node];
            result.value.second[observation] += probability * second_row[node];
        }
        result.value.first[observation] = std::clamp(
            result.value.first[observation],
            numerical::kHFunctionEps,
            1.0 - numerical::kHFunctionEps);
        result.value.second[observation] = std::clamp(
            result.value.second[observation],
            numerical::kHFunctionEps,
            1.0 - numerical::kHFunctionEps);
    }
    result.value.diagnostics = filtered.value.diagnostics;
    return result;
}

JacobiEvaluatorVectorResult PreparedScarJacobiEvaluator::mixture_h(
    const JacobiParams& params) const {
    const JacobiEvaluatorPairResult pair = mixture_h_pair(params);
    if (!pair.is_ok()) {
        JacobiEvaluatorVectorResult result =
            failed<JacobiEvaluatorVectorResult>(pair.status, 19);
        result.failure = pair.failure;
        result.value.diagnostics = pair.value.diagnostics;
        return result;
    }
    JacobiEvaluatorVectorResult result;
    result.value.values = pair.value.first;
    result.value.diagnostics = pair.value.diagnostics;
    return result;
}

JacobiEvaluatorPairResult PreparedScarJacobiEvaluator::rosenblatt(
    const JacobiParams& params) const {
    JacobiEvaluatorPairResult result = mixture_h_pair(params);
    if (!result.is_ok()) {
        return result;
    }
    for (std::size_t observation = 0;
         observation < result.value.first.size(); ++observation) {
        result.value.first[observation] = impl_->observations[2 * observation];
    }
    return result;
}

JacobiEvaluatorPairResult PreparedScarJacobiEvaluator::gaussian_rosenblatt(
    const JacobiParams& params) const {
    JacobiEvaluatorPairResult result = rosenblatt(params);
    if (!result.is_ok()) {
        return result;
    }
    for (double& value : result.value.first) {
        value = math::normal_quantile(value);
    }
    for (double& value : result.value.second) {
        value = math::normal_quantile(value);
    }
    return result;
}

JacobiStateDistributionResult
PreparedScarJacobiEvaluator::state_distribution(
    const JacobiParams& params,
    JacobiStateHorizon horizon) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const JacobiFilterResult& filtered = impl_->filter_for(params);
    if (!filtered.is_ok()) {
        JacobiStateDistributionResult result =
            failed<JacobiStateDistributionResult>(filtered.status, 20);
        result.failure = filtered.failure;
        result.value.diagnostics = filtered.value.diagnostics;
        return result;
    }
    JacobiStateDistributionResult result;
    result.value.tau = filtered.value.tau;
    result.value.probability = horizon == JacobiStateHorizon::Next
        ? filtered.value.next_probability
        : filtered.value.current_probability;
    result.value.horizon = horizon;
    result.value.diagnostics = filtered.value.diagnostics;
    return result;
}

JacobiStateDistributionResult PreparedScarJacobiEvaluator::condition_state(
    const std::vector<double>& tau,
    const std::vector<double>& probability,
    const std::array<double, 2>& observation,
    JacobiStateHorizon horizon) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (tau.empty() || tau.size() != probability.size()
        || !std::isfinite(observation[0]) || !std::isfinite(observation[1])) {
        return failed<JacobiStateDistributionResult>(Status::InvalidSize, 21);
    }
    JacobiStateDistributionResult result;
    result.value.tau = tau;
    result.value.probability = probability;
    result.value.horizon = horizon;
    std::vector<double> log_weight(tau.size(), 0.0);
    double maximum = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (std::size_t node = 0; node < tau.size(); ++node) {
        if (!std::isfinite(tau[node]) || !std::isfinite(probability[node])
            || probability[node] < 0.0) {
            return failed<JacobiStateDistributionResult>(
                Status::InvalidParameter, 21);
        }
        double theta = impl_->pair.tau_to_parameter(tau[node]);
        if (!std::isnan(impl_->config.transition.numerical.theta_cap)) {
            theta = std::min(
                theta, impl_->config.transition.numerical.theta_cap);
        }
        log_weight[node] = impl_->pair.log_pdf(
            observation[0], observation[1], theta);
        if (std::isfinite(log_weight[node])) {
            maximum = std::max(maximum, log_weight[node]);
            any = true;
        }
    }
    if (!any) {
        return result;
    }
    for (std::size_t node = 0; node < tau.size(); ++node) {
        result.value.probability[node] = std::isfinite(log_weight[node])
            ? probability[node] * std::exp(log_weight[node] - maximum)
            : 0.0;
    }
    if (!normalize_probability(result.value.probability, 0.0)) {
        result.value.probability = probability;
    }
    return result;
}

std::uint64_t PreparedScarJacobiEvaluator::preparation_count() const noexcept {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->preparation_generation;
}

}  // namespace scar
