#include "scar/jacobi.hpp"

#include "scar/core/checked_arithmetic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

namespace scar {
namespace {

template <typename ResultType>
ResultType failure(Status status, int operation = -1) {
    ResultType result;
    result.status = status;
    result.failure.operation = operation;
    return result;
}

bool finite_positive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

double pi_value() noexcept {
    return std::acos(-1.0);
}

double resolve_dt(const JacobiTransitionConfig& config, Status& status) {
    const JacobiScalarResult resolved =
        jacobi_resolve_dt(config.numerical.n_obs);
    status = resolved.status;
    return resolved.value;
}

bool checked_elements_to_bytes(
    std::size_t elements,
    std::uint64_t& bytes) noexcept {
    return core::checked_byte_count<double>(elements, bytes);
}

bool add_product(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t factor,
    std::size_t& total) noexcept {
    std::size_t product = 0;
    std::size_t term = 0;
    return core::checked_size_mul(lhs, rhs, product)
        && core::checked_size_mul(product, factor, term)
        && core::checked_size_add(total, term, total);
}

bool sparse_storage_elements(
    const JacobiTransitionConfig& config,
    std::size_t& elements) noexcept {
    const std::size_t order =
        static_cast<std::size_t>(config.numerical.quad_order);
    const std::size_t width =
        2 * static_cast<std::size_t>(config.numerical.gh_order);
    elements = 0;
    if (config.method == JacobiTransitionMethod::LocalFixed) {
        const std::size_t width_arrays = config.derivatives ? 5 : 2;
        if (!add_product(order, width, width_arrays, elements)
            || !add_product(order, 15, 1, elements)) {
            return false;
        }
    } else {
        std::size_t width_arrays = 2;
        if (config.correction ==
                JacobiStationarityCorrection::MetropolisHastings) {
            width_arrays = 4;
        } else if (config.correction ==
                   JacobiStationarityCorrection::IpFp) {
            width_arrays = 5;
        }
        if (!add_product(order, 8, 1, elements)
            || !add_product(order, width, width_arrays, elements)) {
            return false;
        }
    }
    return true;
}

JacobiMemoryEstimate sparse_storage_estimate(
    const JacobiTransitionConfig& config,
    bool& valid) noexcept {
    valid = false;
    JacobiMemoryEstimate estimate;
    estimate.budget_bytes = config.numerical.memory_budget_bytes;
    if (!ok(validate_jacobi_transition_config(config))
        || config.storage != JacobiTransitionStorage::Sparse
        || !sparse_storage_elements(config, estimate.elements)
        || !checked_elements_to_bytes(estimate.elements, estimate.bytes)) {
        return estimate;
    }
    estimate.within_budget = estimate.bytes <= estimate.budget_bytes;
    valid = true;
    return estimate;
}

JacobiMemoryEstimate sparse_workspace_estimate(
    const JacobiTransitionConfig& config,
    bool& valid) noexcept {
    JacobiMemoryEstimate estimate = sparse_storage_estimate(config, valid);
    if (!valid) {
        return estimate;
    }
    valid = false;
    std::size_t elements = estimate.elements;
    // Include the quadrature/basis peak in the same budget instead of
    // allowing direct sparse entry points to allocate it out of band.
    JacobiNumericalConfig quadrature = config.numerical;
    quadrature.basis_order = 1;
    quadrature.n_obs = 0;
    quadrature.matrix = false;
    quadrature.gradient = false;
    quadrature.memory_budget_bytes =
        std::numeric_limits<std::uint64_t>::max();
    const JacobiMemoryResult quadrature_memory =
        estimate_jacobi_workspace(quadrature);
    if (!quadrature_memory.is_ok()
        || !core::checked_size_add(
            elements, quadrature_memory.value.elements, elements)
        || !checked_elements_to_bytes(elements, estimate.bytes)) {
        return estimate;
    }
    estimate.elements = elements;
    estimate.within_budget = estimate.bytes <= estimate.budget_bytes;
    valid = true;
    return estimate;
}

void seed_policy_diagnostics(
    JacobiTransitionDiagnostics& diagnostics,
    const JacobiTransitionConfig& config) noexcept {
    diagnostics.gh_order = config.numerical.gh_order;
    diagnostics.method_requested = config.method;
    diagnostics.method_used = config.method;
    diagnostics.storage = config.storage;
    diagnostics.correction = config.correction;
    diagnostics.memory_budget_bytes =
        config.numerical.memory_budget_bytes;
}

void seed_diagnostics(
    JacobiTransitionDiagnostics& diagnostics,
    const JacobiShapeResult& shape,
    const JacobiTransitionConfig& config,
    double dt) noexcept {
    seed_policy_diagnostics(diagnostics, config);
    diagnostics.dt = dt;
    diagnostics.alpha = shape.value.alpha;
    diagnostics.beta = shape.value.beta;
}

bool normalize_dense_rows(
    std::vector<double>& probabilities,
    int order,
    double& maximum_error) noexcept {
    maximum_error = 0.0;
    const std::size_t count = static_cast<std::size_t>(order);
    for (std::size_t row = 0; row < count; ++row) {
        const std::size_t offset = row * count;
        double sum = 0.0;
        for (std::size_t column = 0; column < count; ++column) {
            sum += probabilities[offset + column];
        }
        if (!finite_positive(sum)) {
            return false;
        }
        maximum_error = std::max(maximum_error, std::abs(sum - 1.0));
        for (std::size_t column = 0; column < count; ++column) {
            probabilities[offset + column] /= sum;
        }
    }
    return true;
}

double dense_stationary_error(
    const std::vector<double>& weights,
    const std::vector<double>& probabilities,
    int order) noexcept {
    const std::size_t count = static_cast<std::size_t>(order);
    double maximum = 0.0;
    for (std::size_t column = 0; column < count; ++column) {
        double propagated = 0.0;
        for (std::size_t row = 0; row < count; ++row) {
            propagated += weights[row]
                * probabilities[row * count + column];
        }
        maximum = std::max(
            maximum, std::abs(propagated - weights[column]));
    }
    return maximum;
}

double dense_minimum(const std::vector<double>& values) noexcept {
    return values.empty()
        ? 0.0
        : *std::min_element(values.begin(), values.end());
}

std::size_t sparse_offset(
    const JacobiSparseTransition& transition,
    std::size_t row,
    std::size_t slot) noexcept {
    return row * static_cast<std::size_t>(transition.max_width) + slot;
}

double sparse_reverse_probability(
    const JacobiSparseTransition& transition,
    std::size_t source,
    std::size_t target) noexcept {
    const std::size_t count = static_cast<std::size_t>(
        transition.counts[source]);
    std::size_t low = 0;
    std::size_t high = count;
    while (low < high) {
        const std::size_t middle = low + (high - low) / 2;
        const std::int64_t index = transition.indices[
            sparse_offset(transition, source, middle)];
        if (index < static_cast<std::int64_t>(target)) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < count
        && transition.indices[sparse_offset(transition, source, low)]
            == static_cast<std::int64_t>(target)) {
        return transition.probabilities[
            sparse_offset(transition, source, low)];
    }
    return 0.0;
}

JacobiSparseValidationCode validate_sparse_transition_detail(
    const JacobiSparseTransition& transition) noexcept {
    if (transition.order <= 0 || transition.max_width <= 0) {
        return JacobiSparseValidationCode::InvalidShape;
    }
    const std::size_t order = static_cast<std::size_t>(transition.order);
    const std::size_t width =
        static_cast<std::size_t>(transition.max_width);
    if (transition.indices.size() != order * width
        || transition.probabilities.size() != order * width
        || transition.counts.size() != order) {
        return JacobiSparseValidationCode::InvalidShape;
    }
    for (std::size_t row = 0; row < order; ++row) {
        const std::int64_t raw_count = transition.counts[row];
        if (raw_count < 1 || raw_count > transition.max_width) {
            return JacobiSparseValidationCode::InvalidCount;
        }
        const std::size_t count = static_cast<std::size_t>(raw_count);
        double sum = 0.0;
        std::int64_t previous = -1;
        for (std::size_t slot = 0; slot < count; ++slot) {
            const std::size_t offset = sparse_offset(transition, row, slot);
            const std::int64_t target = transition.indices[offset];
            const double probability = transition.probabilities[offset];
            if (target < 0
                || target >= static_cast<std::int64_t>(order)) {
                return JacobiSparseValidationCode::IndexOutOfRange;
            }
            if (target <= previous) {
                return JacobiSparseValidationCode::IndicesNotIncreasing;
            }
            if (!std::isfinite(probability) || probability < 0.0) {
                return JacobiSparseValidationCode::InvalidProbability;
            }
            previous = target;
            sum += probability;
        }
        if (std::abs(sum - 1.0) > 1e-14) {
            return JacobiSparseValidationCode::RowSum;
        }
    }
    return JacobiSparseValidationCode::Ok;
}

std::vector<double> sparse_left_multiply_unchecked(
    const JacobiSparseTransition& transition,
    const std::vector<double>& values) {
    const std::size_t order = static_cast<std::size_t>(transition.order);
    std::vector<double> output(order, 0.0);
    for (std::size_t row = 0; row < order; ++row) {
        const double source = values[row];
        const std::size_t count = static_cast<std::size_t>(
            transition.counts[row]);
        for (std::size_t slot = 0; slot < count; ++slot) {
            const std::size_t offset = sparse_offset(transition, row, slot);
            output[static_cast<std::size_t>(transition.indices[offset])]
                += source * transition.probabilities[offset];
        }
    }
    return output;
}

std::size_t insert_mass(
    JacobiSparseTransition& transition,
    std::size_t row,
    std::size_t count,
    std::int64_t target,
    double mass,
    const std::array<double, 3>* derivative) {
    for (std::size_t slot = 0; slot < count; ++slot) {
        const std::size_t offset = sparse_offset(transition, row, slot);
        if (transition.indices[offset] == target) {
            transition.probabilities[offset] += mass;
            if (derivative != nullptr) {
                for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                    transition.derivatives[
                        (parameter * static_cast<std::size_t>(transition.order)
                            + row)
                        * static_cast<std::size_t>(transition.max_width)
                        + slot] += (*derivative)[parameter];
                }
            }
            return count;
        }
    }
    const std::size_t offset = sparse_offset(transition, row, count);
    transition.indices[offset] = target;
    transition.probabilities[offset] = mass;
    if (derivative != nullptr) {
        for (std::size_t parameter = 0; parameter < 3; ++parameter) {
            transition.derivatives[
                (parameter * static_cast<std::size_t>(transition.order)
                    + row)
                * static_cast<std::size_t>(transition.max_width)
                + count] = (*derivative)[parameter];
        }
    }
    return count + 1;
}

void sort_sparse_row(
    JacobiSparseTransition& transition,
    std::size_t row,
    std::size_t count,
    bool derivatives) {
    const std::size_t order = static_cast<std::size_t>(transition.order);
    const std::size_t width = static_cast<std::size_t>(transition.max_width);
    for (std::size_t position = 1; position < count; ++position) {
        const std::size_t current = sparse_offset(transition, row, position);
        const std::int64_t index = transition.indices[current];
        const double probability = transition.probabilities[current];
        std::array<double, 3> derivative{};
        if (derivatives) {
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                derivative[parameter] = transition.derivatives[
                    (parameter * order + row) * width + position];
            }
        }
        std::size_t cursor = position;
        while (cursor > 0
               && transition.indices[
                   sparse_offset(transition, row, cursor - 1)] > index) {
            const std::size_t source =
                sparse_offset(transition, row, cursor - 1);
            const std::size_t target =
                sparse_offset(transition, row, cursor);
            transition.indices[target] = transition.indices[source];
            transition.probabilities[target] =
                transition.probabilities[source];
            if (derivatives) {
                for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                    transition.derivatives[
                        (parameter * order + row) * width + cursor] =
                        transition.derivatives[
                            (parameter * order + row) * width + cursor - 1];
                }
            }
            --cursor;
        }
        const std::size_t target = sparse_offset(transition, row, cursor);
        transition.indices[target] = index;
        transition.probabilities[target] = probability;
        if (derivatives) {
            for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                transition.derivatives[
                    (parameter * order + row) * width + cursor] =
                    derivative[parameter];
            }
        }
    }
}

JacobiSparseTransitionResult build_sparse_raw(
    const JacobiParams& params,
    const JacobiTransitionConfig& config,
    bool fixed_grid) {
    Status dt_status = Status::Ok;
    const double dt = resolve_dt(config, dt_status);
    if (!ok(dt_status)) {
        return failure<JacobiSparseTransitionResult>(dt_status);
    }
    const JacobiShapeResult shape = jacobi_stationary_shape(params);
    if (!shape.is_ok()) {
        return failure<JacobiSparseTransitionResult>(shape.status);
    }
    bool memory_valid = false;
    const JacobiMemoryEstimate memory =
        sparse_workspace_estimate(config, memory_valid);
    if (!memory_valid || !memory.within_budget) {
        JacobiSparseTransitionResult result =
            failure<JacobiSparseTransitionResult>(Status::InvalidSize, 1);
        seed_diagnostics(result.value.diagnostics, shape, config, dt);
        result.value.diagnostics.method_used = fixed_grid
            ? JacobiTransitionMethod::LocalFixed
            : JacobiTransitionMethod::Local;
        result.value.diagnostics.estimated_workspace_bytes = memory.bytes;
        result.value.diagnostics.memory_budget_bytes = memory.budget_bytes;
        return result;
    }

    const int order = config.numerical.quad_order;
    const int width = 2 * config.numerical.gh_order;
    JacobiFixedRuleResult fixed;
    JacobiBasisResult moving;
    if (fixed_grid) {
        fixed = build_fixed_jacobi_rule(
            params, order, config.numerical.memory_budget_bytes);
        if (!fixed.is_ok()) {
            return failure<JacobiSparseTransitionResult>(fixed.status, 1);
        }
    } else {
        moving = build_jacobi_rule(
            shape.value.alpha,
            shape.value.beta,
            order,
            1,
            config.numerical.memory_budget_bytes);
        if (!moving.is_ok()) {
            return failure<JacobiSparseTransitionResult>(moving.status, 1);
        }
    }
    JacobiQuadratureResult hermite = gauss_hermite_probability_rule(
        config.numerical.gh_order,
        config.numerical.memory_budget_bytes);
    if (!hermite.is_ok()) {
        return failure<JacobiSparseTransitionResult>(hermite.status, 1);
    }

    try {
        JacobiSparseTransitionResult result;
        JacobiSparseTransition& transition = result.value;
        transition.order = order;
        transition.max_width = width;
        const std::size_t count = static_cast<std::size_t>(order);
        const std::size_t row_width = static_cast<std::size_t>(width);
        transition.indices.assign(count * row_width, -1);
        transition.probabilities.assign(count * row_width, 0.0);
        transition.counts.assign(count, 0);
        if (fixed_grid && config.derivatives) {
            transition.derivatives.assign(3 * count * row_width, 0.0);
        }
        transition.diagnostics.estimated_workspace_bytes = memory.bytes;
        seed_diagnostics(
            transition.diagnostics, shape, config, dt);
        transition.diagnostics.method_used = fixed_grid
            ? JacobiTransitionMethod::LocalFixed
            : JacobiTransitionMethod::Local;

        const std::vector<double>& tau = fixed_grid
            ? fixed.value.tau : moving.value.tau;
        transition.tau = tau;
        transition.weights = fixed_grid
            ? fixed.value.weights : moving.value.weights;
        std::vector<double> y_grid(count, 0.0);
        std::vector<double> drift(count, 0.0);
        std::vector<std::array<double, 3>> dcenter(count);
        for (std::size_t row = 0; row < count; ++row) {
            const double value = tau[row];
            const double root = std::sqrt(value);
            const double angle = std::asin(root);
            const double denominator = std::sqrt(std::max(
                value * (1.0 - value), kJacobiDriftDenominatorFloor));
            y_grid[row] = 2.0 * angle / params.xi;
            drift[row] = (
                params.kappa * (params.m - value)
                    / (params.xi * denominator)
                - params.xi * (1.0 - 2.0 * value)
                    / (4.0 * denominator));
            if (fixed_grid && config.derivatives) {
                dcenter[row][0] =
                    (params.m - value) / (params.xi * denominator) * dt;
                dcenter[row][1] =
                    params.kappa / (params.xi * denominator) * dt;
                dcenter[row][2] =
                    -2.0 * angle / (params.xi * params.xi)
                    + (-params.kappa * (params.m - value)
                           / (params.xi * params.xi * denominator)
                       - (1.0 - 2.0 * value) / (4.0 * denominator)) * dt;
            }
        }
        const double offset_scale = std::sqrt(2.0 * dt);
        const double y_max = pi_value() / params.xi;
        const std::array<double, 3> zero_derivative{};
        for (std::size_t row = 0; row < count; ++row) {
            std::size_t active = 0;
            const double center = y_grid[row] + drift[row] * dt;
            for (std::size_t node = 0;
                 node < hermite.value.nodes.size(); ++node) {
                double y_next = center
                    + offset_scale * hermite.value.nodes[node];
                const double weight = hermite.value.weights[node];
                std::array<double, 3> dtau{};
                const std::array<double, 3>* derivative = nullptr;
                if (fixed_grid) {
                    if (y_next <= 0.0) {
                        active = insert_mass(
                            transition, row, active, 0, weight,
                            config.derivatives ? &zero_derivative : nullptr);
                        continue;
                    }
                    if (y_next >= y_max) {
                        active = insert_mass(
                            transition,
                            row,
                            active,
                            static_cast<std::int64_t>(count - 1),
                            weight,
                            config.derivatives ? &zero_derivative : nullptr);
                        continue;
                    }
                    if (config.derivatives) {
                        const double common =
                            0.5 * std::sin(params.xi * y_next);
                        dtau[0] = common * params.xi * dcenter[row][0];
                        dtau[1] = common * params.xi * dcenter[row][1];
                        dtau[2] = common
                            * (y_next + params.xi * dcenter[row][2]);
                        derivative = &dtau;
                    }
                } else {
                    y_next = std::clamp(y_next, 0.0, y_max);
                }
                const double tau_y = std::pow(
                    std::sin(0.5 * params.xi * y_next), 2.0);
                if (tau_y <= tau.front()) {
                    active = insert_mass(
                        transition, row, active, 0, weight,
                        config.derivatives ? &zero_derivative : nullptr);
                    continue;
                }
                if (tau_y >= tau.back()) {
                    active = insert_mass(
                        transition,
                        row,
                        active,
                        static_cast<std::int64_t>(count - 1),
                        weight,
                        config.derivatives ? &zero_derivative : nullptr);
                    continue;
                }
                const auto right_iterator = std::upper_bound(
                    tau.begin(), tau.end(), tau_y);
                const std::size_t right = static_cast<std::size_t>(
                    right_iterator - tau.begin());
                const std::size_t left = right - 1;
                const double interval = tau[right] - tau[left];
                if (!(interval > 0.0)) {
                    active = insert_mass(
                        transition,
                        row,
                        active,
                        static_cast<std::int64_t>(left),
                        weight,
                        config.derivatives ? &zero_derivative : nullptr);
                    continue;
                }
                const double fraction = (tau_y - tau[left]) / interval;
                std::array<double, 3> left_derivative{};
                std::array<double, 3> right_derivative{};
                if (derivative != nullptr) {
                    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                        right_derivative[parameter] =
                            weight * (*derivative)[parameter] / interval;
                        left_derivative[parameter] =
                            -right_derivative[parameter];
                    }
                }
                active = insert_mass(
                    transition,
                    row,
                    active,
                    static_cast<std::int64_t>(left),
                    weight * (1.0 - fraction),
                    derivative != nullptr ? &left_derivative : nullptr);
                active = insert_mass(
                    transition,
                    row,
                    active,
                    static_cast<std::int64_t>(right),
                    weight * fraction,
                    derivative != nullptr ? &right_derivative : nullptr);
            }
            sort_sparse_row(
                transition, row, active,
                fixed_grid && config.derivatives);
            double row_sum = 0.0;
            std::array<double, 3> derivative_sum{};
            for (std::size_t slot = 0; slot < active; ++slot) {
                row_sum += transition.probabilities[
                    sparse_offset(transition, row, slot)];
                if (fixed_grid && config.derivatives) {
                    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                        derivative_sum[parameter] += transition.derivatives[
                            (parameter * count + row) * row_width + slot];
                    }
                }
            }
            if (!finite_positive(row_sum)) {
                result.status = Status::NumericalFailure;
                result.failure.row = static_cast<std::int64_t>(row);
                return result;
            }
            for (std::size_t slot = 0; slot < active; ++slot) {
                const std::size_t offset = sparse_offset(transition, row, slot);
                const double raw_probability = transition.probabilities[offset];
                transition.probabilities[offset] = raw_probability / row_sum;
                if (fixed_grid && config.derivatives) {
                    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
                        const std::size_t derivative_offset =
                            (parameter * count + row) * row_width + slot;
                        transition.derivatives[derivative_offset] = (
                            transition.derivatives[derivative_offset] * row_sum
                            - raw_probability * derivative_sum[parameter]
                        ) / (row_sum * row_sum);
                    }
                }
            }
            transition.counts[row] = static_cast<std::int64_t>(active);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiSparseTransitionResult>(Status::InvalidSize);
    }
}

JacobiSparseTransitionResult metropolis_hastings_correction(
    JacobiSparseTransition proposal,
    const std::vector<double>& weights) {
    try {
        const std::size_t order = static_cast<std::size_t>(proposal.order);
        JacobiSparseTransitionResult result;
        JacobiSparseTransition& corrected = result.value;
        corrected.order = proposal.order;
        corrected.max_width = proposal.max_width + 1;
        corrected.tau = std::move(proposal.tau);
        corrected.weights = std::move(proposal.weights);
        const std::size_t width =
            static_cast<std::size_t>(corrected.max_width);
        corrected.indices.assign(order * width, -1);
        corrected.probabilities.assign(order * width, 0.0);
        corrected.counts.assign(order, 0);
        corrected.diagnostics = proposal.diagnostics;
        corrected.diagnostics.correction =
            JacobiStationarityCorrection::MetropolisHastings;

        double accepted_mass = 0.0;
        double proposed_off_diagonal = 0.0;
        std::size_t reverse_missing = 0;
        std::size_t off_diagonal_edges = 0;
        double minimum_acceptance = 1.0;
        bool has_acceptance = false;
        for (std::size_t row = 0; row < order; ++row) {
            std::vector<std::pair<std::int64_t, double>> entries;
            double row_proposed = 0.0;
            double row_accepted = 0.0;
            const std::size_t count = static_cast<std::size_t>(
                proposal.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t offset = sparse_offset(proposal, row, slot);
                const std::size_t target = static_cast<std::size_t>(
                    proposal.indices[offset]);
                const double probability = proposal.probabilities[offset];
                if (target == row) {
                    continue;
                }
                ++off_diagonal_edges;
                proposed_off_diagonal += weights[row] * probability;
                row_proposed += probability;
                const double reverse = sparse_reverse_probability(
                    proposal, target, row);
                double accepted = 0.0;
                if (!(reverse > 0.0)) {
                    ++reverse_missing;
                } else {
                    const double ratio = weights[target] * reverse
                        / (weights[row] * probability);
                    accepted = probability * std::min(1.0, ratio);
                }
                if (accepted > 0.0) {
                    entries.emplace_back(
                        static_cast<std::int64_t>(target), accepted);
                    accepted_mass += weights[row] * accepted;
                    row_accepted += accepted;
                }
            }
            if (row_proposed > 0.0) {
                minimum_acceptance = std::min(
                    minimum_acceptance, row_accepted / row_proposed);
                has_acceptance = true;
            }
            double off_sum = 0.0;
            for (const auto& entry : entries) {
                off_sum += entry.second;
            }
            entries.emplace_back(
                static_cast<std::int64_t>(row), 1.0 - off_sum);
            std::sort(entries.begin(), entries.end());
            corrected.counts[row] =
                static_cast<std::int64_t>(entries.size());
            for (std::size_t slot = 0; slot < entries.size(); ++slot) {
                const std::size_t offset = sparse_offset(
                    corrected, row, slot);
                corrected.indices[offset] = entries[slot].first;
                corrected.probabilities[offset] = entries[slot].second;
            }
        }

        double balance_error = 0.0;
        double mean_stay = 0.0;
        double max_stay = 0.0;
        for (std::size_t row = 0; row < order; ++row) {
            const double stay = sparse_reverse_probability(
                corrected, row, row);
            mean_stay += weights[row] * stay;
            max_stay = std::max(max_stay, stay);
            const std::size_t count = static_cast<std::size_t>(
                corrected.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t offset = sparse_offset(
                    corrected, row, slot);
                const std::size_t target = static_cast<std::size_t>(
                    corrected.indices[offset]);
                const double reverse = sparse_reverse_probability(
                    corrected, target, row);
                balance_error = std::max(
                    balance_error,
                    std::abs(weights[row] * corrected.probabilities[offset]
                        - weights[target] * reverse));
            }
        }
        JacobiTransitionDiagnostics& diagnostics = corrected.diagnostics;
        diagnostics.mean_accepted_off_diagonal_mass = accepted_mass;
        diagnostics.mean_proposed_off_diagonal_mass = proposed_off_diagonal;
        diagnostics.acceptance_mass_ratio = proposed_off_diagonal > 0.0
            ? accepted_mass / proposed_off_diagonal : 1.0;
        diagnostics.min_row_acceptance_ratio = has_acceptance
            ? minimum_acceptance : 1.0;
        diagnostics.mean_stay_probability = mean_stay;
        diagnostics.max_stay_probability = max_stay;
        diagnostics.reverse_missing_edge_fraction = off_diagonal_edges > 0
            ? static_cast<double>(reverse_missing)
                / static_cast<double>(off_diagonal_edges)
            : 0.0;
        diagnostics.detailed_balance_error = balance_error;
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiSparseTransitionResult>(Status::InvalidSize);
    }
}

JacobiSparseTransitionResult ipfp_correction(
    JacobiSparseTransition transition,
    const std::vector<double>& weights,
    double tolerance,
    int max_iterations) {
    try {
        const std::size_t order = static_cast<std::size_t>(transition.order);
        const std::size_t width =
            static_cast<std::size_t>(transition.max_width);
        std::vector<double> flux = transition.probabilities;
        std::vector<double> proposal_flux = transition.probabilities;
        for (std::size_t row = 0; row < order; ++row) {
            const std::size_t count = static_cast<std::size_t>(
                transition.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t offset = row * width + slot;
                flux[offset] *= weights[row];
                proposal_flux[offset] *= weights[row];
            }
        }
        bool converged = false;
        double residual = std::numeric_limits<double>::infinity();
        int iterations = 0;
        std::vector<double> column_sums(order, 0.0);
        std::vector<double> row_sums(order, 0.0);
        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            std::fill(column_sums.begin(), column_sums.end(), 0.0);
            for (std::size_t row = 0; row < order; ++row) {
                const std::size_t count = static_cast<std::size_t>(
                    transition.counts[row]);
                for (std::size_t slot = 0; slot < count; ++slot) {
                    const std::size_t offset = row * width + slot;
                    column_sums[static_cast<std::size_t>(
                        transition.indices[offset])] += flux[offset];
                }
            }
            for (double sum : column_sums) {
                if (!finite_positive(sum)) {
                    return failure<JacobiSparseTransitionResult>(
                        Status::NumericalFailure, 3);
                }
            }
            for (std::size_t row = 0; row < order; ++row) {
                const std::size_t count = static_cast<std::size_t>(
                    transition.counts[row]);
                for (std::size_t slot = 0; slot < count; ++slot) {
                    const std::size_t offset = row * width + slot;
                    const std::size_t target = static_cast<std::size_t>(
                        transition.indices[offset]);
                    flux[offset] *= weights[target] / column_sums[target];
                }
            }
            for (std::size_t row = 0; row < order; ++row) {
                double sum = 0.0;
                const std::size_t count = static_cast<std::size_t>(
                    transition.counts[row]);
                for (std::size_t slot = 0; slot < count; ++slot) {
                    sum += flux[row * width + slot];
                }
                if (!finite_positive(sum)) {
                    return failure<JacobiSparseTransitionResult>(
                        Status::NumericalFailure, 4);
                }
                row_sums[row] = sum;
                for (std::size_t slot = 0; slot < count; ++slot) {
                    flux[row * width + slot] *= weights[row] / sum;
                }
            }
            std::fill(column_sums.begin(), column_sums.end(), 0.0);
            for (std::size_t row = 0; row < order; ++row) {
                const std::size_t count = static_cast<std::size_t>(
                    transition.counts[row]);
                for (std::size_t slot = 0; slot < count; ++slot) {
                    const std::size_t offset = row * width + slot;
                    column_sums[static_cast<std::size_t>(
                        transition.indices[offset])] += flux[offset];
                }
            }
            residual = 0.0;
            for (std::size_t column = 0; column < order; ++column) {
                residual = std::max(
                    residual, std::abs(column_sums[column] - weights[column]));
            }
            iterations = iteration;
            if (residual <= tolerance) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            return failure<JacobiSparseTransitionResult>(
                Status::NumericalFailure, 5);
        }

        double kl_divergence = 0.0;
        double maximum_change = 0.0;
        for (std::size_t row = 0; row < order; ++row) {
            const std::size_t count = static_cast<std::size_t>(
                transition.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t offset = row * width + slot;
                if (proposal_flux[offset] > 0.0) {
                    kl_divergence += flux[offset]
                        * std::log(flux[offset] / proposal_flux[offset]);
                }
                const double probability = flux[offset] / weights[row];
                maximum_change = std::max(
                    maximum_change,
                    std::abs(probability - transition.probabilities[offset]));
                transition.probabilities[offset] = probability;
            }
        }
        double mean_stay = 0.0;
        double max_stay = 0.0;
        for (std::size_t row = 0; row < order; ++row) {
            const double stay = sparse_reverse_probability(
                transition, row, row);
            mean_stay += weights[row] * stay;
            max_stay = std::max(max_stay, stay);
        }
        transition.diagnostics.correction =
            JacobiStationarityCorrection::IpFp;
        transition.diagnostics.ipfp_iterations = iterations;
        transition.diagnostics.ipfp_stationary_residual = residual;
        transition.diagnostics.ipfp_kl_divergence = kl_divergence;
        transition.diagnostics.ipfp_max_probability_change = maximum_change;
        transition.diagnostics.mean_stay_probability = mean_stay;
        transition.diagnostics.max_stay_probability = max_stay;
        JacobiSparseTransitionResult result;
        result.value = std::move(transition);
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiSparseTransitionResult>(Status::InvalidSize);
    }
}

void finish_sparse_diagnostics(
    JacobiSparseTransition& transition,
    const std::vector<double>& weights,
    bool derivatives) {
    const std::size_t order = static_cast<std::size_t>(transition.order);
    const std::size_t width =
        static_cast<std::size_t>(transition.max_width);
    std::size_t nnz = 0;
    double row_error = 0.0;
    for (std::size_t row = 0; row < order; ++row) {
        const std::size_t count = static_cast<std::size_t>(
            transition.counts[row]);
        nnz += count;
        double sum = 0.0;
        for (std::size_t slot = 0; slot < count; ++slot) {
            sum += transition.probabilities[row * width + slot];
        }
        row_error = std::max(row_error, std::abs(sum - 1.0));
    }
    const std::vector<double> propagated =
        sparse_left_multiply_unchecked(transition, weights);
    double stationary_error = 0.0;
    for (std::size_t index = 0; index < order; ++index) {
        stationary_error = std::max(
            stationary_error,
            std::abs(propagated[index] - weights[index]));
    }
    JacobiTransitionDiagnostics& diagnostics = transition.diagnostics;
    diagnostics.nnz = nnz;
    diagnostics.max_width = transition.max_width;
    diagnostics.max_row_sum_error = row_error;
    diagnostics.stationary_error = stationary_error;
    std::uint64_t retained = 0;
    const std::uint64_t index_bytes = static_cast<std::uint64_t>(
        transition.indices.size() * sizeof(std::int64_t));
    const std::uint64_t probability_bytes = static_cast<std::uint64_t>(
        transition.probabilities.size() * sizeof(double));
    const std::uint64_t count_bytes = static_cast<std::uint64_t>(
        transition.counts.size() * sizeof(std::int64_t));
    const std::uint64_t derivative_bytes = derivatives
        ? static_cast<std::uint64_t>(
            transition.derivatives.size() * sizeof(double))
        : 0;
    retained = index_bytes + probability_bytes + count_bytes
        + derivative_bytes;
    diagnostics.retained_bytes = retained;
    const std::uint64_t dense_factor = derivatives ? 4 : 1;
    diagnostics.dense_bytes = static_cast<std::uint64_t>(order)
        * static_cast<std::uint64_t>(order) * sizeof(double) * dense_factor;
}

JacobiDenseTransitionResult dense_from_sparse(
    const std::vector<double>& tau,
    const std::vector<double>& weights,
    JacobiSparseTransitionResult sparse) {
    if (!sparse.is_ok()) {
        JacobiDenseTransitionResult result =
            failure<JacobiDenseTransitionResult>(
                sparse.status, sparse.failure.operation);
        result.failure = sparse.failure;
        result.value.diagnostics = sparse.value.diagnostics;
        return result;
    }
    try {
        JacobiDenseTransitionResult result;
        const std::size_t order = static_cast<std::size_t>(
            sparse.value.order);
        const std::size_t width = static_cast<std::size_t>(
            sparse.value.max_width);
        result.value.order = sparse.value.order;
        result.value.tau = tau;
        result.value.weights = weights;
        result.value.probabilities.assign(order * order, 0.0);
        if (!sparse.value.derivatives.empty()) {
            result.value.derivatives.assign(3 * order * order, 0.0);
        }
        for (std::size_t row = 0; row < order; ++row) {
            const std::size_t count = static_cast<std::size_t>(
                sparse.value.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t sparse_index = row * width + slot;
                const std::size_t column = static_cast<std::size_t>(
                    sparse.value.indices[sparse_index]);
                result.value.probabilities[row * order + column] =
                    sparse.value.probabilities[sparse_index];
                for (std::size_t parameter = 0;
                     parameter < 3 && !result.value.derivatives.empty();
                     ++parameter) {
                    result.value.derivatives[
                        (parameter * order + row) * order + column] =
                        sparse.value.derivatives[
                            (parameter * order + row) * width + slot];
                }
            }
        }
        result.value.diagnostics = sparse.value.diagnostics;
        result.value.diagnostics.storage = JacobiTransitionStorage::Dense;
        result.value.diagnostics.min_entry =
            dense_minimum(result.value.probabilities);
        double row_error = 0.0;
        for (std::size_t row = 0; row < order; ++row) {
            double row_sum = 0.0;
            for (std::size_t column = 0; column < order; ++column) {
                row_sum += result.value.probabilities[row * order + column];
            }
            row_error = std::max(row_error, std::abs(row_sum - 1.0));
        }
        result.value.diagnostics.max_row_sum_error = row_error;
        result.value.diagnostics.stationary_error = dense_stationary_error(
            result.value.weights,
            result.value.probabilities,
            result.value.order);
        result.value.diagnostics.dense_bytes = static_cast<std::uint64_t>(
            (result.value.probabilities.size()
                + result.value.derivatives.size()) * sizeof(double));
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiDenseTransitionResult>(Status::InvalidSize);
    }
}

}  // namespace

JacobiSparseValidationCode validate_jacobi_sparse_transition(
    const JacobiSparseTransition& transition) noexcept {
    return validate_sparse_transition_detail(transition);
}

Status validate_jacobi_transition_config(
    const JacobiTransitionConfig& config) noexcept {
    if (!ok(validate_jacobi_config(config.numerical))) {
        return Status::InvalidSize;
    }
    if (config.numerical.n_obs < 2) {
        return Status::InvalidSize;
    }
    if (!std::isfinite(config.negative_mass_tolerance)
        || config.negative_mass_tolerance < 0.0
        || !finite_positive(config.ipfp_tolerance)
        || config.ipfp_max_iterations <= 0) {
        return Status::InvalidParameter;
    }
    if (config.method < JacobiTransitionMethod::Auto
        || config.method > JacobiTransitionMethod::SpectralCoeff
        || config.storage < JacobiTransitionStorage::Dense
        || config.storage > JacobiTransitionStorage::Sparse
        || config.correction < JacobiStationarityCorrection::None
        || config.correction > JacobiStationarityCorrection::IpFp) {
        return Status::InvalidParameter;
    }
    if (config.storage == JacobiTransitionStorage::Sparse
        && config.method != JacobiTransitionMethod::Local
        && config.method != JacobiTransitionMethod::LocalFixed) {
        return Status::InvalidParameter;
    }
    if (config.correction != JacobiStationarityCorrection::None
        && (config.storage != JacobiTransitionStorage::Sparse
            || config.method != JacobiTransitionMethod::Local)) {
        return Status::InvalidParameter;
    }
    if (config.derivatives
        && config.method != JacobiTransitionMethod::LocalFixed) {
        return Status::InvalidParameter;
    }
    return Status::Ok;
}

JacobiMemoryResult estimate_jacobi_sparse_workspace(
    const JacobiTransitionConfig& config) noexcept {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiMemoryResult>(config_status);
    }
    if (config.storage != JacobiTransitionStorage::Sparse) {
        return failure<JacobiMemoryResult>(Status::InvalidParameter);
    }
    bool valid = false;
    const JacobiMemoryEstimate estimate =
        sparse_workspace_estimate(config, valid);
    if (!valid) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    JacobiMemoryResult result;
    result.value = estimate;
    if (!result.value.within_budget) {
        result.status = Status::InvalidSize;
    }
    return result;
}

JacobiMemoryResult estimate_jacobi_sparse_storage(
    const JacobiTransitionConfig& config) noexcept {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiMemoryResult>(config_status);
    }
    if (config.storage != JacobiTransitionStorage::Sparse) {
        return failure<JacobiMemoryResult>(Status::InvalidParameter);
    }
    bool valid = false;
    const JacobiMemoryEstimate estimate =
        sparse_storage_estimate(config, valid);
    if (!valid) {
        return failure<JacobiMemoryResult>(Status::InvalidSize);
    }
    JacobiMemoryResult result;
    result.value = estimate;
    if (!result.value.within_budget) {
        result.status = Status::InvalidSize;
    }
    return result;
}

JacobiIntResult default_jacobi_quad_order(int basis_order) noexcept {
    if (basis_order <= 0 || basis_order > kMaxJacobiOrder) {
        return failure<JacobiIntResult>(Status::InvalidSize);
    }
    const int order = std::max(2 * basis_order + 16, 48);
    if (order > kMaxJacobiOrder) {
        return failure<JacobiIntResult>(Status::InvalidSize);
    }
    JacobiIntResult result;
    result.value = order;
    return result;
}

JacobiIntResult resolve_jacobi_basis_order(
    int requested_basis_order,
    int quad_order) noexcept {

    if (requested_basis_order <= 0
        || quad_order <= 0
        || quad_order > kMaxJacobiOrder) {
        return failure<JacobiIntResult>(Status::InvalidSize);
    }
    JacobiIntResult result;
    result.value = std::min(requested_basis_order, quad_order);
    return result;
}

JacobiIntResult jacobi_horizon_steps(std::int64_t n_obs) noexcept {
    if (n_obs < 2
        || n_obs - 1 > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())) {
        return failure<JacobiIntResult>(Status::InvalidSize);
    }
    JacobiIntResult result;
    result.value = static_cast<int>(n_obs - 1);
    return result;
}

JacobiVectorResult jacobi_transition_powers(
    const JacobiParams& params,
    std::int64_t n_obs,
    int basis_order) {
    const JacobiScalarResult resolved = jacobi_resolve_dt(n_obs);
    if (!ok(validate_jacobi_params(
            params, std::numeric_limits<double>::infinity()))
        || !resolved.is_ok()
        || basis_order <= 0
        || basis_order > kMaxJacobiOrder) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        const double dt = resolved.value;
        JacobiVectorResult result;
        result.value.resize(static_cast<std::size_t>(basis_order));
        for (int degree = 0; degree < basis_order; ++degree) {
            const double n = static_cast<double>(degree);
            const double eigenvalue = n * params.kappa
                + 0.5 * params.xi * params.xi * n * (n - 1.0);
            result.value[static_cast<std::size_t>(degree)] =
                std::exp(-eigenvalue * dt);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiCoefficientTransitionResult build_jacobi_coefficient_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config) {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiCoefficientTransitionResult>(config_status);
    }
    if (config.method != JacobiTransitionMethod::SpectralCoeff
        || config.storage != JacobiTransitionStorage::Dense
        || config.correction != JacobiStationarityCorrection::None
        || config.derivatives) {
        return failure<JacobiCoefficientTransitionResult>(
            Status::InvalidParameter);
    }
    Status dt_status = Status::Ok;
    const double dt = resolve_dt(config, dt_status);
    if (!ok(dt_status)) {
        return failure<JacobiCoefficientTransitionResult>(dt_status);
    }
    const JacobiShapeResult shape = jacobi_stationary_shape(params);
    if (!shape.is_ok()) {
        return failure<JacobiCoefficientTransitionResult>(shape.status);
    }
    JacobiNumericalConfig numerical = config.numerical;
    numerical.n_obs = 0;
    numerical.matrix = false;
    numerical.gradient = false;
    const JacobiMemoryResult memory = estimate_jacobi_workspace(numerical);
    if (!memory.is_ok()) {
        JacobiCoefficientTransitionResult result =
            failure<JacobiCoefficientTransitionResult>(
                Status::InvalidSize, 1);
        seed_diagnostics(result.value.diagnostics, shape, config, dt);
        result.value.diagnostics.method_used =
            JacobiTransitionMethod::SpectralCoeff;
        result.value.diagnostics.estimated_workspace_bytes =
            memory.value.bytes;
        return result;
    }
    JacobiBasisResult rule = build_jacobi_rule(
        shape.value.alpha,
        shape.value.beta,
        numerical.quad_order,
        numerical.basis_order,
        numerical.memory_budget_bytes);
    if (!rule.is_ok()) {
        return failure<JacobiCoefficientTransitionResult>(
            rule.status, 1);
    }
    JacobiVectorResult powers = jacobi_transition_powers(
        params, config.numerical.n_obs, numerical.basis_order);
    if (!powers.is_ok()) {
        return failure<JacobiCoefficientTransitionResult>(powers.status);
    }
    try {
        JacobiCoefficientTransitionResult result;
        JacobiCoefficientTransition& transition = result.value;
        transition.quad_order = numerical.quad_order;
        transition.basis_order = numerical.basis_order;
        transition.tau = std::move(rule.value.tau);
        transition.weights = std::move(rule.value.weights);
        transition.basis = std::move(rule.value.basis);
        transition.spectral_powers = std::move(powers.value);
        seed_diagnostics(transition.diagnostics, shape, config, dt);
        transition.diagnostics.method_used =
            JacobiTransitionMethod::SpectralCoeff;
        transition.diagnostics.estimated_workspace_bytes =
            memory.value.bytes;
        transition.diagnostics.retained_bytes =
            static_cast<std::uint64_t>(
                transition.tau.size()
                + transition.weights.size()
                + transition.basis.size()
                + transition.spectral_powers.size()) * sizeof(double);
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiCoefficientTransitionResult>(
            Status::InvalidSize);
    }
}

JacobiVectorResult apply_jacobi_coefficient_transition(
    const JacobiCoefficientTransition& transition,
    const std::vector<double>& coefficients) {
    const std::size_t basis_order = transition.basis_order > 0
        ? static_cast<std::size_t>(transition.basis_order) : 0;
    if (basis_order == 0
        || transition.spectral_powers.size() != basis_order
        || coefficients.size() != basis_order
        || !std::all_of(
            transition.spectral_powers.begin(),
            transition.spectral_powers.end(),
            [](double value) { return std::isfinite(value); })
        || !std::all_of(
            coefficients.begin(), coefficients.end(),
            [](double value) { return std::isfinite(value); })) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        JacobiVectorResult result;
        result.value.resize(basis_order);
        for (std::size_t index = 0; index < basis_order; ++index) {
            result.value[index] =
                transition.spectral_powers[index] * coefficients[index];
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiDenseTransitionResult build_jacobi_spectral_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config) {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiDenseTransitionResult>(config_status);
    }
    Status dt_status = Status::Ok;
    const double dt = resolve_dt(config, dt_status);
    if (!ok(dt_status)) {
        return failure<JacobiDenseTransitionResult>(dt_status);
    }
    const JacobiShapeResult shape = jacobi_stationary_shape(params);
    if (!shape.is_ok()) {
        return failure<JacobiDenseTransitionResult>(shape.status);
    }
    JacobiNumericalConfig numerical = config.numerical;
    numerical.n_obs = 0;
    numerical.matrix = true;
    numerical.gradient = false;
    const JacobiMemoryResult memory = estimate_jacobi_workspace(numerical);
    if (!memory.is_ok()) {
        JacobiDenseTransitionResult result =
            failure<JacobiDenseTransitionResult>(Status::InvalidSize, 1);
        seed_diagnostics(result.value.diagnostics, shape, config, dt);
        result.value.diagnostics.method_used =
            JacobiTransitionMethod::SpectralMatrix;
        result.value.diagnostics.estimated_workspace_bytes = memory.value.bytes;
        result.value.diagnostics.memory_budget_bytes =
            numerical.memory_budget_bytes;
        return result;
    }
    JacobiBasisResult rule = build_jacobi_rule(
        shape.value.alpha,
        shape.value.beta,
        numerical.quad_order,
        numerical.basis_order,
        numerical.memory_budget_bytes);
    if (!rule.is_ok()) {
        return failure<JacobiDenseTransitionResult>(rule.status, 1);
    }
    JacobiVectorResult powers = jacobi_transition_powers(
        params, config.numerical.n_obs, numerical.basis_order);
    if (!powers.is_ok()) {
        return failure<JacobiDenseTransitionResult>(powers.status);
    }
    try {
        JacobiDenseTransitionResult result;
        JacobiDenseTransition& transition = result.value;
        transition.order = numerical.quad_order;
        transition.tau = std::move(rule.value.tau);
        transition.weights = std::move(rule.value.weights);
        transition.spectral_powers = std::move(powers.value);
        const std::size_t order = static_cast<std::size_t>(transition.order);
        const std::size_t basis_order = static_cast<std::size_t>(
            numerical.basis_order);
        transition.probabilities.assign(order * order, 0.0);
        seed_diagnostics(transition.diagnostics, shape, config, dt);
        transition.diagnostics.method_used =
            JacobiTransitionMethod::SpectralMatrix;
        transition.diagnostics.estimated_workspace_bytes = memory.value.bytes;
        transition.diagnostics.dense_bytes = static_cast<std::uint64_t>(
            transition.probabilities.size() * sizeof(double));
        double minimum = std::numeric_limits<double>::infinity();
        double negative_mass = 0.0;
        for (std::size_t row = 0; row < order; ++row) {
            for (std::size_t column = 0; column < order; ++column) {
                double kernel = 0.0;
                for (std::size_t degree = 0;
                     degree < basis_order; ++degree) {
                    kernel += rule.value.basis[row * basis_order + degree]
                        * transition.spectral_powers[degree]
                        * rule.value.basis[column * basis_order + degree];
                }
                double probability = kernel * transition.weights[column];
                minimum = std::min(minimum, probability);
                if (probability < 0.0) {
                    negative_mass -= probability;
                }
                if (config.clip_negative && probability < 0.0) {
                    probability = 0.0;
                }
                transition.probabilities[row * order + column] = probability;
            }
        }
        transition.diagnostics.raw_min_entry = minimum;
        transition.diagnostics.raw_negative_mass = negative_mass;
        transition.diagnostics.clipped_negative = config.clip_negative;
        if (!normalize_dense_rows(
                transition.probabilities,
                transition.order,
                transition.diagnostics.
                    max_row_sum_error_before_normalization)) {
            result.status = Status::NumericalFailure;
            result.failure.operation = 2;
            return result;
        }
        transition.diagnostics.min_entry =
            dense_minimum(transition.probabilities);
        transition.diagnostics.stationary_error = dense_stationary_error(
            transition.weights, transition.probabilities, transition.order);
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiDenseTransitionResult>(Status::InvalidSize);
    }
}

JacobiDenseTransitionResult build_jacobi_local_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config) {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiDenseTransitionResult>(config_status);
    }
    JacobiNumericalConfig numerical = config.numerical;
    numerical.n_obs = 0;
    numerical.matrix = true;
    numerical.gradient = false;
    const JacobiMemoryResult memory = estimate_jacobi_workspace(numerical);
    if (!memory.is_ok()) {
        JacobiDenseTransitionResult result =
            failure<JacobiDenseTransitionResult>(Status::InvalidSize, 1);
        seed_policy_diagnostics(result.value.diagnostics, config);
        result.value.diagnostics.method_used = JacobiTransitionMethod::Local;
        result.value.diagnostics.estimated_workspace_bytes = memory.value.bytes;
        result.value.diagnostics.memory_budget_bytes =
            numerical.memory_budget_bytes;
        return result;
    }
    JacobiTransitionConfig sparse_config = config;
    sparse_config.method = JacobiTransitionMethod::Local;
    sparse_config.storage = JacobiTransitionStorage::Sparse;
    sparse_config.correction = JacobiStationarityCorrection::None;
    sparse_config.derivatives = false;
    JacobiSparseTransitionResult sparse = build_sparse_raw(
        params, sparse_config, false);
    if (!sparse.is_ok()) {
        return dense_from_sparse({}, {}, std::move(sparse));
    }
    const std::vector<double> tau = sparse.value.tau;
    const std::vector<double> weights = sparse.value.weights;
    JacobiDenseTransitionResult result = dense_from_sparse(
        tau, weights, std::move(sparse));
    if (result.is_ok()) {
        result.value.diagnostics.method_requested = config.method;
        result.value.diagnostics.storage = JacobiTransitionStorage::Dense;
        result.value.diagnostics.estimated_workspace_bytes = memory.value.bytes;
    }
    return result;
}

JacobiDenseTransitionResult build_jacobi_fixed_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config) {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiDenseTransitionResult>(config_status);
    }
    JacobiNumericalConfig numerical = config.numerical;
    numerical.n_obs = 0;
    numerical.matrix = true;
    numerical.gradient = config.derivatives;
    const JacobiMemoryResult memory = estimate_jacobi_workspace(numerical);
    if (!memory.is_ok()) {
        JacobiDenseTransitionResult result =
            failure<JacobiDenseTransitionResult>(Status::InvalidSize, 1);
        seed_policy_diagnostics(result.value.diagnostics, config);
        result.value.diagnostics.method_used =
            JacobiTransitionMethod::LocalFixed;
        result.value.diagnostics.estimated_workspace_bytes = memory.value.bytes;
        result.value.diagnostics.memory_budget_bytes =
            numerical.memory_budget_bytes;
        return result;
    }
    JacobiTransitionConfig sparse_config = config;
    sparse_config.method = JacobiTransitionMethod::LocalFixed;
    sparse_config.storage = JacobiTransitionStorage::Sparse;
    sparse_config.correction = JacobiStationarityCorrection::None;
    JacobiSparseTransitionResult sparse = build_sparse_raw(
        params, sparse_config, true);
    if (!sparse.is_ok()) {
        return dense_from_sparse({}, {}, std::move(sparse));
    }
    const std::vector<double> tau = sparse.value.tau;
    const std::vector<double> weights = sparse.value.weights;
    JacobiDenseTransitionResult result = dense_from_sparse(
        tau, weights, std::move(sparse));
    if (result.is_ok()) {
        result.value.diagnostics.method_requested = config.method;
        result.value.diagnostics.storage = JacobiTransitionStorage::Dense;
        result.value.diagnostics.estimated_workspace_bytes = memory.value.bytes;
    }
    return result;
}

JacobiDenseTransitionResult build_jacobi_dense_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config) {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiDenseTransitionResult>(config_status);
    }
    if (config.storage != JacobiTransitionStorage::Dense) {
        return failure<JacobiDenseTransitionResult>(Status::InvalidParameter);
    }
    if (config.method == JacobiTransitionMethod::Local) {
        return build_jacobi_local_transition(params, config);
    }
    if (config.method == JacobiTransitionMethod::LocalFixed) {
        return build_jacobi_fixed_transition(params, config);
    }
    if (config.method == JacobiTransitionMethod::SpectralCoeff) {
        return failure<JacobiDenseTransitionResult>(Status::InvalidParameter);
    }

    JacobiDenseTransitionResult spectral =
        build_jacobi_spectral_transition(params, config);
    const bool spectral_ok = spectral.is_ok();
    const bool bad_negative = spectral_ok && (
        spectral.value.diagnostics.raw_min_entry
            < -config.negative_mass_tolerance
        || spectral.value.diagnostics.raw_negative_mass
            > config.negative_mass_tolerance);
    if (config.method == JacobiTransitionMethod::SpectralMatrix) {
        if (!spectral_ok) {
            return spectral;
        }
        if (bad_negative && !config.clip_negative) {
            spectral.status = Status::NumericalFailure;
            spectral.failure.operation = 6;
            return spectral;
        }
    }
    if (spectral_ok
        && config.method == JacobiTransitionMethod::Auto
        && !bad_negative) {
        // Convert immaterial signed roundoff into a probability transition.
        JacobiTransitionDiagnostics& diagnostics = spectral.value.diagnostics;
        diagnostics.method_requested = config.method;
        diagnostics.probability_cleanup_negative_mass = 0.0;
        diagnostics.probability_min_entry_before_cleanup =
            dense_minimum(spectral.value.probabilities);
        for (double value : spectral.value.probabilities) {
            if (value < 0.0) {
                diagnostics.probability_cleanup_negative_mass -= value;
            }
        }
        diagnostics.probability_cleanup_applied =
            diagnostics.probability_cleanup_negative_mass > 0.0;
        if (diagnostics.probability_cleanup_applied) {
            for (double& value : spectral.value.probabilities) {
                value = std::max(value, 0.0);
            }
            double ignored = 0.0;
            if (!normalize_dense_rows(
                    spectral.value.probabilities,
                    spectral.value.order,
                    ignored)) {
                spectral.status = Status::NumericalFailure;
                spectral.failure.operation = 2;
                return spectral;
            }
            diagnostics.stationary_error = dense_stationary_error(
                spectral.value.weights,
                spectral.value.probabilities,
                spectral.value.order);
        }
        diagnostics.min_entry = dense_minimum(spectral.value.probabilities);
        return spectral;
    }
    if (spectral_ok
        && config.method == JacobiTransitionMethod::SpectralMatrix) {
        JacobiTransitionDiagnostics& diagnostics = spectral.value.diagnostics;
        diagnostics.probability_cleanup_negative_mass = 0.0;
        diagnostics.probability_min_entry_before_cleanup =
            dense_minimum(spectral.value.probabilities);
        for (double value : spectral.value.probabilities) {
            if (value < 0.0) {
                diagnostics.probability_cleanup_negative_mass -= value;
            }
        }
        diagnostics.probability_cleanup_applied =
            diagnostics.probability_cleanup_negative_mass > 0.0;
        if (diagnostics.probability_cleanup_applied) {
            for (double& value : spectral.value.probabilities) {
                value = std::max(value, 0.0);
            }
            double ignored = 0.0;
            if (!normalize_dense_rows(
                    spectral.value.probabilities,
                    spectral.value.order,
                    ignored)) {
                spectral.status = Status::NumericalFailure;
                spectral.failure.operation = 2;
                return spectral;
            }
        }
        diagnostics.stationary_error = dense_stationary_error(
            spectral.value.weights,
            spectral.value.probabilities,
            spectral.value.order);
        diagnostics.min_entry = dense_minimum(spectral.value.probabilities);
        return spectral;
    }

    JacobiTransitionConfig local_config = config;
    local_config.method = JacobiTransitionMethod::Local;
    JacobiDenseTransitionResult local =
        build_jacobi_local_transition(params, local_config);
    if (local.is_ok()) {
        local.value.diagnostics.method_requested = config.method;
        local.value.diagnostics.spectral_status = spectral.status;
    }
    return local;
}

JacobiSparseTransitionResult build_jacobi_sparse_transition(
    const JacobiParams& params,
    const JacobiTransitionConfig& config) {
    const Status config_status = validate_jacobi_transition_config(config);
    if (!ok(config_status)) {
        return failure<JacobiSparseTransitionResult>(config_status);
    }
    if (config.storage != JacobiTransitionStorage::Sparse) {
        return failure<JacobiSparseTransitionResult>(Status::InvalidParameter);
    }
    const bool fixed = config.method == JacobiTransitionMethod::LocalFixed;
    JacobiSparseTransitionResult result = build_sparse_raw(
        params, config, fixed);
    if (!result.is_ok()) {
        return result;
    }
    const std::vector<double> weights = result.value.weights;
    if (config.correction ==
            JacobiStationarityCorrection::MetropolisHastings) {
        result = metropolis_hastings_correction(
            std::move(result.value), weights);
    } else if (config.correction == JacobiStationarityCorrection::IpFp) {
        result = ipfp_correction(
            std::move(result.value),
            weights,
            config.ipfp_tolerance,
            config.ipfp_max_iterations);
    }
    if (result.is_ok()) {
        finish_sparse_diagnostics(
            result.value, weights,
            fixed && config.derivatives);
    }
    return result;
}

JacobiVectorResult jacobi_sparse_left_multiply(
    const JacobiSparseTransition& transition,
    const std::vector<double>& values) {
    if (validate_jacobi_sparse_transition(transition)
            != JacobiSparseValidationCode::Ok
        || values.size() != static_cast<std::size_t>(transition.order)
        || !std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        JacobiVectorResult result;
        result.value = sparse_left_multiply_unchecked(transition, values);
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiVectorResult jacobi_sparse_to_dense(
    const JacobiSparseTransition& transition) {
    if (validate_jacobi_sparse_transition(transition)
            != JacobiSparseValidationCode::Ok) {
        return failure<JacobiVectorResult>(Status::InvalidParameter);
    }
    try {
        const std::size_t order = static_cast<std::size_t>(transition.order);
        const std::size_t width = static_cast<std::size_t>(
            transition.max_width);
        std::vector<double> dense(order * order, 0.0);
        for (std::size_t row = 0; row < order; ++row) {
            const std::size_t count = static_cast<std::size_t>(
                transition.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t sparse_offset = row * width + slot;
                const std::size_t column = static_cast<std::size_t>(
                    transition.indices[sparse_offset]);
                dense[row * order + column] =
                    transition.probabilities[sparse_offset];
            }
        }
        return success(std::move(dense));
    } catch (const std::bad_alloc&) {
        return failure<JacobiVectorResult>(Status::InvalidSize);
    }
}

JacobiHorizonResult jacobi_sparse_full_horizon_diagnostics(
    const JacobiParams& params,
    const std::vector<double>& tau,
    const std::vector<double>& weights,
    const JacobiSparseTransition& transition,
    std::int64_t steps) {
    if (!ok(validate_jacobi_params(
            params, std::numeric_limits<double>::infinity()))
        || validate_jacobi_sparse_transition(transition)
            != JacobiSparseValidationCode::Ok
        || steps < 0
        || tau.size() != static_cast<std::size_t>(transition.order)
        || weights.size() != tau.size()) {
        return failure<JacobiHorizonResult>(Status::InvalidParameter);
    }
    try {
        JacobiHorizonResult result;
        JacobiHorizonDiagnostics& diagnostics = result.value;
        diagnostics.steps = steps;
        std::vector<double> propagated = weights;
        const std::vector<double> one_step =
            sparse_left_multiply_unchecked(transition, propagated);
        for (std::int64_t step = 0; step < steps; ++step) {
            propagated = sparse_left_multiply_unchecked(
                transition, propagated);
        }
        double target_mean = 0.0;
        double propagated_mean = 0.0;
        for (std::size_t index = 0; index < tau.size(); ++index) {
            diagnostics.one_step_stationary_tv +=
                std::abs(one_step[index] - weights[index]);
            diagnostics.full_horizon_stationary_tv +=
                std::abs(propagated[index] - weights[index]);
            target_mean += weights[index] * tau[index];
            propagated_mean += propagated[index] * tau[index];
        }
        diagnostics.one_step_stationary_tv *= 0.5;
        diagnostics.full_horizon_stationary_tv *= 0.5;
        diagnostics.target_mean = target_mean;
        diagnostics.propagated_mean = propagated_mean;
        for (std::size_t index = 0; index < tau.size(); ++index) {
            const double target_delta = tau[index] - target_mean;
            const double propagated_delta = tau[index] - propagated_mean;
            diagnostics.target_variance +=
                weights[index] * target_delta * target_delta;
            diagnostics.propagated_variance +=
                propagated[index] * propagated_delta * propagated_delta;
        }
        diagnostics.relative_variance_error =
            diagnostics.target_variance > 0.0
            ? std::abs(
                diagnostics.propagated_variance
                - diagnostics.target_variance)
                / diagnostics.target_variance
            : 0.0;
        const double dt = steps > 0
            ? 1.0 / static_cast<double>(steps) : 0.0;
        const double target_correlation = steps > 0
            ? std::exp(-params.kappa * dt) : 1.0;
        double weighted_square_error = 0.0;
        double lag_covariance = 0.0;
        for (std::size_t row = 0; row < tau.size(); ++row) {
            double conditional = 0.0;
            const std::size_t count = static_cast<std::size_t>(
                transition.counts[row]);
            for (std::size_t slot = 0; slot < count; ++slot) {
                const std::size_t offset = sparse_offset(
                    transition, row, slot);
                conditional += transition.probabilities[offset]
                    * tau[static_cast<std::size_t>(transition.indices[offset])];
            }
            const double expected = steps > 0
                ? params.m + (tau[row] - params.m) * target_correlation
                : tau[row];
            const double error = conditional - expected;
            weighted_square_error += weights[row] * error * error;
            diagnostics.conditional_mean_max_error = std::max(
                diagnostics.conditional_mean_max_error, std::abs(error));
            lag_covariance += weights[row] * (tau[row] - target_mean)
                * (conditional - target_mean);
        }
        diagnostics.conditional_mean_rmse =
            std::sqrt(weighted_square_error);
        diagnostics.lag_one_correlation =
            diagnostics.target_variance > 0.0
            ? lag_covariance / diagnostics.target_variance : 0.0;
        diagnostics.target_lag_one_correlation = target_correlation;
        diagnostics.lag_one_correlation_error =
            diagnostics.lag_one_correlation - target_correlation;
        return result;
    } catch (const std::bad_alloc&) {
        return failure<JacobiHorizonResult>(Status::InvalidSize);
    }
}

JacobiAdaptiveSelectionResult select_sparse_jacobi_order(
    const JacobiParams& params,
    const JacobiTransitionConfig& config,
    const std::vector<int>& quad_orders,
    const JacobiAdaptiveThresholds& thresholds,
    bool require_pass) {
    if (config.numerical.n_obs < 2
        || quad_orders.empty()
        || !std::is_sorted(quad_orders.begin(), quad_orders.end())
        || std::adjacent_find(quad_orders.begin(), quad_orders.end())
            != quad_orders.end()
        || !std::all_of(quad_orders.begin(), quad_orders.end(), [](int order) {
            return order > 0 && order <= kMaxJacobiOrder;
        })
        || !std::isfinite(thresholds.max_full_horizon_tv)
        || thresholds.max_full_horizon_tv < 0.0
        || !std::isfinite(thresholds.max_relative_variance_error)
        || thresholds.max_relative_variance_error < 0.0
        || !std::isfinite(thresholds.max_conditional_mean_rmse)
        || thresholds.max_conditional_mean_rmse < 0.0
        || !std::isfinite(thresholds.max_lag_one_correlation_error)
        || thresholds.max_lag_one_correlation_error < 0.0) {
        return failure<JacobiAdaptiveSelectionResult>(Status::InvalidParameter);
    }
    JacobiTransitionConfig candidate_config = config;
    candidate_config.method = JacobiTransitionMethod::Local;
    candidate_config.storage = JacobiTransitionStorage::Sparse;
    candidate_config.correction = JacobiStationarityCorrection::None;
    candidate_config.derivatives = false;
    const int requested_basis_order = config.numerical.basis_order;
    JacobiAdaptiveSelectionResult result;
    bool have_successful = false;
    for (int order : quad_orders) {
        candidate_config.numerical.quad_order = order;
        const JacobiIntResult basis_order = resolve_jacobi_basis_order(
            requested_basis_order, order);
        if (!basis_order.is_ok()) {
            return failure<JacobiAdaptiveSelectionResult>(basis_order.status);
        }
        candidate_config.numerical.basis_order = basis_order.value;
        JacobiAdaptiveCandidate record;
        record.quad_order = order;
        JacobiSparseTransitionResult transition =
            build_jacobi_sparse_transition(params, candidate_config);
        if (!transition.is_ok()) {
            record.status = transition.status;
            record.memory_limited =
                transition.status == Status::InvalidSize;
            result.value.candidates.push_back(record);
            if (record.memory_limited) {
                break;
            }
            result.status = transition.status;
            result.failure = transition.failure;
            return result;
        }
        const JacobiShapeResult shape = jacobi_stationary_shape(params);
        JacobiBasisResult rule = build_jacobi_rule(
            shape.value.alpha,
            shape.value.beta,
            order,
            1,
            candidate_config.numerical.memory_budget_bytes);
        if (!rule.is_ok()) {
            result.status = rule.status;
            return result;
        }
        JacobiHorizonResult horizon =
            jacobi_sparse_full_horizon_diagnostics(
                params,
                rule.value.tau,
                rule.value.weights,
                transition.value,
                config.numerical.n_obs - 1);
        if (!horizon.is_ok()) {
            result.status = horizon.status;
            return result;
        }
        record.status = Status::Ok;
        record.retained_bytes =
            transition.value.diagnostics.retained_bytes;
        record.diagnostics = horizon.value;
        record.passed =
            horizon.value.full_horizon_stationary_tv
                <= thresholds.max_full_horizon_tv
            && horizon.value.relative_variance_error
                <= thresholds.max_relative_variance_error
            && horizon.value.conditional_mean_rmse
                <= thresholds.max_conditional_mean_rmse
            && std::abs(horizon.value.lag_one_correlation_error)
                <= thresholds.max_lag_one_correlation_error;
        result.value.candidates.push_back(record);
        have_successful = true;
        result.value.transition = std::move(transition.value);
        result.value.tau = std::move(rule.value.tau);
        result.value.weights = std::move(rule.value.weights);
        result.value.selected_quad_order = order;
        result.value.passed = record.passed;
        if (record.passed) {
            break;
        }
    }
    if (!have_successful) {
        result.status = Status::InvalidSize;
        result.failure.operation = 7;
        return result;
    }
    result.value.exhausted = !result.value.passed;
    if (require_pass && !result.value.passed) {
        result.status = Status::NumericalFailure;
        result.failure.operation = 6;
    }
    return result;
}

}  // namespace scar
