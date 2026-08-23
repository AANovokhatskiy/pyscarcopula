#include "scar/rvine.hpp"

#include "scar/core/threading.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace scar::rvine {
namespace detail {

int prepare_density_plan_request(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads,
    std::vector<PreparedEdge>& prepared_edges,
    std::size_t& value_count,
    std::int64_t& failure_row) {
    if (!scar_internal::valid_thread_count(n_threads) || observation_rows < 0
        || observation_columns != plan.dimension
        || parameters.n_rows != observation_rows
        || !validate_density_plan(plan, edges.size())) {
        return SCAR_INVALID_SIZE;
    }

    const auto rows = static_cast<std::size_t>(observation_rows);
    const auto dimension = static_cast<std::size_t>(plan.dimension);
    if (!scar_internal::checked_size_mul(rows, dimension, value_count)
        || observations.size() != value_count
        || (value_count > 0 && observations.data() == nullptr)) {
        return SCAR_INVALID_SIZE;
    }
    const int parameter_status = validate_parameter_pack(edges, parameters);
    if (parameter_status != SCAR_OK) {
        return parameter_status;
    }
    const int prepare_status = prepare_edges(edges, prepared_edges);
    if (prepare_status != SCAR_OK) {
        return prepare_status;
    }
    for (std::size_t index = 0; index < value_count; ++index) {
        if (!std::isfinite(observations[index])) {
            failure_row = static_cast<std::int64_t>(index / dimension);
            return SCAR_INVALID_PARAMETER;
        }
    }
    return SCAR_OK;
}

int evaluate_density_plan_rows(
    const RVineDensityPlan& plan,
    const std::vector<PreparedEdge>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    double* log_pdf,
    double* residuals,
    bool tolerate_non_finite,
    std::vector<double>& node_workspace,
    std::int64_t& failure_row,
    int& failure_edge,
    int& failure_operation,
    std::uint64_t& non_finite_rows,
    DensityDiagnostics& diagnostics) {
    if (observation_rows < 0 || observation_columns != plan.dimension
        || (observation_rows > 0
            && log_pdf == nullptr && residuals == nullptr)) {
        return SCAR_INVALID_SIZE;
    }
    const auto rows = static_cast<std::size_t>(observation_rows);
    const auto dimension = static_cast<std::size_t>(plan.dimension);
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(rows, dimension, value_count)
        || observations.size() != value_count
        || (value_count > 0 && observations.data() == nullptr)) {
        return SCAR_INVALID_SIZE;
    }

    const double missing = std::numeric_limits<double>::quiet_NaN();
    node_workspace.assign(
        static_cast<std::size_t>(plan.node_count), missing);
    for (std::size_t row = 0; row < rows; ++row) {
        std::fill(node_workspace.begin(), node_workspace.end(), missing);
        const double* observation_row = observations.data() + row * dimension;
        for (std::size_t variable = 0; variable < dimension; ++variable) {
            node_workspace[static_cast<std::size_t>(
                plan.input_nodes[variable])] = clip_open_unit(
                    observation_row[variable]);
        }

        double row_log_pdf = 0.0;
        bool row_is_non_finite = false;
        for (std::size_t operation = 0;
             operation < plan.edge_indices.size(); ++operation) {
            const int edge_index = plan.edge_indices[operation];
            const PreparedEdge& edge = edges[
                static_cast<std::size_t>(edge_index)];
            const double first = node_workspace[static_cast<std::size_t>(
                plan.input1_nodes[operation])];
            const double second = node_workspace[static_cast<std::size_t>(
                plan.input2_nodes[operation])];
            const bool transposed = plan.transposed[operation] != 0;
            const double parameter = parameter_at(
                edge.edge, parameters, static_cast<std::int64_t>(row));

            if (log_pdf != nullptr) {
                double contribution = 0.0;
                if (edge.edge.parameter_free) {
                    ++diagnostics.independence_fast_paths;
                } else {
                    contribution = edge_log_pdf(
                        edge, transposed, first, second, parameter);
                }
                ++diagnostics.density_operations;
                if (!std::isfinite(contribution)
                    || !std::isfinite(row_log_pdf + contribution)) {
                    if (!tolerate_non_finite) {
                        failure_row = static_cast<std::int64_t>(row);
                        failure_edge = edge_index;
                        failure_operation = static_cast<int>(operation);
                        return SCAR_NUMERICAL_FAILURE;
                    }
                    log_pdf[row] = -std::numeric_limits<double>::infinity();
                    ++non_finite_rows;
                    row_is_non_finite = true;
                    break;
                }
                row_log_pdf += contribution;
            }

            const int output1 = plan.output1_nodes[operation];
            if (output1 == -1) {
                continue;
            }
            double first_next = first;
            double second_next = second;
            if (edge.edge.parameter_free) {
                // The independence h-functions are the identity in both
                // directions, so no family dispatch is needed.
                if (log_pdf == nullptr) {
                    ++diagnostics.independence_fast_paths;
                }
            } else {
                h_pair(
                    edge,
                    transposed,
                    first,
                    second,
                    parameter,
                    first_next,
                    second_next);
            }
            ++diagnostics.h_pair_operations;
            if (!std::isfinite(first_next) || !std::isfinite(second_next)) {
                if (!tolerate_non_finite) {
                    failure_row = static_cast<std::int64_t>(row);
                    failure_edge = edge_index;
                    failure_operation = static_cast<int>(operation);
                    return SCAR_NUMERICAL_FAILURE;
                }
                log_pdf[row] = -std::numeric_limits<double>::infinity();
                ++non_finite_rows;
                row_is_non_finite = true;
                break;
            }
            node_workspace[static_cast<std::size_t>(output1)] =
                clip_open_unit(first_next);
            node_workspace[static_cast<std::size_t>(
                plan.output2_nodes[operation])] =
                    clip_open_unit(second_next);
        }
        if (!row_is_non_finite) {
            if (log_pdf != nullptr) {
                log_pdf[row] = row_log_pdf;
            }
            if (residuals != nullptr) {
                double* residual_row = residuals + row * dimension;
                for (std::size_t column = 0;
                     column < dimension;
                     ++column) {
                    const double value = node_workspace[
                        static_cast<std::size_t>(
                            plan.residual_nodes[column])];
                    if (!std::isfinite(value)) {
                        failure_row = static_cast<std::int64_t>(row);
                        failure_edge = -1;
                        failure_operation = static_cast<int>(
                            plan.edge_indices.size());
                        return SCAR_NUMERICAL_FAILURE;
                    }
                    residual_row[column] = clip_open_unit(value);
                }
            }
        }
    }
    return SCAR_OK;
}

}  // namespace detail

DensityResult log_pdf_rows(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads) {
    DensityResult out;
    out.n_rows = observation_rows;
    out.dimension = plan.dimension;
    out.n_threads_requested = n_threads;
    std::size_t value_count = 0;
    std::vector<PreparedEdge> prepared_edges;
    const int request_status = detail::prepare_density_plan_request(
        plan,
        edges,
        parameters,
        observations,
        observation_rows,
        observation_columns,
        n_threads,
        prepared_edges,
        value_count,
        out.failure_row);
    if (request_status != SCAR_OK) {
        out.status = request_status;
        return out;
    }

    const auto rows = static_cast<std::size_t>(observation_rows);
    out.log_pdf.assign(rows, 0.0);
    std::vector<double> node_workspace;
    std::uint64_t non_finite_rows = 0;
    const int status = detail::evaluate_density_plan_rows(
        plan,
        prepared_edges,
        parameters,
        observations,
        observation_rows,
        observation_columns,
        out.log_pdf.data(),
        nullptr,
        false,
        node_workspace,
        out.failure_row,
        out.failure_edge,
        out.failure_operation,
        non_finite_rows,
        out.diagnostics);
    if (status != SCAR_OK) {
        out.status = status;
    }
    return out;
}

}  // namespace scar::rvine
