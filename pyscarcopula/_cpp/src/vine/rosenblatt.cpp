#include "scar/rvine.hpp"

#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace scar::rvine {
namespace {

void fail_rosenblatt(
    RosenblattResult& out,
    int status,
    std::int64_t row,
    int edge,
    int operation) noexcept {
    out.status = status;
    out.failure_row = row;
    out.failure_edge = edge;
    out.failure_operation = operation;
}

}  // namespace

RosenblattResult rosenblatt_transform(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads) {
    RosenblattResult out;
    out.n_rows = observation_rows;
    out.dimension = plan.dimension;
    out.n_threads_requested = n_threads;
    if (n_threads <= 0 || observation_rows < 0
        || observation_columns != plan.dimension
        || parameters.n_rows != observation_rows
        || plan.residual_nodes.size()
            != static_cast<std::size_t>(plan.dimension)
        || !validate_density_plan(plan, edges.size())) {
        fail_rosenblatt(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }

    const auto rows = static_cast<std::size_t>(observation_rows);
    const auto dimension = static_cast<std::size_t>(plan.dimension);
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(rows, dimension, value_count)
        || observations.size() != value_count
        || (value_count > 0 && observations.data() == nullptr)) {
        fail_rosenblatt(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    const int parameter_status = validate_parameter_pack(edges, parameters);
    if (parameter_status != SCAR_OK) {
        fail_rosenblatt(out, parameter_status, -1, -1, -1);
        return out;
    }
    std::vector<PreparedEdge> prepared_edges;
    const int prepare_status = prepare_edges(edges, prepared_edges);
    if (prepare_status != SCAR_OK) {
        fail_rosenblatt(out, prepare_status, -1, -1, -1);
        return out;
    }
    for (std::size_t index = 0; index < value_count; ++index) {
        if (!std::isfinite(observations[index])) {
            fail_rosenblatt(
                out,
                SCAR_INVALID_PARAMETER,
                static_cast<std::int64_t>(index / dimension),
                -1,
                -1);
            return out;
        }
    }

    out.residuals.assign(value_count, 0.0);
    const double missing = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> nodes(
        static_cast<std::size_t>(plan.node_count), missing);
    for (std::size_t row = 0; row < rows; ++row) {
        std::fill(nodes.begin(), nodes.end(), missing);
        const double* observation_row = observations.data() + row * dimension;
        for (std::size_t variable = 0; variable < dimension; ++variable) {
            nodes[static_cast<std::size_t>(plan.input_nodes[variable])] =
                clip_open_unit(observation_row[variable]);
        }

        for (std::size_t operation = 0;
             operation < plan.edge_indices.size(); ++operation) {
            const int output1 = plan.output1_nodes[operation];
            if (output1 == -1) {
                continue;
            }
            const int edge_index = plan.edge_indices[operation];
            const PreparedEdge& edge = prepared_edges[
                static_cast<std::size_t>(edge_index)];
            const double first = nodes[static_cast<std::size_t>(
                plan.input1_nodes[operation])];
            const double second = nodes[static_cast<std::size_t>(
                plan.input2_nodes[operation])];
            double first_next = first;
            double second_next = second;
            if (edge.edge.parameter_free) {
                ++out.independence_fast_paths;
            } else {
                const double parameter = parameter_at(
                    edge.edge, parameters, static_cast<std::int64_t>(row));
                h_pair(
                    edge,
                    plan.transposed[operation] != 0,
                    first,
                    second,
                    parameter,
                    first_next,
                    second_next);
            }
            ++out.h_pair_operations;
            if (!std::isfinite(first_next) || !std::isfinite(second_next)) {
                fail_rosenblatt(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(row),
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            nodes[static_cast<std::size_t>(output1)] =
                clip_open_unit(first_next);
            nodes[static_cast<std::size_t>(
                plan.output2_nodes[operation])] =
                    clip_open_unit(second_next);
        }

        double* residual_row = out.residuals.data() + row * dimension;
        for (std::size_t column = 0; column < dimension; ++column) {
            const double value = nodes[static_cast<std::size_t>(
                plan.residual_nodes[column])];
            if (!std::isfinite(value)) {
                fail_rosenblatt(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(row),
                    -1,
                    static_cast<int>(plan.edge_indices.size()));
                return out;
            }
            residual_row[column] = clip_open_unit(value);
        }
    }
    return out;
}

}  // namespace scar::rvine
