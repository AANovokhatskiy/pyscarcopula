#include "scar/gas_rvine.hpp"

#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/rvine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>

namespace scar {
namespace {

void fail(
    GasRvineSampleResult& out,
    int status,
    std::int64_t row,
    int edge) {
    out.status = status_from_int(status);
    out.failure.row = row;
    out.failure.edge = edge;
}

}  // namespace

GasRvineSampleResult gas_rvine_sample(
    const std::vector<GasRvineEdge>& edges,
    const RVineTraversalPlan& plan,
    const double* uniforms,
    std::int64_t n_rows,
    std::int64_t uniform_columns,
    const double* parameter_paths,
    std::int64_t parameter_rows,
    std::int64_t parameter_edges) {

    GasRvineSampleResult out;
    out.n_rows = n_rows;
    out.dimension = plan.dimension;
    const std::size_t edge_count = edges.size();
    if (uniforms == nullptr || parameter_paths == nullptr || n_rows <= 0
        || uniform_columns != plan.dimension
        || parameter_rows != n_rows
        || parameter_edges != static_cast<std::int64_t>(edge_count)
        || edge_count == 0
        || !rvine::validate_traversal_plan(plan, edge_count)) {
        fail(out, SCAR_INVALID_SIZE, -1, -1);
        return out;
    }
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    const std::size_t dimension = static_cast<std::size_t>(plan.dimension);
    std::size_t result_size = 0;
    std::size_t parameter_value_count = 0;
    if (!scar_internal::checked_shape_size(
            rows, dimension, result_size)
        || !scar_internal::checked_shape_size(
            rows, edge_count, parameter_value_count)) {
        fail(out, SCAR_INVALID_SIZE, -1, -1);
        return out;
    }
    std::vector<rvine::EdgeSpec> edge_specs;
    edge_specs.reserve(edge_count);
    for (const GasRvineEdge& edge : edges) {
        rvine::EdgeSpec spec;
        spec.copula = edge.copula;
        edge_specs.push_back(spec);
    }
    std::vector<rvine::PreparedEdge> prepared_edges;
    const int prepare_status = rvine::prepare_edges(
        edge_specs, prepared_edges);
    if (prepare_status != SCAR_OK) {
        fail(out, prepare_status, -1, -1);
        return out;
    }

    GasEvaluator evaluator;
    std::vector<double> gas_g(edge_count, 0.0);
    std::vector<double> gas_r(edge_count, 0.0);
    std::vector<std::unique_ptr<PreparedDynamicEmission>> gas_emissions(
        edge_count);
    std::vector<std::unique_ptr<PreparedDynamicEmissionWorkspace>>
        gas_workspaces(edge_count);
    for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
        if (!edges[edge_index].dynamic) {
            continue;
        }
        gas_emissions[edge_index] =
            std::make_unique<PreparedDynamicEmission>(
                edges[edge_index].copula);
        gas_workspaces[edge_index] =
            std::make_unique<PreparedDynamicEmissionWorkspace>(
                gas_emissions[edge_index]->make_workspace(true));
        const GasStateResult state = evaluator.initial_state_prepared(
            edges[edge_index].gas_params,
            *gas_emissions[edge_index],
            edges[edge_index].gas_config);
        if (!state.is_ok()) {
            fail(
                out,
                static_cast<int>(state.status),
                -1,
                static_cast<int>(edge_index));
            return out;
        }
        gas_g[edge_index] = state.g;
        gas_r[edge_index] = state.parameter;
    }

    out.values.assign(result_size, 0.0);
    std::vector<double> nodes(
        static_cast<std::size_t>(plan.node_count),
        std::numeric_limits<double>::quiet_NaN());

    auto edge_parameter = [&](std::int64_t row, int edge_index) {
        if (edges[static_cast<std::size_t>(edge_index)].dynamic) {
            return gas_r[static_cast<std::size_t>(edge_index)];
        }
        return parameter_paths[
            static_cast<std::size_t>(row) * edge_count
            + static_cast<std::size_t>(edge_index)];
    };

    for (std::int64_t row = 0; row < n_rows; ++row) {
        std::fill(
            nodes.begin(), nodes.end(),
            std::numeric_limits<double>::quiet_NaN());
        const double* uniform_row = uniforms
            + static_cast<std::size_t>(row)
                * static_cast<std::size_t>(plan.dimension);
        nodes[static_cast<std::size_t>(plan.last_output_node)] =
            rvine::clip_open_unit(uniform_row[plan.last_uniform_column]);

        for (std::size_t column = 0;
             column < plan.column_uniforms.size(); ++column) {
            double current = rvine::clip_open_unit(
                uniform_row[plan.column_uniforms[column]]);
            for (int index = plan.inverse_offsets[column];
                 index < plan.inverse_offsets[column + 1]; ++index) {
                const int edge_index = plan.inverse_edges[index];
                const double partner = nodes[static_cast<std::size_t>(
                    plan.inverse_partner_nodes[index])];
                const double parameter = edge_parameter(row, edge_index);
                current = rvine::h_inverse(
                    prepared_edges[static_cast<std::size_t>(edge_index)],
                    plan.inverse_transposed[index] != 0,
                    current,
                    partner,
                    parameter);
                if (!std::isfinite(current)) {
                    fail(out, SCAR_NUMERICAL_FAILURE, row, edge_index);
                    return out;
                }
                current = rvine::clip_open_unit(current);
                nodes[static_cast<std::size_t>(
                    plan.inverse_output_nodes[index])] = current;
            }

            for (int index = plan.forward_offsets[column];
                 index < plan.forward_offsets[column + 1]; ++index) {
                const int edge_index = plan.forward_edges[index];
                const double leaf = nodes[static_cast<std::size_t>(
                    plan.forward_leaf_nodes[index])];
                const double partner = nodes[static_cast<std::size_t>(
                    plan.forward_partner_nodes[index])];
                const double parameter = edge_parameter(row, edge_index);
                const bool leaf_is_transposed =
                    plan.forward_transposed[index] != 0;
                double leaf_next = 0.0;
                double partner_next = 0.0;
                rvine::h_pair(
                    prepared_edges[static_cast<std::size_t>(edge_index)],
                    leaf_is_transposed,
                    leaf,
                    partner,
                    parameter,
                    leaf_next,
                    partner_next);
                if (!std::isfinite(leaf_next) || !std::isfinite(partner_next)) {
                    fail(out, SCAR_NUMERICAL_FAILURE, row, edge_index);
                    return out;
                }
                nodes[static_cast<std::size_t>(
                    plan.forward_leaf_output_nodes[index])] =
                    rvine::clip_open_unit(leaf_next);
                nodes[static_cast<std::size_t>(
                    plan.forward_partner_output_nodes[index])] =
                    rvine::clip_open_unit(partner_next);
            }
        }

        double* result_row = out.values.data()
            + static_cast<std::size_t>(row)
                * static_cast<std::size_t>(plan.dimension);
        for (int variable = 0; variable < plan.dimension; ++variable) {
            result_row[variable] = nodes[static_cast<std::size_t>(
                plan.output_nodes[static_cast<std::size_t>(variable)])];
            if (!std::isfinite(result_row[variable])) {
                fail(out, SCAR_NUMERICAL_FAILURE, row, -1);
                return out;
            }
        }

        for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
            const GasRvineEdge& edge = edges[edge_index];
            if (!edge.dynamic) {
                continue;
            }
            const double u1 = nodes[static_cast<std::size_t>(
                plan.update_u1_nodes[edge_index])];
            const double u2 = nodes[static_cast<std::size_t>(
                plan.update_u2_nodes[edge_index])];
            const GasUpdateResult update = evaluator.update_one_prepared(
                edge.gas_params,
                *gas_emissions[edge_index],
                *gas_workspaces[edge_index],
                gas_g[edge_index],
                u1,
                u2,
                edge.gas_config);
            if (!update.is_ok()) {
                fail(
                    out,
                    static_cast<int>(update.status),
                    row,
                    static_cast<int>(edge_index));
                return out;
            }
            gas_g[edge_index] = update.g_next;
            gas_r[edge_index] = update.r_next;
        }
    }
    return out;
}

}  // namespace scar
