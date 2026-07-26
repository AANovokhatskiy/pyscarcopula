#include "scar/gas_rvine.hpp"

#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace scar {
namespace {

bool valid_index(int value, int limit) {
    return value >= 0 && value < limit;
}

bool same_size(const std::vector<int>& left, const std::vector<int>& right) {
    return left.size() == right.size();
}

bool valid_offsets(const std::vector<int>& offsets, std::size_t item_count) {
    if (offsets.empty() || offsets.front() != 0
        || offsets.back() != static_cast<int>(item_count)) {
        return false;
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            return false;
        }
    }
    return true;
}

bool valid_plan(const RVineTraversalPlan& plan, std::size_t edge_count) {
    const std::size_t column_count = plan.column_uniforms.size();
    if (plan.dimension < 2 || plan.node_count < plan.dimension
        || !valid_index(plan.last_uniform_column, plan.dimension)
        || !valid_index(plan.last_output_node, plan.node_count)
        || plan.output_nodes.size() != static_cast<std::size_t>(plan.dimension)
        || plan.inverse_offsets.size() != column_count + 1
        || plan.forward_offsets.size() != column_count + 1
        || !valid_offsets(plan.inverse_offsets, plan.inverse_edges.size())
        || !valid_offsets(plan.forward_offsets, plan.forward_edges.size())
        || !same_size(plan.inverse_edges, plan.inverse_partner_nodes)
        || !same_size(plan.inverse_edges, plan.inverse_output_nodes)
        || !same_size(plan.inverse_edges, plan.inverse_transposed)
        || !same_size(plan.forward_edges, plan.forward_leaf_nodes)
        || !same_size(plan.forward_edges, plan.forward_partner_nodes)
        || !same_size(plan.forward_edges, plan.forward_leaf_output_nodes)
        || !same_size(plan.forward_edges, plan.forward_partner_output_nodes)
        || !same_size(plan.forward_edges, plan.forward_transposed)
        || plan.update_u1_nodes.size() != edge_count
        || plan.update_u2_nodes.size() != edge_count) {
        return false;
    }
    for (int value : plan.output_nodes) {
        if (!valid_index(value, plan.node_count)) {
            return false;
        }
    }
    for (int value : plan.column_uniforms) {
        if (!valid_index(value, plan.dimension)) {
            return false;
        }
    }
    for (std::size_t i = 0; i < plan.inverse_edges.size(); ++i) {
        if (!valid_index(plan.inverse_edges[i], static_cast<int>(edge_count))
            || !valid_index(plan.inverse_partner_nodes[i], plan.node_count)
            || !valid_index(plan.inverse_output_nodes[i], plan.node_count)
            || (plan.inverse_transposed[i] != 0
                && plan.inverse_transposed[i] != 1)) {
            return false;
        }
    }
    for (std::size_t i = 0; i < plan.forward_edges.size(); ++i) {
        if (!valid_index(plan.forward_edges[i], static_cast<int>(edge_count))
            || !valid_index(plan.forward_leaf_nodes[i], plan.node_count)
            || !valid_index(plan.forward_partner_nodes[i], plan.node_count)
            || !valid_index(plan.forward_leaf_output_nodes[i], plan.node_count)
            || !valid_index(plan.forward_partner_output_nodes[i], plan.node_count)
            || (plan.forward_transposed[i] != 0
                && plan.forward_transposed[i] != 1)) {
            return false;
        }
    }
    for (std::size_t i = 0; i < edge_count; ++i) {
        if (!valid_index(plan.update_u1_nodes[i], plan.node_count)
            || !valid_index(plan.update_u2_nodes[i], plan.node_count)) {
            return false;
        }
    }
    return true;
}

double clipped(double value) {
    return scar_internal::clip_pseudo_observation(value);
}

void fail(
    GasRvineSampleResult& out,
    int status,
    std::int64_t row,
    int edge) {
    out.status = status;
    out.failure_row = row;
    out.failure_edge = edge;
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
        || !valid_plan(plan, edge_count)) {
        fail(out, SCAR_INVALID_SIZE, -1, -1);
        return out;
    }
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    const std::size_t dimension = static_cast<std::size_t>(plan.dimension);
    if (rows > std::numeric_limits<std::size_t>::max() / dimension
        || rows > std::numeric_limits<std::size_t>::max() / edge_count) {
        fail(out, SCAR_INVALID_SIZE, -1, -1);
        return out;
    }
    std::vector<CopulaSpec> transposed_copulas;
    transposed_copulas.reserve(edge_count);
    for (const GasRvineEdge& edge : edges) {
        if (!is_supported(edge.copula) || edge.copula.dim != 2) {
            fail(out, SCAR_INVALID_FAMILY, -1, -1);
            return out;
        }
        transposed_copulas.push_back(
            scar_internal::transposed_copula_spec(edge.copula));
    }

    GasEvaluator evaluator;
    std::vector<double> gas_g(edge_count, 0.0);
    std::vector<double> gas_r(edge_count, 0.0);
    for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
        if (!edges[edge_index].dynamic) {
            continue;
        }
        const GasStateResult state = evaluator.initial_state(
            edges[edge_index].gas_params,
            edges[edge_index].copula,
            edges[edge_index].gas_config);
        if (state.status != SCAR_OK) {
            fail(out, state.status, -1, static_cast<int>(edge_index));
            return out;
        }
        gas_g[edge_index] = state.g;
        gas_r[edge_index] = state.parameter;
    }

    const std::size_t result_size = rows * dimension;
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
        nodes[static_cast<std::size_t>(plan.last_output_node)] = clipped(
            uniform_row[plan.last_uniform_column]);

        for (std::size_t column = 0;
             column < plan.column_uniforms.size(); ++column) {
            double current = clipped(uniform_row[plan.column_uniforms[column]]);
            for (int index = plan.inverse_offsets[column];
                 index < plan.inverse_offsets[column + 1]; ++index) {
                const int edge_index = plan.inverse_edges[index];
                const double partner = nodes[static_cast<std::size_t>(
                    plan.inverse_partner_nodes[index])];
                const double parameter = edge_parameter(row, edge_index);
                const std::size_t edge_position =
                    static_cast<std::size_t>(edge_index);
                const CopulaSpec& inverse_copula =
                    plan.inverse_transposed[index] != 0
                    ? transposed_copulas[edge_position]
                    : edges[edge_position].copula;
                current = scar_internal::copula_h_inverse_rotated(
                    inverse_copula,
                    current,
                    partner,
                    parameter);
                if (!std::isfinite(current)) {
                    fail(out, SCAR_NUMERICAL_FAILURE, row, edge_index);
                    return out;
                }
                current = clipped(current);
                nodes[static_cast<std::size_t>(
                    plan.inverse_output_nodes[index])] = current;
            }

            for (int index = plan.forward_offsets[column];
                 index < plan.forward_offsets[column + 1]; ++index) {
                const int edge_index = plan.forward_edges[index];
                const GasRvineEdge& edge =
                    edges[static_cast<std::size_t>(edge_index)];
                const double leaf = nodes[static_cast<std::size_t>(
                    plan.forward_leaf_nodes[index])];
                const double partner = nodes[static_cast<std::size_t>(
                    plan.forward_partner_nodes[index])];
                const double parameter = edge_parameter(row, edge_index);
                const std::size_t edge_position =
                    static_cast<std::size_t>(edge_index);
                const bool leaf_is_transposed =
                    plan.forward_transposed[index] != 0;
                const CopulaSpec& leaf_copula =
                    leaf_is_transposed
                    ? transposed_copulas[edge_position]
                    : edge.copula;
                const CopulaSpec& partner_copula =
                    leaf_is_transposed
                    ? edge.copula
                    : transposed_copulas[edge_position];
                const double leaf_next = scar_internal::copula_h_rotated(
                    leaf_copula, leaf, partner, parameter);
                const double partner_next = scar_internal::copula_h_rotated(
                    partner_copula,
                    partner,
                    leaf,
                    parameter);
                if (!std::isfinite(leaf_next) || !std::isfinite(partner_next)) {
                    fail(out, SCAR_NUMERICAL_FAILURE, row, edge_index);
                    return out;
                }
                nodes[static_cast<std::size_t>(
                    plan.forward_leaf_output_nodes[index])] = clipped(leaf_next);
                nodes[static_cast<std::size_t>(
                    plan.forward_partner_output_nodes[index])] = clipped(
                        partner_next);
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
            const GasUpdateResult update = evaluator.update_one(
                edge.gas_params,
                edge.copula,
                gas_g[edge_index],
                u1,
                u2,
                edge.gas_config);
            if (update.status != SCAR_OK) {
                fail(out, update.status, row, static_cast<int>(edge_index));
                return out;
            }
            gas_g[edge_index] = update.g_next;
            gas_r[edge_index] = update.r_next;
        }
    }
    return out;
}

}  // namespace scar
