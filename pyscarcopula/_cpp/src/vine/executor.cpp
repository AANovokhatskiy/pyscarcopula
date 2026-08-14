#include "scar/rvine.hpp"

#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace scar::rvine {
namespace {

bool same_size(const std::vector<int>& left, const std::vector<int>& right) {
    return left.size() == right.size();
}

bool valid_offsets(
    const std::vector<int>& offsets,
    std::size_t item_count) {
    if (offsets.empty() || offsets.front() != 0
        || offsets.back() != static_cast<int>(item_count)) {
        return false;
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            return false;
        }
    }
    return true;
}

bool valid_orientation(int value) {
    return value == 0 || value == 1;
}

bool valid_optional_node(int value, int node_count) {
    return value == -1 || valid_index(value, node_count);
}

bool same_operation_size(
    std::size_t operation_count,
    const std::vector<int>& values) {
    return values.size() == operation_count;
}

bool matches_used_edges(
    const std::vector<int>& operation_edges,
    const std::vector<int>& used_edges) {
    std::vector<int> expected = operation_edges;
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    return expected == used_edges;
}

}  // namespace

bool valid_index(int value, int limit) noexcept {
    return value >= 0 && value < limit;
}

bool validate_traversal_plan(
    const RVineTraversalPlan& plan,
    std::size_t edge_count) noexcept {
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
    for (std::size_t index = 0; index < plan.inverse_edges.size(); ++index) {
        if (!valid_index(
                plan.inverse_edges[index], static_cast<int>(edge_count))
            || !valid_index(plan.inverse_partner_nodes[index], plan.node_count)
            || !valid_index(plan.inverse_output_nodes[index], plan.node_count)
            || (plan.inverse_transposed[index] != 0
                && plan.inverse_transposed[index] != 1)) {
            return false;
        }
    }
    for (std::size_t index = 0; index < plan.forward_edges.size(); ++index) {
        if (!valid_index(
                plan.forward_edges[index], static_cast<int>(edge_count))
            || !valid_index(plan.forward_leaf_nodes[index], plan.node_count)
            || !valid_index(
                plan.forward_partner_nodes[index], plan.node_count)
            || !valid_index(
                plan.forward_leaf_output_nodes[index], plan.node_count)
            || !valid_index(
                plan.forward_partner_output_nodes[index], plan.node_count)
            || (plan.forward_transposed[index] != 0
                && plan.forward_transposed[index] != 1)) {
            return false;
        }
    }
    for (std::size_t index = 0; index < edge_count; ++index) {
        if (!valid_index(plan.update_u1_nodes[index], plan.node_count)
            || !valid_index(plan.update_u2_nodes[index], plan.node_count)) {
            return false;
        }
    }
    return true;
}

bool validate_conditional_plan(
    const RVineConditionalPlan& plan,
    std::size_t edge_count) noexcept {
    const std::size_t operation_count = plan.opcodes.size();
    if (plan.dimension < 2 || plan.node_count < plan.dimension
        || plan.output_nodes.size()
            != static_cast<std::size_t>(plan.dimension)
        || plan.given_variables.size() != plan.given_nodes.size()
        || plan.node_sources.size()
            != static_cast<std::size_t>(plan.node_count)
        || plan.node_source_indices.size()
            != static_cast<std::size_t>(plan.node_count)
        || !same_operation_size(operation_count, plan.edge_indices)
        || !same_operation_size(operation_count, plan.input1_nodes)
        || !same_operation_size(operation_count, plan.input2_nodes)
        || !same_operation_size(operation_count, plan.output1_nodes)
        || !same_operation_size(operation_count, plan.output2_nodes)
        || !same_operation_size(operation_count, plan.transposed)) {
        return false;
    }

    std::vector<unsigned char> initialized(
        static_cast<std::size_t>(plan.node_count), 0);
    std::vector<unsigned char> declared_source(
        static_cast<std::size_t>(plan.node_count), 0);
    std::vector<unsigned char> given_variables(
        static_cast<std::size_t>(plan.dimension), 0);
    for (std::size_t index = 0; index < plan.given_nodes.size(); ++index) {
        const int variable = plan.given_variables[index];
        const int node = plan.given_nodes[index];
        if (!valid_index(variable, plan.dimension)
            || !valid_index(node, plan.node_count)
            || given_variables[static_cast<std::size_t>(variable)] != 0
            || declared_source[static_cast<std::size_t>(node)] != 0
            || plan.node_sources[static_cast<std::size_t>(node)]
                != static_cast<int>(RVineNodeSource::Given)
            || plan.node_source_indices[static_cast<std::size_t>(node)]
                != variable) {
            return false;
        }
        given_variables[static_cast<std::size_t>(variable)] = 1;
        declared_source[static_cast<std::size_t>(node)] = 1;
        initialized[static_cast<std::size_t>(node)] = 1;
    }
    for (int node : plan.uniform_nodes) {
        if (!valid_index(node, plan.node_count)
            || declared_source[static_cast<std::size_t>(node)] != 0
            || plan.node_sources[static_cast<std::size_t>(node)]
                != static_cast<int>(RVineNodeSource::Uniform)
            || !valid_index(
                plan.node_source_indices[static_cast<std::size_t>(node)],
                plan.dimension)) {
            return false;
        }
        declared_source[static_cast<std::size_t>(node)] = 1;
        initialized[static_cast<std::size_t>(node)] = 1;
    }
    for (int node = 0; node < plan.node_count; ++node) {
        const std::size_t position = static_cast<std::size_t>(node);
        const int source = plan.node_sources[position];
        const int source_index = plan.node_source_indices[position];
        if (source == static_cast<int>(RVineNodeSource::Computed)) {
            if (source_index != -1 || declared_source[position] != 0) {
                return false;
            }
        } else if (
            source != static_cast<int>(RVineNodeSource::Given)
            && source != static_cast<int>(RVineNodeSource::Uniform)) {
            return false;
        } else if (declared_source[position] == 0) {
            return false;
        }
    }

    std::vector<int> operation_edges;
    operation_edges.reserve(operation_count);
    for (std::size_t index = 0; index < operation_count; ++index) {
        const int opcode = plan.opcodes[index];
        const int edge = plan.edge_indices[index];
        const int input1 = plan.input1_nodes[index];
        const int input2 = plan.input2_nodes[index];
        const int output1 = plan.output1_nodes[index];
        const int output2 = plan.output2_nodes[index];
        if (!valid_index(input1, plan.node_count)
            || initialized[static_cast<std::size_t>(input1)] == 0
            || !valid_index(output1, plan.node_count)
            || !valid_orientation(plan.transposed[index])) {
            return false;
        }
        if (opcode == static_cast<int>(RVineOpcode::COPY)) {
            if (edge != -1 || input2 != -1 || output2 != -1
                || plan.transposed[index] != 0) {
                return false;
            }
        } else {
            if (!valid_index(edge, static_cast<int>(edge_count))
                || !valid_index(input2, plan.node_count)
                || initialized[static_cast<std::size_t>(input2)] == 0) {
                return false;
            }
            operation_edges.push_back(edge);
            if (opcode == static_cast<int>(RVineOpcode::H_PAIR)) {
                if (!valid_index(output2, plan.node_count)) {
                    return false;
                }
            } else if (
                (opcode == static_cast<int>(RVineOpcode::H)
                 || opcode == static_cast<int>(RVineOpcode::H_INV))
                && output2 == -1) {
                // One-output edge operations use the common sentinel.
            } else {
                return false;
            }
        }
        initialized[static_cast<std::size_t>(output1)] = 1;
        if (output2 != -1) {
            initialized[static_cast<std::size_t>(output2)] = 1;
        }
    }
    if (!matches_used_edges(operation_edges, plan.used_edges)) {
        return false;
    }
    for (int node : plan.output_nodes) {
        if (!valid_index(node, plan.node_count)
            || initialized[static_cast<std::size_t>(node)] == 0) {
            return false;
        }
    }
    return true;
}

bool validate_density_plan(
    const RVineDensityPlan& plan,
    std::size_t edge_count) noexcept {
    const std::size_t operation_count = plan.edge_indices.size();
    if (plan.dimension < 2 || plan.node_count < plan.dimension
        || plan.input_nodes.size()
            != static_cast<std::size_t>(plan.dimension)
        || !same_operation_size(operation_count, plan.input1_nodes)
        || !same_operation_size(operation_count, plan.input2_nodes)
        || !same_operation_size(operation_count, plan.output1_nodes)
        || !same_operation_size(operation_count, plan.output2_nodes)
        || !same_operation_size(operation_count, plan.transposed)) {
        return false;
    }
    std::vector<unsigned char> initialized(
        static_cast<std::size_t>(plan.node_count), 0);
    for (int node : plan.input_nodes) {
        if (!valid_index(node, plan.node_count)
            || initialized[static_cast<std::size_t>(node)] != 0) {
            return false;
        }
        initialized[static_cast<std::size_t>(node)] = 1;
    }
    for (std::size_t index = 0; index < operation_count; ++index) {
        const int input1 = plan.input1_nodes[index];
        const int input2 = plan.input2_nodes[index];
        const int output1 = plan.output1_nodes[index];
        const int output2 = plan.output2_nodes[index];
        if (!valid_index(plan.edge_indices[index], static_cast<int>(edge_count))
            || !valid_index(input1, plan.node_count)
            || !valid_index(input2, plan.node_count)
            || initialized[static_cast<std::size_t>(input1)] == 0
            || initialized[static_cast<std::size_t>(input2)] == 0
            || !valid_optional_node(output1, plan.node_count)
            || !valid_optional_node(output2, plan.node_count)
            || !valid_orientation(plan.transposed[index])) {
            return false;
        }
        if (output1 != -1) {
            initialized[static_cast<std::size_t>(output1)] = 1;
        }
        if (output2 != -1) {
            initialized[static_cast<std::size_t>(output2)] = 1;
        }
    }
    if (!matches_used_edges(plan.edge_indices, plan.used_edges)) {
        return false;
    }
    for (int node : plan.residual_nodes) {
        if (!valid_index(node, plan.node_count)
            || initialized[static_cast<std::size_t>(node)] == 0) {
            return false;
        }
    }
    return true;
}

int prepare_edges(
    const std::vector<EdgeSpec>& edges,
    std::vector<PreparedEdge>& prepared) {
    prepared.clear();
    prepared.reserve(edges.size());
    for (const EdgeSpec& edge : edges) {
        if (!is_supported(edge.copula) || edge.copula.dim != 2) {
            prepared.clear();
            return SCAR_INVALID_FAMILY;
        }
        prepared.push_back({
            edge,
            scar_internal::transposed_copula_spec(edge.copula),
        });
    }
    return SCAR_OK;
}

int validate_parameter_pack(
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters) noexcept {
    if (parameters.n_rows < 0 || parameters.row_parameter_columns < 0) {
        return SCAR_INVALID_SIZE;
    }
    const auto rows = static_cast<std::size_t>(parameters.n_rows);
    const auto columns = static_cast<std::size_t>(
        parameters.row_parameter_columns);
    if ((parameters.scalar_parameters.size() > 0
         && parameters.scalar_parameters.data() == nullptr)
        || (rows > 0 && columns > 0
            && parameters.row_parameters.data() == nullptr)
        || (columns > 0
            && rows > std::numeric_limits<std::size_t>::max() / columns)
        || parameters.row_parameters.size() != rows * columns) {
        return SCAR_INVALID_SIZE;
    }
    for (std::size_t index = 0;
         index < parameters.scalar_parameters.size(); ++index) {
        if (!std::isfinite(parameters.scalar_parameters[index])) {
            return SCAR_INVALID_PARAMETER;
        }
    }
    for (std::size_t index = 0;
         index < parameters.row_parameters.size(); ++index) {
        if (!std::isfinite(parameters.row_parameters[index])) {
            return SCAR_INVALID_PARAMETER;
        }
    }
    for (const EdgeSpec& edge : edges) {
        switch (edge.parameter_source) {
        case ParameterSource::None:
            if (!edge.parameter_free || edge.parameter_index != -1) {
                return SCAR_INVALID_PARAMETER;
            }
            break;
        case ParameterSource::Scalar:
            if (edge.parameter_free
                || !valid_index(
                    edge.parameter_index,
                    static_cast<int>(parameters.scalar_parameters.size()))) {
                return SCAR_INVALID_PARAMETER;
            }
            break;
        case ParameterSource::RowPath:
            if (edge.parameter_free
                || !valid_index(
                    edge.parameter_index,
                    static_cast<int>(columns))) {
                return SCAR_INVALID_PARAMETER;
            }
            break;
        default:
            return SCAR_INVALID_PARAMETER;
        }
    }
    return SCAR_OK;
}

double parameter_at(
    const EdgeSpec& edge,
    const ParameterPack& parameters,
    std::int64_t row) noexcept {
    switch (edge.parameter_source) {
    case ParameterSource::None:
        return 0.0;
    case ParameterSource::Scalar:
        return parameters.scalar_parameters[
            static_cast<std::size_t>(edge.parameter_index)];
    case ParameterSource::RowPath:
        return parameters.row_parameters[
            static_cast<std::size_t>(row)
                * static_cast<std::size_t>(parameters.row_parameter_columns)
            + static_cast<std::size_t>(edge.parameter_index)];
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double clip_open_unit(double value) noexcept {
    return scar_internal::clip_pseudo_observation(value);
}

double h(
    const PreparedEdge& edge,
    bool transposed,
    double value,
    double partner,
    double parameter) {
    const CopulaSpec& copula = transposed
        ? edge.transposed_copula
        : edge.edge.copula;
    return scar_internal::copula_h_rotated(
        copula, value, partner, parameter);
}

void h_pair(
    const PreparedEdge& edge,
    bool first_transposed,
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next) {
    first_next = h(
        edge, first_transposed, first, second, parameter);
    second_next = h(
        edge, !first_transposed, second, first, parameter);
}

double h_inverse(
    const PreparedEdge& edge,
    bool transposed,
    double quantile,
    double given,
    double parameter) {
    const CopulaSpec& copula = transposed
        ? edge.transposed_copula
        : edge.edge.copula;
    return scar_internal::copula_h_inverse_rotated(
        copula, quantile, given, parameter);
}

}  // namespace scar::rvine
