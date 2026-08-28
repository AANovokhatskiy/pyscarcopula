#include "scar/rvine.hpp"

#include "density_internal.hpp"

#include "scar/core/threading.hpp"
#include "scar/detail/copula/common.hpp"
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

template <typename Plan>
bool same_pair_operation_sizes(
    const Plan& plan,
    std::size_t operation_count) {
    return same_operation_size(operation_count, plan.edge_indices)
        && same_operation_size(operation_count, plan.input1_nodes)
        && same_operation_size(operation_count, plan.input2_nodes)
        && same_operation_size(operation_count, plan.output1_nodes)
        && same_operation_size(operation_count, plan.output2_nodes)
        && same_operation_size(operation_count, plan.transposed);
}

bool matches_used_edges(
    const std::vector<int>& operation_edges,
    const std::vector<int>& used_edges) {
    std::vector<int> expected = operation_edges;
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    return expected == used_edges;
}

bool bounded_traversal_node_count(const RVineTraversalPlan& plan) noexcept {
    if (plan.dimension < 0 || plan.node_count < 0) {
        return false;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    std::size_t bound = static_cast<std::size_t>(plan.dimension);
    if (plan.inverse_edges.size() > maximum - bound) {
        return false;
    }
    bound += plan.inverse_edges.size();
    if (plan.forward_edges.size() > (maximum - bound) / 2U) {
        return false;
    }
    bound += 2U * plan.forward_edges.size();
    return static_cast<std::size_t>(plan.node_count) <= bound;
}

bool bounded_conditional_node_count(
    const RVineConditionalPlan& plan) noexcept {
    if (plan.dimension < 0 || plan.node_count < 0) {
        return false;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    std::size_t bound = plan.given_nodes.size();
    if (plan.uniform_nodes.size() > maximum - bound) {
        return false;
    }
    bound += plan.uniform_nodes.size();
    if (plan.opcodes.size() > (maximum - bound) / 2U) {
        return false;
    }
    bound += 2U * plan.opcodes.size();
    return static_cast<std::size_t>(plan.node_count) <= bound;
}

bool bounded_density_node_count(const RVineDensityPlan& plan) noexcept {
    if (plan.dimension < 0 || plan.node_count < 0) {
        return false;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    std::size_t bound = static_cast<std::size_t>(plan.dimension);
    if (plan.edge_indices.size() > (maximum - bound) / 2U) {
        return false;
    }
    bound += 2U * plan.edge_indices.size();
    return static_cast<std::size_t>(plan.node_count) <= bound;
}

bool valid_parameter_for_family(
    const EdgeSpec& edge,
    double parameter) noexcept {
    if (edge.copula.family == CopulaFamily::Independent) {
        return true;
    }
    switch (edge.copula.family) {
    case CopulaFamily::Clayton:
    case CopulaFamily::Gumbel:
    case CopulaFamily::Frank:
    case CopulaFamily::Joe:
    case CopulaFamily::Gaussian:
        return std::isfinite(scar_internal::copula_param_to_tau(
            edge.copula, parameter));
    default:
        // Unsupported families are reported by prepare_edges as such.
        return true;
    }
}

}  // namespace

bool valid_index(int value, int limit) noexcept {
    return value >= 0 && value < limit;
}

bool validate_traversal_plan(
    const RVineTraversalPlan& plan,
    std::size_t edge_count) {
    const std::size_t column_count = plan.column_uniforms.size();
    if (plan.dimension < 2 || plan.node_count < plan.dimension
        || !valid_index(plan.last_uniform_column, plan.dimension)
        || !valid_index(plan.last_output_node, plan.node_count)
        || plan.output_nodes.size() != static_cast<std::size_t>(plan.dimension)
        || plan.inverse_offsets.size() != column_count + 1
        || plan.forward_offsets.size() != column_count + 1
        || !bounded_traversal_node_count(plan)
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

    if (edge_count == 0) {
        if (plan.node_count != plan.dimension
            || column_count
                != static_cast<std::size_t>(plan.dimension - 1)
            || plan.last_uniform_column != plan.dimension - 1
            || plan.last_output_node != plan.dimension - 1
            || !plan.inverse_edges.empty()
            || !plan.forward_edges.empty()) {
            return false;
        }
        for (int variable = 0; variable < plan.dimension; ++variable) {
            if (plan.output_nodes[static_cast<std::size_t>(variable)]
                    != variable) {
                return false;
            }
        }
        for (std::size_t column = 0; column < column_count; ++column) {
            if (plan.column_uniforms[column]
                    != plan.dimension - 2 - static_cast<int>(column)) {
                return false;
            }
        }
        return true;
    }

    // The topology builder validates initialization order, but the native
    // boundary repeats it so malformed direct calls cannot read NaN sentinel
    // nodes or rely on unchecked plan topology.
    std::vector<unsigned char> initialized(
        static_cast<std::size_t>(plan.node_count), 0);
    initialized[static_cast<std::size_t>(plan.last_output_node)] = 1;
    for (std::size_t column = 0; column < column_count; ++column) {
        for (int index = plan.inverse_offsets[column];
             index < plan.inverse_offsets[column + 1]; ++index) {
            const int partner = plan.inverse_partner_nodes[index];
            if (initialized[static_cast<std::size_t>(partner)] == 0) {
                return false;
            }
            initialized[static_cast<std::size_t>(
                plan.inverse_output_nodes[index])] = 1;
        }
        for (int index = plan.forward_offsets[column];
             index < plan.forward_offsets[column + 1]; ++index) {
            const int leaf = plan.forward_leaf_nodes[index];
            const int partner = plan.forward_partner_nodes[index];
            if (initialized[static_cast<std::size_t>(leaf)] == 0
                || initialized[static_cast<std::size_t>(partner)] == 0) {
                return false;
            }
            initialized[static_cast<std::size_t>(
                plan.forward_leaf_output_nodes[index])] = 1;
            initialized[static_cast<std::size_t>(
                plan.forward_partner_output_nodes[index])] = 1;
        }
    }
    for (int node : plan.output_nodes) {
        if (initialized[static_cast<std::size_t>(node)] == 0) {
            return false;
        }
    }
    for (std::size_t index = 0; index < edge_count; ++index) {
        if (initialized[static_cast<std::size_t>(
                plan.update_u1_nodes[index])] == 0
            || initialized[static_cast<std::size_t>(
                plan.update_u2_nodes[index])] == 0) {
            return false;
        }
    }
    return true;
}

bool validate_conditional_plan(
    const RVineConditionalPlan& plan,
    std::size_t edge_count) {
    const std::size_t operation_count = plan.opcodes.size();
    if (plan.dimension < 2 || plan.node_count < plan.dimension
        || !bounded_conditional_node_count(plan)
        || plan.output_nodes.size()
            != static_cast<std::size_t>(plan.dimension)
        || plan.given_variables.size() != plan.given_nodes.size()
        || plan.node_sources.size()
            != static_cast<std::size_t>(plan.node_count)
        || plan.node_source_indices.size()
            != static_cast<std::size_t>(plan.node_count)
        || !same_pair_operation_sizes(plan, operation_count)) {
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
    std::size_t edge_count) {
    const std::size_t operation_count = plan.edge_indices.size();
    if (plan.dimension < 2 || plan.node_count < plan.dimension
        || !bounded_density_node_count(plan)
        || plan.input_nodes.size()
            != static_cast<std::size_t>(plan.dimension)
        || !same_pair_operation_sizes(plan, operation_count)) {
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
        const bool has_first_output = output1 != -1;
        const bool has_second_output = output2 != -1;
        if (!valid_index(plan.edge_indices[index], static_cast<int>(edge_count))
            || !valid_index(input1, plan.node_count)
            || !valid_index(input2, plan.node_count)
            || initialized[static_cast<std::size_t>(input1)] == 0
            || initialized[static_cast<std::size_t>(input2)] == 0
            || !valid_optional_node(output1, plan.node_count)
            || !valid_optional_node(output2, plan.node_count)
            || has_first_output != has_second_output
            || !valid_orientation(plan.transposed[index])) {
            return false;
        }
        if (output1 != -1) {
            if (initialized[static_cast<std::size_t>(output1)] != 0
                || initialized[static_cast<std::size_t>(output2)] != 0
                || output1 == output2) {
                return false;
            }
            initialized[static_cast<std::size_t>(output1)] = 1;
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

    const auto dimension = static_cast<std::size_t>(plan.dimension);
    if (plan.affected_operation_offsets.size() != dimension + 1U
        || plan.affected_node_offsets.size() != dimension + 1U
        || plan.affected_operation_offsets.front() != 0
        || plan.affected_node_offsets.front() != 0
        || plan.affected_operation_offsets.back()
            != static_cast<int>(plan.affected_operations.size())
        || plan.affected_node_offsets.back()
            != static_cast<int>(plan.affected_nodes.size())) {
        return false;
    }
    for (std::size_t variable = 0; variable < dimension; ++variable) {
        const int operation_begin = plan.affected_operation_offsets[variable];
        const int operation_end =
            plan.affected_operation_offsets[variable + 1U];
        const int node_begin = plan.affected_node_offsets[variable];
        const int node_end = plan.affected_node_offsets[variable + 1U];
        if (operation_begin < 0 || operation_end < operation_begin
            || node_begin < 0 || node_end <= node_begin
            || operation_end
                > static_cast<int>(plan.affected_operations.size())
            || node_end > static_cast<int>(plan.affected_nodes.size())
            || plan.affected_nodes[static_cast<std::size_t>(node_begin)]
                != plan.input_nodes[variable]) {
            return false;
        }

        std::vector<unsigned char> affected(
            static_cast<std::size_t>(plan.node_count), 0);
        affected[static_cast<std::size_t>(plan.input_nodes[variable])] = 1;
        int expected_operation_position = operation_begin;
        int expected_node_position = node_begin + 1;
        for (std::size_t operation = 0;
             operation < operation_count; ++operation) {
            const bool operation_is_affected =
                affected[static_cast<std::size_t>(
                    plan.input1_nodes[operation])] != 0
                || affected[static_cast<std::size_t>(
                    plan.input2_nodes[operation])] != 0;
            if (!operation_is_affected) {
                continue;
            }
            if (expected_operation_position >= operation_end
                || plan.affected_operations[static_cast<std::size_t>(
                    expected_operation_position)]
                    != static_cast<int>(operation)) {
                return false;
            }
            ++expected_operation_position;
            const int output1 = plan.output1_nodes[operation];
            if (output1 == -1) {
                continue;
            }
            const int output2 = plan.output2_nodes[operation];
            if (expected_node_position + 1 >= node_end
                || plan.affected_nodes[static_cast<std::size_t>(
                    expected_node_position)] != output1
                || plan.affected_nodes[static_cast<std::size_t>(
                    expected_node_position + 1)] != output2) {
                return false;
            }
            expected_node_position += 2;
            affected[static_cast<std::size_t>(output1)] = 1;
            affected[static_cast<std::size_t>(output2)] = 1;
        }
        if (expected_operation_position != operation_end
            || expected_node_position != node_end) {
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
        const PreparedPairKernel kernel(edge.copula);
        if (!kernel.is_supported() || edge.copula.dim != 2) {
            prepared.clear();
            return SCAR_INVALID_FAMILY;
        }
        prepared.emplace_back(
            edge,
            scar_internal::transposed_copula_spec(edge.copula));
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
    std::size_t row_parameter_count = 0;
    if ((parameters.scalar_parameters.size() > 0
         && parameters.scalar_parameters.data() == nullptr)
        || (rows > 0 && columns > 0
            && parameters.row_parameters.data() == nullptr)
        || !scar_internal::checked_shape_size(
            rows, columns, row_parameter_count)
        || parameters.row_parameters.size() != row_parameter_count) {
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
        // A fitted IndependentResult is authoritative even when the retained
        // retained source descriptor belongs to another family. The reverse is not
        // valid: an Independent family must never request a parameter.
        if (edge.copula.family == CopulaFamily::Independent
            && !edge.parameter_free) {
            return SCAR_INVALID_PARAMETER;
        }
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
                    static_cast<int>(parameters.scalar_parameters.size()))
                || !valid_parameter_for_family(
                    edge,
                    parameters.scalar_parameters[static_cast<std::size_t>(
                        edge.parameter_index)])) {
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
            for (std::size_t row = 0; row < rows; ++row) {
                const double parameter = parameters.row_parameters[
                    row * columns
                    + static_cast<std::size_t>(edge.parameter_index)];
                if (!valid_parameter_for_family(edge, parameter)) {
                    return SCAR_INVALID_PARAMETER;
                }
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
    const PreparedPairKernel& kernel = transposed
        ? edge.transposed_kernel
        : edge.kernel;
    return kernel.h(value, partner, parameter);
}

void h_pair(
    const PreparedEdge& edge,
    bool first_transposed,
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next) {
    const PreparedPairKernel& first_kernel = first_transposed
        ? edge.transposed_kernel
        : edge.kernel;
    const PreparedPairKernel& second_kernel = first_transposed
        ? edge.kernel
        : edge.transposed_kernel;
    if (first_kernel.is_unrotated_gaussian()
        && second_kernel.is_unrotated_gaussian()) {
        first_kernel.h_pair(
            clip_open_unit(first),
            clip_open_unit(second),
            parameter,
            first_next,
            second_next);
        return;
    }
    first_next = first_kernel.h(first, second, parameter);
    second_next = second_kernel.h(second, first, parameter);
}

double h_inverse(
    const PreparedEdge& edge,
    bool transposed,
    double quantile,
    double given,
    double parameter) {
    const PreparedPairKernel& kernel = transposed
        ? edge.transposed_kernel
        : edge.kernel;
    return kernel.inverse_h(quantile, given, parameter);
}

namespace detail {

double edge_log_pdf(
    const PreparedEdge& edge,
    bool transposed,
    double first,
    double second,
    double parameter) {
    const PreparedPairKernel& kernel = transposed
        ? edge.transposed_kernel
        : edge.kernel;
    return kernel.log_pdf(first, second, parameter);
}

}  // namespace detail

namespace {

void fail_sample(
    SampleResult& out,
    int status,
    std::int64_t row,
    int edge,
    int operation) noexcept {
    out.status = status_from_int(status);
    out.failure.row = row;
    out.failure.edge = edge;
    out.failure.operation = operation;
}

void fail_conditional_sample(
    ConditionalSampleResult& out,
    int status,
    std::int64_t row,
    int edge,
    int operation) noexcept {
    out.status = status_from_int(status);
    out.failure.row = row;
    out.failure.edge = edge;
    out.failure.operation = operation;
}

bool is_unrotated_gaussian(const PreparedEdge& edge) noexcept {
    return edge.kernel.is_unrotated_gaussian()
        && edge.transposed_kernel.is_unrotated_gaussian();
}

double gaussian_inverse_from_quantiles(
    double quantile,
    double given,
    double rho) noexcept {
    const double clipped_rho = std::min(std::max(rho, -0.999999), 0.999999);
    const double z = quantile
        * std::sqrt(1.0 - clipped_rho * clipped_rho)
        + clipped_rho * given;
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

}  // namespace

SampleResult sample(
    const RVineTraversalPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView uniforms,
    std::int64_t uniform_rows,
    std::int64_t uniform_columns,
    int n_threads) {
    SampleResult out;
    out.n_rows = uniform_rows;
    out.dimension = plan.dimension;
    out.n_threads_requested = n_threads;
    const std::size_t edge_count = edges.size();
    if (!scar_internal::valid_thread_count(n_threads) || uniform_rows < 0
        || uniform_columns != plan.dimension
        || parameters.n_rows != uniform_rows
        || !validate_traversal_plan(plan, edge_count)) {
        fail_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }

    const auto rows = static_cast<std::size_t>(uniform_rows);
    const auto dimension = static_cast<std::size_t>(plan.dimension);
    std::size_t uniform_values = 0;
    std::size_t output_values = 0;
    if (!scar_internal::checked_shape_size(
            rows, dimension, uniform_values)
        || !scar_internal::checked_shape_size(
            rows, dimension, output_values)
        || uniforms.size() != uniform_values
        || (uniform_values > 0 && uniforms.data() == nullptr)) {
        fail_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    const int parameter_status = validate_parameter_pack(edges, parameters);
    if (parameter_status != SCAR_OK) {
        fail_sample(out, parameter_status, -1, -1, -1);
        return out;
    }
    std::vector<PreparedEdge> prepared_edges;
    const int prepare_status = prepare_edges(edges, prepared_edges);
    if (prepare_status != SCAR_OK) {
        fail_sample(out, prepare_status, -1, -1, -1);
        return out;
    }
    for (std::size_t index = 0; index < uniform_values; ++index) {
        if (!std::isfinite(uniforms[index])
            || uniforms[index] <= 0.0 || uniforms[index] >= 1.0) {
            const auto row = static_cast<std::int64_t>(index / dimension);
            fail_sample(out, SCAR_INVALID_PARAMETER, row, -1, -1);
            return out;
        }
    }

    out.values.assign(output_values, 0.0);
    if (rows == 0) {
        return out;
    }
    if (edge_count == 0) {
        std::copy(
            uniforms.data(), uniforms.data() + uniform_values,
            out.values.begin());
        return out;
    }
    std::vector<double> nodes(
        static_cast<std::size_t>(plan.node_count),
        std::numeric_limits<double>::quiet_NaN());

    for (std::int64_t row = 0; row < uniform_rows; ++row) {
        std::fill(
            nodes.begin(), nodes.end(),
            std::numeric_limits<double>::quiet_NaN());
        const double* uniform_row = uniforms.data()
            + static_cast<std::size_t>(row) * dimension;
        nodes[static_cast<std::size_t>(plan.last_output_node)] =
            clip_open_unit(uniform_row[plan.last_uniform_column]);

        int operation = 0;
        for (std::size_t column = 0;
             column < plan.column_uniforms.size(); ++column) {
            double current = clip_open_unit(
                uniform_row[plan.column_uniforms[column]]);
            for (int index = plan.inverse_offsets[column];
                 index < plan.inverse_offsets[column + 1];
                 ++index, ++operation) {
                const int edge_index = plan.inverse_edges[index];
                const PreparedEdge& edge = prepared_edges[
                    static_cast<std::size_t>(edge_index)];
                if (edge.edge.parameter_free) {
                    nodes[static_cast<std::size_t>(
                        plan.inverse_output_nodes[index])] = current;
                    ++out.independence_fast_paths;
                    ++out.inverse_operations;
                    continue;
                }
                const double partner = nodes[static_cast<std::size_t>(
                    plan.inverse_partner_nodes[index])];
                const double parameter = parameter_at(
                    edge.edge, parameters, row);
                current = h_inverse(
                    edge,
                    plan.inverse_transposed[index] != 0,
                    current,
                    partner,
                    parameter);
                if (!std::isfinite(current)) {
                    fail_sample(
                        out, SCAR_NUMERICAL_FAILURE, row, edge_index,
                        operation);
                    return out;
                }
                current = clip_open_unit(current);
                nodes[static_cast<std::size_t>(
                    plan.inverse_output_nodes[index])] = current;
                ++out.inverse_operations;
            }

            for (int index = plan.forward_offsets[column];
                 index < plan.forward_offsets[column + 1];
                 ++index, ++operation) {
                const int edge_index = plan.forward_edges[index];
                const PreparedEdge& edge = prepared_edges[
                    static_cast<std::size_t>(edge_index)];
                const double leaf = nodes[static_cast<std::size_t>(
                    plan.forward_leaf_nodes[index])];
                const double partner = nodes[static_cast<std::size_t>(
                    plan.forward_partner_nodes[index])];
                double leaf_next = leaf;
                double partner_next = partner;
                if (edge.edge.parameter_free) {
                    ++out.independence_fast_paths;
                } else {
                    const double parameter = parameter_at(
                        edge.edge, parameters, row);
                    h_pair(
                        edge,
                        plan.forward_transposed[index] != 0,
                        leaf,
                        partner,
                        parameter,
                        leaf_next,
                        partner_next);
                    if (!std::isfinite(leaf_next)
                        || !std::isfinite(partner_next)) {
                        fail_sample(
                            out, SCAR_NUMERICAL_FAILURE, row, edge_index,
                            operation);
                        return out;
                    }
                }
                nodes[static_cast<std::size_t>(
                    plan.forward_leaf_output_nodes[index])] =
                    clip_open_unit(leaf_next);
                nodes[static_cast<std::size_t>(
                    plan.forward_partner_output_nodes[index])] =
                    clip_open_unit(partner_next);
                ++out.forward_operations;
            }
        }

        double* result_row = out.values.data()
            + static_cast<std::size_t>(row) * dimension;
        for (int variable = 0; variable < plan.dimension; ++variable) {
            const double value = nodes[static_cast<std::size_t>(
                plan.output_nodes[static_cast<std::size_t>(variable)])];
            if (!std::isfinite(value)) {
                fail_sample(
                    out, SCAR_NUMERICAL_FAILURE, row, -1, operation);
                return out;
            }
            result_row[variable] = clip_open_unit(value);
        }
    }
    return out;
}

ConditionalSampleResult conditional_sample(
    const RVineConditionalPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView given_values,
    DoubleView uniforms,
    std::int64_t uniform_rows,
    std::int64_t uniform_columns,
    int n_threads) {
    ConditionalSampleResult out;
    out.n_rows = uniform_rows;
    out.dimension = plan.dimension;
    out.n_threads_requested = n_threads;
    const std::size_t edge_count = edges.size();
    if (!scar_internal::valid_thread_count(n_threads) || uniform_rows < 0
        || uniform_columns != plan.dimension
        || parameters.n_rows != uniform_rows
        || !validate_conditional_plan(plan, edge_count)) {
        fail_conditional_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }

    const auto rows = static_cast<std::size_t>(uniform_rows);
    const auto dimension = static_cast<std::size_t>(plan.dimension);
    std::size_t uniform_value_count = 0;
    std::size_t output_value_count = 0;
    if (!scar_internal::checked_shape_size(
            rows, dimension, uniform_value_count)
        || !scar_internal::checked_shape_size(
            rows, dimension, output_value_count)
        || given_values.size() != dimension
        || uniforms.size() != uniform_value_count
        || (dimension > 0 && given_values.data() == nullptr)
        || (uniform_value_count > 0 && uniforms.data() == nullptr)) {
        fail_conditional_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    const int parameter_status = validate_parameter_pack(edges, parameters);
    if (parameter_status != SCAR_OK) {
        fail_conditional_sample(out, parameter_status, -1, -1, -1);
        return out;
    }
    std::vector<PreparedEdge> prepared_edges;
    const int prepare_status = prepare_edges(edges, prepared_edges);
    if (prepare_status != SCAR_OK) {
        fail_conditional_sample(out, prepare_status, -1, -1, -1);
        return out;
    }
    for (std::size_t variable = 0; variable < dimension; ++variable) {
        const double value = given_values[variable];
        if (!std::isfinite(value) || value <= 0.0 || value >= 1.0) {
            fail_conditional_sample(
                out, SCAR_INVALID_PARAMETER, -1, -1, -1);
            return out;
        }
    }
    for (std::size_t index = 0; index < uniform_value_count; ++index) {
        if (!std::isfinite(uniforms[index])
            || uniforms[index] <= 0.0 || uniforms[index] >= 1.0) {
            const auto row = static_cast<std::int64_t>(index / dimension);
            fail_conditional_sample(
                out, SCAR_INVALID_PARAMETER, row, -1, -1);
            return out;
        }
    }

    out.values.assign(output_value_count, 0.0);
    if (rows == 0) {
        return out;
    }
    // Preserve the Stage 0 row-chunk contract.  Any future tuning of this
    // constant requires an explicitly approved Gate 1/Gate 3 change.
    constexpr std::size_t conditional_max_block_rows = 1024;
    constexpr std::size_t conditional_workspace_budget =
        64U * 1024U * 1024U;
    constexpr std::size_t bytes_per_node_value =
        2U * sizeof(double) + sizeof(unsigned char);
    const std::size_t node_count = static_cast<std::size_t>(plan.node_count);
    std::size_t bytes_per_row = 0;
    if (!scar_internal::checked_size_mul(
            node_count, bytes_per_node_value, bytes_per_row)) {
        fail_conditional_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    if (bytes_per_row > conditional_workspace_budget) {
        fail_conditional_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    const std::size_t memory_limited_rows = bytes_per_row == 0
        ? conditional_max_block_rows
        : conditional_workspace_budget / bytes_per_row;
    const std::size_t block_capacity = std::min(
        rows,
        std::min(conditional_max_block_rows, memory_limited_rows));
    std::size_t node_value_count = 0;
    std::size_t peak_workspace_bytes = 0;
    if (!scar_internal::checked_size_mul(
            block_capacity, node_count, node_value_count)
        || !scar_internal::checked_size_mul(
            block_capacity, bytes_per_row, peak_workspace_bytes)) {
        fail_conditional_sample(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    out.max_block_rows = static_cast<std::uint64_t>(block_capacity);
    out.peak_workspace_bytes = static_cast<std::uint64_t>(
        peak_workspace_bytes);

    // Conditional plans operate on bounded row blocks. Keeping each node's
    // block contiguous hoists opcode/family dispatch out of the hot row loop
    // while bounding workspace independently of the public request size.
    const double missing = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> nodes(node_value_count, missing);
    std::vector<double> gaussian_quantiles(node_value_count, 0.0);
    std::vector<unsigned char> gaussian_quantile_valid(node_value_count, 0);
    const auto node_offset = [block_capacity](int node) noexcept {
        return static_cast<std::size_t>(node) * block_capacity;
    };

    for (std::size_t row_start = 0; row_start < rows;
         row_start += block_capacity) {
        const std::size_t block_rows = std::min(
            block_capacity, rows - row_start);
        ++out.row_blocks;
        std::fill(nodes.begin(), nodes.end(), missing);
        std::fill(
            gaussian_quantile_valid.begin(),
            gaussian_quantile_valid.end(),
            static_cast<unsigned char>(0));
        const auto copy_node_state = [
                &nodes,
                &gaussian_quantiles,
                &gaussian_quantile_valid,
                block_rows](
                    std::size_t input_offset,
                    std::size_t output_offset) {
            std::copy_n(
                nodes.data() + input_offset,
                block_rows,
                nodes.data() + output_offset);
            std::copy_n(
                gaussian_quantiles.data() + input_offset,
                block_rows,
                gaussian_quantiles.data() + output_offset);
            std::copy_n(
                gaussian_quantile_valid.data() + input_offset,
                block_rows,
                gaussian_quantile_valid.data() + output_offset);
        };

    for (int node = 0; node < plan.node_count; ++node) {
        const std::size_t plan_position = static_cast<std::size_t>(node);
        const int source = plan.node_sources[plan_position];
        const int source_index = plan.node_source_indices[plan_position];
        double* destination = nodes.data() + node_offset(node);
        if (source == static_cast<int>(RVineNodeSource::Given)) {
            std::fill(
                destination,
                destination + block_rows,
                given_values[static_cast<std::size_t>(source_index)]);
        } else if (source == static_cast<int>(RVineNodeSource::Uniform)) {
            for (std::size_t row = 0; row < block_rows; ++row) {
                destination[row] = uniforms[
                    (row_start + row) * dimension
                    + static_cast<std::size_t>(source_index)];
            }
        }
    }

    for (std::size_t operation = 0;
         operation < plan.opcodes.size(); ++operation) {
        const int opcode = plan.opcodes[operation];
        const int input1 = plan.input1_nodes[operation];
        const int output1 = plan.output1_nodes[operation];
        const std::size_t input1_offset = node_offset(input1);
        const std::size_t output1_offset = node_offset(output1);
        if (opcode == static_cast<int>(RVineOpcode::COPY)) {
            copy_node_state(input1_offset, output1_offset);
            out.copy_operations += static_cast<std::int64_t>(block_rows);
            continue;
        }

        const int edge_index = plan.edge_indices[operation];
        const PreparedEdge& edge = prepared_edges[
            static_cast<std::size_t>(edge_index)];
        const int input2 = plan.input2_nodes[operation];
        const int output2 = plan.output2_nodes[operation];
        const std::size_t input2_offset = node_offset(input2);
        const std::size_t output2_offset = opcode
                == static_cast<int>(RVineOpcode::H_PAIR)
            ? node_offset(output2)
            : 0;
        const bool transposed = plan.transposed[operation] != 0;

        if (edge.edge.parameter_free) {
            copy_node_state(input1_offset, output1_offset);
            if (opcode == static_cast<int>(RVineOpcode::H_PAIR)) {
                copy_node_state(input2_offset, output2_offset);
                out.h_pair_operations += static_cast<std::int64_t>(
                    block_rows);
            } else if (opcode == static_cast<int>(RVineOpcode::H)) {
                out.h_operations += static_cast<std::int64_t>(block_rows);
            } else {
                out.inverse_operations += static_cast<std::int64_t>(
                    block_rows);
            }
            out.independence_fast_paths += static_cast<std::int64_t>(
                block_rows);
            continue;
        }

        const bool gaussian = is_unrotated_gaussian(edge);
        for (std::size_t row = 0; row < block_rows; ++row) {
            const double first = nodes[input1_offset + row];
            const double second = nodes[input2_offset + row];
            const double parameter = parameter_at(
                edge.edge,
                parameters,
                static_cast<std::int64_t>(row_start + row));
            double first_next = missing;
            double second_next = missing;
            if (gaussian) {
                const auto quantile_at = [&](std::size_t position) {
                    if (gaussian_quantile_valid[position] == 0) {
                        gaussian_quantiles[position] =
                            edge.kernel.prepare_conditional_value(
                                nodes[position]);
                        gaussian_quantile_valid[position] = 1;
                    }
                    return gaussian_quantiles[position];
                };
                const double first_quantile = quantile_at(input1_offset + row);
                const double second_quantile = quantile_at(input2_offset + row);
                if (opcode == static_cast<int>(RVineOpcode::H)) {
                    first_next = edge.kernel.h_from_prepared_values(
                        first_quantile, second_quantile, parameter);
                } else if (opcode == static_cast<int>(RVineOpcode::H_PAIR)) {
                    edge.kernel.h_pair_from_prepared_values(
                        first_quantile,
                        second_quantile,
                        parameter,
                        first_next,
                        second_next);
                } else {
                    first_next = gaussian_inverse_from_quantiles(
                        first_quantile, second_quantile, parameter);
                }
            } else if (opcode == static_cast<int>(RVineOpcode::H)) {
                first_next = h(
                    edge, transposed, first, second, parameter);
            } else if (opcode == static_cast<int>(RVineOpcode::H_PAIR)) {
                h_pair(
                    edge,
                    transposed,
                    first,
                    second,
                    parameter,
                    first_next,
                    second_next);
            } else {
                first_next = h_inverse(
                    edge, transposed, first, second, parameter);
            }
            if (!std::isfinite(first_next)
                || (opcode == static_cast<int>(RVineOpcode::H_PAIR)
                    && !std::isfinite(second_next))) {
                fail_conditional_sample(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(row_start + row),
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            nodes[output1_offset + row] = clip_open_unit(first_next);
            gaussian_quantile_valid[output1_offset + row] = 0;
            if (opcode == static_cast<int>(RVineOpcode::H_PAIR)) {
                nodes[output2_offset + row] = clip_open_unit(second_next);
                gaussian_quantile_valid[output2_offset + row] = 0;
            }
        }
        if (opcode == static_cast<int>(RVineOpcode::H)) {
            out.h_operations += static_cast<std::int64_t>(block_rows);
        } else if (opcode == static_cast<int>(RVineOpcode::H_PAIR)) {
            out.h_pair_operations += static_cast<std::int64_t>(block_rows);
        } else {
            out.inverse_operations += static_cast<std::int64_t>(block_rows);
        }
    }

    for (std::size_t row = 0; row < block_rows; ++row) {
        const std::size_t result_index = row_start + row;
        double* result_row = out.values.data()
            + result_index * dimension;
        for (int variable = 0; variable < plan.dimension; ++variable) {
            const int output_node = plan.output_nodes[
                static_cast<std::size_t>(variable)];
            const double value = nodes[
                node_offset(output_node)
                + row];
            if (!std::isfinite(value)) {
                fail_conditional_sample(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(result_index),
                    -1,
                    static_cast<int>(plan.opcodes.size()));
                return out;
            }
            // Conditioning values are public outputs, not internal pseudo-
            // observations. Preserve every valid open-interval bit exactly;
            // clipping is only part of the numerical kernel contract.
            result_row[variable] = plan.node_sources[
                    static_cast<std::size_t>(output_node)]
                    == static_cast<int>(RVineNodeSource::Given)
                ? value
                : clip_open_unit(value);
        }
    }
    }
    return out;
}

}  // namespace scar::rvine
