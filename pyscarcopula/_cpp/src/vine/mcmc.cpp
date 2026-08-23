#include "scar/rvine.hpp"

#include "scar/core/threading.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace scar::rvine {
namespace {

void fail_mcmc(
    MCMCResult& out,
    int status,
    std::int64_t row,
    int edge,
    int operation) noexcept {
    out.status = status;
    out.failure_row = row;
    out.failure_edge = edge;
    out.failure_operation = operation;
}

bool validate_partition(
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_values,
    const std::vector<int>& free_indices) {
    if (given_indices.size() != given_values.size()
        || free_indices.empty()
        || given_indices.size() + free_indices.size()
            != static_cast<std::size_t>(dimension)) {
        return false;
    }
    std::vector<unsigned char> seen(
        static_cast<std::size_t>(dimension), 0);
    for (int variable : given_indices) {
        if (!valid_index(variable, dimension)
            || seen[static_cast<std::size_t>(variable)] != 0) {
            return false;
        }
        seen[static_cast<std::size_t>(variable)] = 1;
    }
    for (int variable : free_indices) {
        if (!valid_index(variable, dimension)
            || seen[static_cast<std::size_t>(variable)] != 0) {
            return false;
        }
        seen[static_cast<std::size_t>(variable)] = 1;
    }
    return true;
}

bool valid_open_uniforms(DoubleView values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])
            || values[index] <= 0.0 || values[index] >= 1.0) {
            return false;
        }
    }
    return true;
}

std::uint64_t affected_operation_count(
    const RVineDensityPlan& plan,
    int variable) noexcept {
    const auto index = static_cast<std::size_t>(variable);
    return static_cast<std::uint64_t>(
        plan.affected_operation_offsets[index + 1U]
        - plan.affected_operation_offsets[index]);
}

bool incremental_is_structurally_profitable(
    const RVineDensityPlan& plan,
    const std::vector<int>& free_indices) noexcept {
    const auto operation_count = static_cast<std::uint64_t>(
        plan.edge_indices.size());
    if (operation_count == 0) {
        return false;
    }
    // Copying cached values and summing all contributions still costs O(E).
    // Avoid incremental dispatch when every proposal also recomputes almost
    // the complete expensive traversal.
    for (int variable : free_indices) {
        if (affected_operation_count(plan, variable) * 100U
            > operation_count * 85U) {
            return false;
        }
    }
    return true;
}

bool incremental_memory_layout(
    const RVineDensityPlan& plan,
    std::size_t rows,
    std::size_t state_value_count,
    std::size_t draw_value_count,
    bool has_proposals,
    std::uint64_t memory_budget_bytes,
    std::size_t& chunk_rows,
    std::uint64_t& per_row_workspace_bytes,
    std::uint64_t& fixed_bytes) noexcept {
    const std::size_t node_count = static_cast<std::size_t>(plan.node_count);
    std::uint64_t state_bytes = 0;
    std::uint64_t log_pdf_bytes = 0;
    std::uint64_t draw_bytes = 0;
    if (!scar_internal::checked_byte_count<double>(
            state_value_count, state_bytes)
        || !scar_internal::checked_byte_count<double>(
            rows, log_pdf_bytes)
        || !scar_internal::checked_byte_count<double>(
            draw_value_count, draw_bytes)) {
        return false;
    }
    if (!scar_internal::checked_uint64_add(
            state_bytes, log_pdf_bytes, fixed_bytes)) {
        return false;
    }
    if (!has_proposals || rows == 0) {
        chunk_rows = 0;
        per_row_workspace_bytes = 0;
        return fixed_bytes <= memory_budget_bytes;
    }

    std::size_t node_and_operation_values = 0;
    std::size_t per_row_values = 0;
    const std::size_t operation_count = plan.edge_indices.size();
    if (!scar_internal::checked_size_add(
            node_count, operation_count, node_and_operation_values)
        || !scar_internal::checked_size_mul(
            node_and_operation_values, 2U, per_row_values)
        || !scar_internal::checked_byte_count<double>(
            per_row_values, per_row_workspace_bytes)) {
        return false;
    }
    std::uint64_t marker_bytes = 0;
    if (!scar_internal::checked_byte_count<int>(
            node_count, marker_bytes)) {
        return false;
    }
    if (!scar_internal::checked_uint64_add(
            per_row_workspace_bytes,
            marker_bytes,
            per_row_workspace_bytes)
        || !scar_internal::checked_uint64_add(
            draw_bytes, draw_bytes, draw_bytes)
        || !scar_internal::checked_uint64_add(
            fixed_bytes, draw_bytes, fixed_bytes)) {
        return false;
    }
    if (fixed_bytes > memory_budget_bytes
        || per_row_workspace_bytes
            > memory_budget_bytes - fixed_bytes) {
        return false;
    }
    const std::uint64_t capacity =
        (memory_budget_bytes - fixed_bytes) / per_row_workspace_bytes;
    chunk_rows = static_cast<std::size_t>(std::min<std::uint64_t>(
        static_cast<std::uint64_t>(rows), capacity));
    return chunk_rows > 0;
}

bool full_memory_layout(
    const RVineDensityPlan& plan,
    std::size_t rows,
    std::size_t state_value_count,
    std::size_t draw_value_count,
    bool has_proposals,
    std::uint64_t memory_budget_bytes,
    std::uint64_t& peak_bytes) noexcept {
    std::uint64_t state_bytes = 0;
    std::uint64_t log_pdf_bytes = 0;
    if (!scar_internal::checked_byte_count<double>(
            state_value_count, state_bytes)
        || !scar_internal::checked_byte_count<double>(
            rows, log_pdf_bytes)
        || !scar_internal::checked_uint64_add(
            state_bytes, log_pdf_bytes, peak_bytes)) {
        return false;
    }
    if (!has_proposals || rows == 0) {
        return peak_bytes <= memory_budget_bytes;
    }

    std::uint64_t draw_bytes = 0;
    std::uint64_t node_bytes = 0;
    std::uint64_t proposal_bytes = 0;
    if (!scar_internal::checked_byte_count<double>(
            draw_value_count, draw_bytes)
        || !scar_internal::checked_byte_count<double>(
            static_cast<std::size_t>(plan.node_count), node_bytes)
        || !scar_internal::checked_uint64_add(
            state_bytes, log_pdf_bytes, proposal_bytes)
        || !scar_internal::checked_uint64_add(
            peak_bytes, proposal_bytes, peak_bytes)
        || !scar_internal::checked_uint64_add(
            draw_bytes, draw_bytes, draw_bytes)
        || !scar_internal::checked_uint64_add(
            peak_bytes, draw_bytes, peak_bytes)
        || !scar_internal::checked_uint64_add(
            peak_bytes, node_bytes, peak_bytes)) {
        return false;
    }
    return peak_bytes <= memory_budget_bytes;
}

bool propagate_h_pair_nodes(
    const PreparedEdge& edge,
    bool transposed,
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next) {
    first_next = first;
    second_next = second;
    if (!edge.edge.parameter_free) {
        h_pair(
            edge,
            transposed,
            first,
            second,
            parameter,
            first_next,
            second_next);
    }
    if (!std::isfinite(first_next) || !std::isfinite(second_next)) {
        return false;
    }
    first_next = clip_open_unit(first_next);
    second_next = clip_open_unit(second_next);
    return true;
}

bool initialize_density_cache(
    const RVineDensityPlan& plan,
    const std::vector<PreparedEdge>& edges,
    const ParameterPack& parameters,
    const double* state_row,
    std::int64_t row,
    double* nodes,
    double* contributions,
    double& log_pdf,
    int& failure_edge,
    int& failure_operation) {
    const double missing = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(nodes, static_cast<std::size_t>(plan.node_count), missing);
    for (int variable = 0; variable < plan.dimension; ++variable) {
        nodes[static_cast<std::size_t>(plan.input_nodes[
            static_cast<std::size_t>(variable)])] = clip_open_unit(
                state_row[static_cast<std::size_t>(variable)]);
    }

    log_pdf = 0.0;
    for (std::size_t operation = 0;
         operation < plan.edge_indices.size(); ++operation) {
        const int edge_index = plan.edge_indices[operation];
        const PreparedEdge& edge = edges[static_cast<std::size_t>(edge_index)];
        const double first = nodes[static_cast<std::size_t>(
            plan.input1_nodes[operation])];
        const double second = nodes[static_cast<std::size_t>(
            plan.input2_nodes[operation])];
        const bool transposed = plan.transposed[operation] != 0;
        const double parameter = parameter_at(edge.edge, parameters, row);
        const double contribution = edge.edge.parameter_free
            ? 0.0
            : detail::edge_log_pdf(
                edge, transposed, first, second, parameter);
        contributions[operation] = contribution;
        if (!std::isfinite(contribution)
            || !std::isfinite(log_pdf + contribution)) {
            failure_edge = edge_index;
            failure_operation = static_cast<int>(operation);
            return false;
        }
        log_pdf += contribution;

        const int output1 = plan.output1_nodes[operation];
        if (output1 == -1) {
            continue;
        }
        double first_next = 0.0;
        double second_next = 0.0;
        if (!propagate_h_pair_nodes(
                edge,
                transposed,
                first,
                second,
                parameter,
                first_next,
                second_next)) {
            failure_edge = edge_index;
            failure_operation = static_cast<int>(operation);
            return false;
        }
        nodes[static_cast<std::size_t>(output1)] = first_next;
        nodes[static_cast<std::size_t>(
            plan.output2_nodes[operation])] = second_next;
    }
    return true;
}

bool evaluate_incremental_proposal(
    const RVineDensityPlan& plan,
    const std::vector<PreparedEdge>& edges,
    const ParameterPack& parameters,
    int variable,
    double proposal_value,
    std::int64_t row,
    const double* accepted_nodes,
    const double* accepted_contributions,
    double* proposal_nodes,
    double* proposal_contributions,
    int* proposal_node_generations,
    int proposal_generation,
    double& proposal_log_pdf) {
    const auto operation_count = plan.edge_indices.size();
    const auto input_node = static_cast<std::size_t>(plan.input_nodes[
        static_cast<std::size_t>(variable)]);
    proposal_nodes[input_node] = clip_open_unit(proposal_value);
    proposal_node_generations[input_node] = proposal_generation;

    const auto node_value = [
            accepted_nodes,
            proposal_nodes,
            proposal_node_generations,
            proposal_generation](int node) noexcept {
        const auto index = static_cast<std::size_t>(node);
        return proposal_node_generations[index] == proposal_generation
            ? proposal_nodes[index]
            : accepted_nodes[index];
    };

    const auto variable_index = static_cast<std::size_t>(variable);
    const int begin = plan.affected_operation_offsets[variable_index];
    const int end = plan.affected_operation_offsets[variable_index + 1U];
    for (int position = begin; position < end; ++position) {
        const auto operation = static_cast<std::size_t>(
            plan.affected_operations[static_cast<std::size_t>(position)]);
        const int edge_index = plan.edge_indices[operation];
        const PreparedEdge& edge = edges[static_cast<std::size_t>(edge_index)];
        const double first = node_value(plan.input1_nodes[operation]);
        const double second = node_value(plan.input2_nodes[operation]);
        const bool transposed = plan.transposed[operation] != 0;
        const double parameter = parameter_at(edge.edge, parameters, row);
        const double contribution = edge.edge.parameter_free
            ? 0.0
            : detail::edge_log_pdf(
                edge, transposed, first, second, parameter);
        proposal_contributions[operation] = contribution;
        if (!std::isfinite(contribution)) {
            return false;
        }

        const int output1 = plan.output1_nodes[operation];
        if (output1 == -1) {
            continue;
        }
        double first_next = 0.0;
        double second_next = 0.0;
        if (!propagate_h_pair_nodes(
                edge,
                transposed,
                first,
                second,
                parameter,
                first_next,
                second_next)) {
            return false;
        }
        proposal_nodes[static_cast<std::size_t>(output1)] = first_next;
        proposal_node_generations[static_cast<std::size_t>(output1)] =
            proposal_generation;
        const auto output2 = static_cast<std::size_t>(
            plan.output2_nodes[operation]);
        proposal_nodes[output2] = second_next;
        proposal_node_generations[output2] = proposal_generation;
    }

    // Preserve the full density plan's original summation order exactly.
    proposal_log_pdf = 0.0;
    int affected_position = begin;
    for (std::size_t operation = 0;
         operation < operation_count; ++operation) {
        const bool affected = affected_position < end
            && plan.affected_operations[static_cast<std::size_t>(
                affected_position)] == static_cast<int>(operation);
        const double contribution = affected
            ? proposal_contributions[operation]
            : accepted_contributions[operation];
        if (affected) {
            ++affected_position;
        }
        if (!std::isfinite(contribution)
            || !std::isfinite(proposal_log_pdf + contribution)) {
            return false;
        }
        proposal_log_pdf += contribution;
    }
    return true;
}

}  // namespace

MCMCResult mcmc_chunk(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    const std::vector<int>& given_indices,
    DoubleView given_values,
    const std::vector<int>& free_indices,
    DoubleView current_state,
    std::int64_t state_rows,
    std::int64_t state_columns,
    DoubleView current_log_pdf,
    std::int64_t global_step_offset,
    DoubleView proposal_uniforms,
    std::int64_t proposal_steps,
    std::int64_t proposal_rows,
    DoubleView acceptance_uniforms,
    std::int64_t acceptance_steps,
    std::int64_t acceptance_rows,
    int n_threads,
    MCMCDensityAlgorithm density_algorithm,
    std::uint64_t memory_budget_bytes) {
    MCMCResult out;
    out.n_rows = state_rows;
    out.dimension = plan.dimension;
    out.coordinate_steps = proposal_steps;
    out.n_threads_requested = n_threads;
    out.memory_budget_bytes = memory_budget_bytes;
    if (!scar_internal::valid_thread_count(n_threads) || state_rows < 0
        || state_columns != plan.dimension
        || global_step_offset < 0 || proposal_steps < 0
        || proposal_rows != state_rows
        || acceptance_steps != proposal_steps
        || acceptance_rows != state_rows
        || parameters.n_rows != state_rows
        || !validate_density_plan(plan, edges.size())
        || !validate_partition(
            plan.dimension,
            given_indices,
            given_values,
            free_indices)) {
        fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }

    const auto rows = static_cast<std::size_t>(state_rows);
    const auto dimension = static_cast<std::size_t>(plan.dimension);
    const auto steps = static_cast<std::size_t>(proposal_steps);
    std::size_t state_value_count = 0;
    std::size_t draw_value_count = 0;
    if (!scar_internal::checked_size_mul(
            rows, dimension, state_value_count)
        || !scar_internal::checked_size_mul(
            steps, rows, draw_value_count)
        || current_state.size() != state_value_count
        || current_log_pdf.size() != rows
        || proposal_uniforms.size() != draw_value_count
        || acceptance_uniforms.size() != draw_value_count
        || (state_value_count > 0 && current_state.data() == nullptr)
        || (rows > 0 && current_log_pdf.data() == nullptr)
        || (draw_value_count > 0
            && (proposal_uniforms.data() == nullptr
                || acceptance_uniforms.data() == nullptr))) {
        fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    const int parameter_status = validate_parameter_pack(edges, parameters);
    if (parameter_status != SCAR_OK) {
        fail_mcmc(out, parameter_status, -1, -1, -1);
        return out;
    }
    std::vector<PreparedEdge> prepared_edges;
    const int prepare_status = prepare_edges(edges, prepared_edges);
    if (prepare_status != SCAR_OK) {
        fail_mcmc(out, prepare_status, -1, -1, -1);
        return out;
    }
    for (std::size_t index = 0; index < given_values.size(); ++index) {
        if (!std::isfinite(given_values[index])
            || given_values[index] <= 0.0 || given_values[index] >= 1.0) {
            fail_mcmc(out, SCAR_INVALID_PARAMETER, -1, -1, -1);
            return out;
        }
    }
    for (std::size_t index = 0; index < state_value_count; ++index) {
        if (!std::isfinite(current_state[index])) {
            fail_mcmc(
                out,
                SCAR_INVALID_PARAMETER,
                static_cast<std::int64_t>(index / dimension),
                -1,
                -1);
            return out;
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        if (!std::isfinite(current_log_pdf[row])) {
            fail_mcmc(
                out,
                SCAR_INVALID_PARAMETER,
                static_cast<std::int64_t>(row),
                -1,
                -1);
            return out;
        }
        for (std::size_t index = 0; index < given_indices.size(); ++index) {
            const auto variable = static_cast<std::size_t>(
                given_indices[index]);
            if (current_state[row * dimension + variable]
                != given_values[index]) {
                fail_mcmc(
                    out,
                    SCAR_INVALID_PARAMETER,
                    static_cast<std::int64_t>(row),
                    -1,
                    -1);
                return out;
            }
        }
    }
    if (!valid_open_uniforms(proposal_uniforms)
        || !valid_open_uniforms(acceptance_uniforms)) {
        fail_mcmc(out, SCAR_INVALID_PARAMETER, -1, -1, -1);
        return out;
    }

    out.affected_operations.reserve(free_indices.size());
    for (int variable : free_indices) {
        out.affected_operations.push_back(
            affected_operation_count(plan, variable));
    }
    std::size_t incremental_chunk_rows = 0;
    std::uint64_t per_row_workspace_bytes = 0;
    std::uint64_t incremental_fixed_bytes = 0;
    std::uint64_t full_peak_bytes = 0;
    const bool has_proposals = steps > 0 && rows > 0;
    const bool incremental_fits = incremental_memory_layout(
        plan,
        rows,
        state_value_count,
        draw_value_count,
        has_proposals,
        memory_budget_bytes,
        incremental_chunk_rows,
        per_row_workspace_bytes,
        incremental_fixed_bytes);
    const bool full_fits = full_memory_layout(
        plan,
        rows,
        state_value_count,
        draw_value_count,
        has_proposals,
        memory_budget_bytes,
        full_peak_bytes);
    if (density_algorithm == MCMCDensityAlgorithm::Auto) {
        if (
            incremental_fits
            && rows != 1U
            && incremental_is_structurally_profitable(plan, free_indices)) {
            density_algorithm = MCMCDensityAlgorithm::Incremental;
        } else if (full_fits) {
            density_algorithm = MCMCDensityAlgorithm::FullRecompute;
        } else {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
    }
    if (density_algorithm == MCMCDensityAlgorithm::Incremental
        && !incremental_fits) {
        // Explicit incremental requests fail before result/workspace
        // allocation.  Auto requests select the existing full driver.
        fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    if (density_algorithm == MCMCDensityAlgorithm::FullRecompute
        && !full_fits) {
        fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    out.density_algorithm = density_algorithm;
    out.peak_workspace_bytes = density_algorithm
        == MCMCDensityAlgorithm::Incremental
        ? incremental_fixed_bytes
        : full_peak_bytes;

    out.state.assign(state_value_count, 0.0);
    if (state_value_count > 0) {
        std::copy_n(
            current_state.data(), state_value_count, out.state.data());
    }
    out.log_pdf.assign(rows, 0.0);
    if (rows > 0) {
        std::copy_n(current_log_pdf.data(), rows, out.log_pdf.data());
    }
    out.proposed.assign(free_indices.size(), 0);
    out.accepted.assign(free_indices.size(), 0);
    if (steps == 0 || rows == 0) {
        return out;
    }

    const auto free_count = static_cast<std::int64_t>(free_indices.size());
    const auto starting_coordinate = global_step_offset % free_count;
    if (density_algorithm == MCMCDensityAlgorithm::Incremental) {
        const auto node_count = static_cast<std::size_t>(plan.node_count);
        const auto operation_count = plan.edge_indices.size();
        std::size_t node_values = 0;
        std::size_t contribution_values = 0;
        if (!scar_internal::checked_size_mul(
                incremental_chunk_rows, node_count, node_values)
            || !scar_internal::checked_size_mul(
                incremental_chunk_rows,
                operation_count,
                contribution_values)) {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
        std::vector<double> accepted_nodes(node_values, 0.0);
        std::vector<double> proposal_nodes(node_values, 0.0);
        std::vector<double> accepted_contributions(
            contribution_values, 0.0);
        std::vector<double> proposal_contributions(
            contribution_values, 0.0);
        std::vector<int> proposal_node_generations(node_values, 0);

        std::size_t cache_values = 0;
        std::uint64_t cache_bytes = 0;
        if (!scar_internal::checked_size_add(
                node_values,
                contribution_values,
                cache_values)
            || !scar_internal::checked_byte_count<double>(
                cache_values, cache_bytes)) {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
        out.cache_bytes = cache_bytes;
        std::uint64_t total_cache_bytes = 0;
        std::uint64_t marker_bytes = 0;
        if (!scar_internal::checked_uint64_add(
                cache_bytes, cache_bytes, total_cache_bytes)) {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
        if (!scar_internal::checked_byte_count<int>(
                node_values, marker_bytes)) {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
        if (!scar_internal::checked_uint64_add(
                total_cache_bytes,
                marker_bytes,
                total_cache_bytes)) {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
        if (!scar_internal::checked_uint64_add(
                incremental_fixed_bytes,
                total_cache_bytes,
                out.peak_workspace_bytes)) {
            fail_mcmc(out, SCAR_INVALID_SIZE, -1, -1, -1);
            return out;
        }
        out.max_chunk_rows = static_cast<std::uint64_t>(
            incremental_chunk_rows);

        for (std::size_t step = 0; step < steps; ++step) {
            const auto local_coordinate = static_cast<std::int64_t>(
                step % free_indices.size());
            const auto coordinate_index = static_cast<std::size_t>(
                (starting_coordinate + local_coordinate) % free_count);
            out.proposed[coordinate_index] += static_cast<std::uint64_t>(rows);
        }

        for (std::size_t row_begin = 0;
             row_begin < rows; row_begin += incremental_chunk_rows) {
            const std::size_t chunk_row_count = std::min(
                incremental_chunk_rows, rows - row_begin);
            ++out.row_chunks;
            for (std::size_t local_row = 0;
                 local_row < chunk_row_count; ++local_row) {
                const std::size_t row = row_begin + local_row;
                double* row_accepted_nodes =
                    accepted_nodes.data() + local_row * node_count;
                double* row_proposal_nodes =
                    proposal_nodes.data() + local_row * node_count;
                double* row_accepted_contributions =
                    accepted_contributions.data()
                    + local_row * operation_count;
                double* row_proposal_contributions =
                    proposal_contributions.data()
                    + local_row * operation_count;
                int* row_proposal_node_generations =
                    proposal_node_generations.data()
                    + local_row * node_count;
                double initialized_log_pdf = 0.0;
                int failure_edge = -1;
                int failure_operation = -1;
                if (!initialize_density_cache(
                        plan,
                        prepared_edges,
                        parameters,
                        out.state.data() + row * dimension,
                        static_cast<std::int64_t>(row),
                        row_accepted_nodes,
                        row_accepted_contributions,
                        initialized_log_pdf,
                        failure_edge,
                        failure_operation)) {
                    fail_mcmc(
                        out,
                        SCAR_NUMERICAL_FAILURE,
                        static_cast<std::int64_t>(row),
                        failure_edge,
                        failure_operation);
                    return out;
                }

                for (std::size_t step = 0; step < steps; ++step) {
                    const auto local_coordinate = static_cast<std::int64_t>(
                        step % free_indices.size());
                    const auto coordinate_index = static_cast<std::size_t>(
                        (starting_coordinate + local_coordinate) % free_count);
                    const int variable = free_indices[coordinate_index];
                    const double proposal_value = proposal_uniforms[
                        step * rows + row];
                    const std::uint64_t closure_operations =
                        out.affected_operations[coordinate_index];
                    out.affected_operation_evaluations += closure_operations;
                    double proposal_log_pdf = 0.0;
                    const bool finite_proposal = evaluate_incremental_proposal(
                        plan,
                        prepared_edges,
                        parameters,
                        variable,
                        proposal_value,
                        static_cast<std::int64_t>(row),
                        row_accepted_nodes,
                        row_accepted_contributions,
                        row_proposal_nodes,
                        row_proposal_contributions,
                        row_proposal_node_generations,
                        static_cast<int>(step + 1U),
                        proposal_log_pdf);
                    if (!finite_proposal) {
                        ++out.non_finite_proposals;
                        continue;
                    }

                    const double log_alpha =
                        proposal_log_pdf - out.log_pdf[row];
                    const double acceptance = acceptance_uniforms[
                        step * rows + row];
                    if (std::log(acceptance) >= log_alpha) {
                        continue;
                    }
                    out.state[row * dimension
                        + static_cast<std::size_t>(variable)] = proposal_value;
                    out.log_pdf[row] = proposal_log_pdf;
                    const auto variable_index = static_cast<std::size_t>(
                        variable);
                    const int node_begin =
                        plan.affected_node_offsets[variable_index];
                    const int node_end =
                        plan.affected_node_offsets[variable_index + 1U];
                    for (int position = node_begin;
                         position < node_end; ++position) {
                        const auto node = static_cast<std::size_t>(
                            plan.affected_nodes[
                                static_cast<std::size_t>(position)]);
                        row_accepted_nodes[node] = row_proposal_nodes[node];
                    }
                    const int operation_begin =
                        plan.affected_operation_offsets[variable_index];
                    const int operation_end =
                        plan.affected_operation_offsets[variable_index + 1U];
                    for (int position = operation_begin;
                         position < operation_end; ++position) {
                        const auto operation = static_cast<std::size_t>(
                            plan.affected_operations[
                                static_cast<std::size_t>(position)]);
                        row_accepted_contributions[operation] =
                            row_proposal_contributions[operation];
                    }
                    ++out.accepted[coordinate_index];
                }
            }
        }
        return out;
    }

    std::vector<double> proposal(state_value_count, 0.0);
    std::vector<double> proposal_log_pdf(rows, 0.0);
    std::vector<double> node_workspace;
    for (std::size_t step = 0; step < steps; ++step) {
        std::copy(out.state.begin(), out.state.end(), proposal.begin());
        const auto local_coordinate = static_cast<std::int64_t>(
            step % free_indices.size());
        const auto coordinate_index = static_cast<std::size_t>(
            (starting_coordinate + local_coordinate) % free_count);
        const auto variable = static_cast<std::size_t>(
            free_indices[coordinate_index]);
        for (std::size_t row = 0; row < rows; ++row) {
            proposal[row * dimension + variable] = proposal_uniforms[
                step * rows + row];
        }

        std::int64_t failure_row = -1;
        int failure_edge = -1;
        int failure_operation = -1;
        std::uint64_t non_finite_rows = 0;
        DensityDiagnostics density_diagnostics;
        const DoubleView proposal_view = {
            proposal.data(), proposal.size()};
        const int density_status = detail::evaluate_density_plan_rows(
            plan,
            prepared_edges,
            parameters,
            proposal_view,
            state_rows,
            state_columns,
            proposal_log_pdf.data(),
            nullptr,
            true,
            node_workspace,
            failure_row,
            failure_edge,
            failure_operation,
            non_finite_rows,
            density_diagnostics);
        if (density_status != SCAR_OK) {
            fail_mcmc(
                out,
                density_status,
                failure_row,
                failure_edge,
                failure_operation);
            return out;
        }
        out.non_finite_proposals += non_finite_rows;
        out.affected_operation_evaluations +=
            static_cast<std::uint64_t>(rows)
            * static_cast<std::uint64_t>(plan.edge_indices.size());
        out.proposed[coordinate_index] += static_cast<std::uint64_t>(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            const double proposal_value = proposal_log_pdf[row];
            const double log_alpha = proposal_value - out.log_pdf[row];
            const double acceptance = acceptance_uniforms[
                step * rows + row];
            if (std::log(acceptance) < log_alpha) {
                out.state[row * dimension + variable] =
                    proposal[row * dimension + variable];
                out.log_pdf[row] = proposal_value;
                ++out.accepted[coordinate_index];
            }
        }
    }
    return out;
}

}  // namespace scar::rvine
