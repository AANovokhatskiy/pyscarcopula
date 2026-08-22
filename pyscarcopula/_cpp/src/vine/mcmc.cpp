#include "scar/rvine.hpp"

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
    int n_threads) {
    MCMCResult out;
    out.n_rows = state_rows;
    out.dimension = plan.dimension;
    out.coordinate_steps = proposal_steps;
    out.n_threads_requested = n_threads;
    if (n_threads <= 0 || state_rows < 0
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

    std::vector<double> proposal(state_value_count, 0.0);
    std::vector<double> proposal_log_pdf(rows, 0.0);
    std::vector<double> node_workspace;
    const auto free_count = static_cast<std::int64_t>(free_indices.size());
    const auto starting_coordinate = global_step_offset % free_count;
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
        const int density_status = detail::evaluate_log_pdf_rows(
            plan,
            prepared_edges,
            parameters,
            proposal_view,
            state_rows,
            state_columns,
            proposal_log_pdf.data(),
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
