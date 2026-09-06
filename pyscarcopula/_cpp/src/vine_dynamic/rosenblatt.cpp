#include "scar/dynamic_rvine.hpp"

#include "scar/core/checked_arithmetic.hpp"
#include "scar/core/threading.hpp"
#include "scar/numerical_validation.hpp"
#include "scar/observation.hpp"

#include "../vine/density_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace scar {
namespace {

void fail_dynamic_rvine(
    rvine::RosenblattResult& out,
    Status status,
    std::int64_t row,
    int edge,
    int operation) {
    out.status = status;
    out.failure.row = row;
    out.failure.edge = edge;
    out.failure.operation = operation;
}

void fail_dynamic_rvine(
    rvine::RosenblattResult& out,
    int status,
    std::int64_t row,
    int edge,
    int operation) {
    fail_dynamic_rvine(
        out, status_from_int(status), row, edge, operation);
}

std::vector<rvine::EdgeSpec> validation_edges(
    const std::vector<DynamicRvineEdge>& edges) {
    std::vector<rvine::EdgeSpec> out;
    out.reserve(edges.size());
    for (const DynamicRvineEdge& dynamic_edge : edges) {
        rvine::EdgeSpec edge = dynamic_edge.edge;
        if (dynamic_edge.dynamics != DynamicRvineKind::Static) {
            // Dynamic evaluators own their parameter storage.  Marking the
            // copied validation descriptor parameter-free lets the common
            // pack validator ignore its deliberately empty source.
            edge.parameter_free = true;
            edge.parameter_source = rvine::ParameterSource::None;
            edge.parameter_index = -1;
        }
        out.push_back(std::move(edge));
    }
    return out;
}

void canonical_pairs(
    const RVineDensityPlan& plan,
    std::size_t operation,
    const std::vector<double>& nodes,
    std::size_t rows,
    std::size_t node_count,
    std::vector<double>& pairs) {
    pairs.resize(rows * 2U);
    const bool transposed = plan.transposed[operation] != 0;
    const std::size_t first_node = static_cast<std::size_t>(
        plan.input1_nodes[operation]);
    const std::size_t second_node = static_cast<std::size_t>(
        plan.input2_nodes[operation]);
    for (std::size_t row = 0; row < rows; ++row) {
        const double first = nodes[row * node_count + first_node];
        const double second = nodes[row * node_count + second_node];
        pairs[row * 2U] = transposed ? second : first;
        pairs[row * 2U + 1U] = transposed ? first : second;
    }
}

bool store_dynamic_pair(
    const RVineDensityPlan& plan,
    std::size_t operation,
    const std::vector<double>& canonical_second_given_first,
    const std::vector<double>& canonical_first_given_second,
    std::vector<double>& nodes,
    std::size_t rows,
    std::size_t node_count,
    std::int64_t& failure_row) {
    if (canonical_second_given_first.size() != rows
        || canonical_first_given_second.size() != rows) {
        return false;
    }
    const bool transposed = plan.transposed[operation] != 0;
    const std::size_t output1 = static_cast<std::size_t>(
        plan.output1_nodes[operation]);
    const std::size_t output2 = static_cast<std::size_t>(
        plan.output2_nodes[operation]);
    for (std::size_t row = 0; row < rows; ++row) {
        const double first_next = transposed
            ? canonical_second_given_first[row]
            : canonical_first_given_second[row];
        const double second_next = transposed
            ? canonical_first_given_second[row]
            : canonical_second_given_first[row];
        if (!std::isfinite(first_next) || !std::isfinite(second_next)) {
            failure_row = static_cast<std::int64_t>(row);
            return false;
        }
        nodes[row * node_count + output1] =
            rvine::clip_open_unit(first_next);
        nodes[row * node_count + output2] =
            rvine::clip_open_unit(second_next);
    }
    return true;
}

}  // namespace

rvine::RosenblattResult dynamic_rvine_rosenblatt_transform(
    const RVineDensityPlan& plan,
    const std::vector<DynamicRvineEdge>& edges,
    const rvine::ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads,
    bool capture_node_values) {
    rvine::RosenblattResult out;
    out.n_rows = observation_rows;
    out.dimension = plan.dimension;
    out.node_count = plan.node_count;
    out.n_threads_requested = n_threads;
    out.n_threads_used = 1;

    if (!scar_internal::valid_thread_count(n_threads)
        || observation_rows < 0
        || observation_columns != plan.dimension
        || parameters.n_rows != observation_rows
        || plan.residual_nodes.size()
            != static_cast<std::size_t>(plan.dimension)
        || !rvine::validate_density_plan(plan, edges.size())) {
        fail_dynamic_rvine(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    // Density-only plans may omit h outputs. This traversal always stores
    // both directions, so the more permissive density contract is not enough.
    if (std::any_of(plan.output1_nodes.begin(), plan.output1_nodes.end(),
            [](int node) { return node < 0; })
        || std::any_of(plan.output2_nodes.begin(), plan.output2_nodes.end(),
            [](int node) { return node < 0; })) {
        fail_dynamic_rvine(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }
    const std::size_t rows = static_cast<std::size_t>(observation_rows);
    const std::size_t dimension = static_cast<std::size_t>(plan.dimension);
    const std::size_t node_count = static_cast<std::size_t>(plan.node_count);
    std::size_t observation_count = 0;
    std::size_t node_value_count = 0;
    if (!scar_internal::checked_size_mul(
            rows, dimension, observation_count)
        || !scar_internal::checked_size_mul(
            rows, node_count, node_value_count)
        || observations.size() != observation_count
        || (observation_count > 0 && observations.data() == nullptr)) {
        fail_dynamic_rvine(out, SCAR_INVALID_SIZE, -1, -1, -1);
        return out;
    }

    const auto observation_validation = validate_pseudo_observations(observations);
    if (!observation_validation.is_ok()) {
        fail_dynamic_rvine(
            out, SCAR_INVALID_PARAMETER,
            observation_validation.failure.index / plan.dimension, -1, -1);
        return out;
    }

    const std::vector<rvine::EdgeSpec> parameter_validation =
        validation_edges(edges);
    const int parameter_status = rvine::validate_parameter_pack(
        parameter_validation, parameters);
    if (parameter_status != SCAR_OK) {
        fail_dynamic_rvine(out, parameter_status, -1, -1, -1);
        return out;
    }

    std::vector<rvine::EdgeSpec> edge_specs;
    edge_specs.reserve(edges.size());
    for (const DynamicRvineEdge& edge : edges) {
        edge_specs.push_back(edge.edge);
    }
    std::vector<rvine::PreparedEdge> prepared_edges;
    const int prepare_status = rvine::prepare_edges(
        edge_specs, prepared_edges);
    if (prepare_status != SCAR_OK) {
        fail_dynamic_rvine(out, prepare_status, -1, -1, -1);
        return out;
    }

    const double missing = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> nodes(node_value_count, missing);
    if (rows == 0) {
        return out;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t variable = 0; variable < dimension; ++variable) {
            const double value = observations[row * dimension + variable];
            nodes[
                row * node_count
                + static_cast<std::size_t>(plan.input_nodes[variable])] =
                    rvine::clip_open_unit(value);
        }
    }

    GasEvaluator gas_evaluator;
    std::vector<double> pairs;
    std::vector<double> first_dynamic;
    std::vector<double> second_dynamic;
    for (std::size_t operation = 0;
         operation < plan.edge_indices.size(); ++operation) {
        const int edge_index = plan.edge_indices[operation];
        const std::size_t edge_position = static_cast<std::size_t>(edge_index);
        const DynamicRvineEdge& dynamic_edge = edges[edge_position];
        const rvine::PreparedEdge& prepared_edge = prepared_edges[edge_position];
        const std::size_t first_node = static_cast<std::size_t>(
            plan.input1_nodes[operation]);
        const std::size_t second_node = static_cast<std::size_t>(
            plan.input2_nodes[operation]);
        const std::size_t output1 = static_cast<std::size_t>(
            plan.output1_nodes[operation]);
        const std::size_t output2 = static_cast<std::size_t>(
            plan.output2_nodes[operation]);

        if (dynamic_edge.dynamics == DynamicRvineKind::Static) {
            for (std::size_t row = 0; row < rows; ++row) {
                const double first = nodes[row * node_count + first_node];
                const double second = nodes[row * node_count + second_node];
                double first_next = first;
                double second_next = second;
                if (dynamic_edge.edge.parameter_free) {
                    ++out.independence_fast_paths;
                } else {
                    const double parameter = rvine::parameter_at(
                        dynamic_edge.edge,
                        parameters,
                        static_cast<std::int64_t>(row));
                    const double contribution = rvine::detail::edge_log_pdf(
                        prepared_edge,
                        plan.transposed[operation] != 0,
                        first,
                        second,
                        parameter);
                    if (!std::isfinite(contribution)
                        || !std::isfinite(out.log_likelihood + contribution)) {
                        fail_dynamic_rvine(
                            out,
                            SCAR_NUMERICAL_FAILURE,
                            static_cast<std::int64_t>(row),
                            edge_index,
                            static_cast<int>(operation));
                        return out;
                    }
                    out.log_likelihood += contribution;
                    rvine::h_pair(
                        prepared_edge,
                        plan.transposed[operation] != 0,
                        first,
                        second,
                        parameter,
                        first_next,
                        second_next);
                }
                if (!std::isfinite(first_next)
                    || !std::isfinite(second_next)) {
                    fail_dynamic_rvine(
                        out,
                        SCAR_NUMERICAL_FAILURE,
                        static_cast<std::int64_t>(row),
                        edge_index,
                        static_cast<int>(operation));
                    return out;
                }
                nodes[row * node_count + output1] =
                    rvine::clip_open_unit(first_next);
                nodes[row * node_count + output2] =
                    rvine::clip_open_unit(second_next);
            }
            out.h_pair_operations += rows;
            continue;
        }

        canonical_pairs(
            plan, operation, nodes, rows, node_count, pairs);
        const ObservationView pair_view = {
            pairs.empty() ? nullptr : pairs.data(), rows, 2};

        first_dynamic.clear();
        second_dynamic.clear();
        if (dynamic_edge.dynamics == DynamicRvineKind::Gas) {
            const GasFilterResult filtered = gas_evaluator.filter(
                dynamic_edge.gas_params,
                dynamic_edge.edge.copula,
                pair_view,
                dynamic_edge.gas_config);
            if (!filtered.is_ok()) {
                out.status = filtered.status;
                out.failure = filtered.failure;
                out.failure.edge = edge_index;
                out.failure.operation = static_cast<int>(operation);
                return out;
            }
            if (!std::isfinite(filtered.log_likelihood)
                || !std::isfinite(
                    out.log_likelihood + filtered.log_likelihood)) {
                fail_dynamic_rvine(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    -1,
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            out.log_likelihood += filtered.log_likelihood;
            if (filtered.r_path.size() != rows) {
                fail_dynamic_rvine(
                    out,
                    SCAR_INVALID_SIZE,
                    -1,
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            first_dynamic.resize(rows);
            second_dynamic.resize(rows);
            const bool transposed = plan.transposed[operation] != 0;
            for (std::size_t row = 0; row < rows; ++row) {
                double first_next = 0.0;
                double second_next = 0.0;
                rvine::h_pair(
                    prepared_edge,
                    transposed,
                    nodes[row * node_count + first_node],
                    nodes[row * node_count + second_node],
                    filtered.r_path[row],
                    first_next,
                    second_next);
                first_dynamic[row] = first_next;
                second_dynamic[row] = second_next;
            }
            // GAS values are already in plan input order, unlike the latent
            // evaluators' canonical directional pair below.
            for (std::size_t row = 0; row < rows; ++row) {
                if (!std::isfinite(first_dynamic[row])
                    || !std::isfinite(second_dynamic[row])) {
                    fail_dynamic_rvine(
                        out,
                        SCAR_NUMERICAL_FAILURE,
                        static_cast<std::int64_t>(row),
                        edge_index,
                        static_cast<int>(operation));
                    return out;
                }
                nodes[row * node_count + output1] =
                    rvine::clip_open_unit(first_dynamic[row]);
                nodes[row * node_count + output2] =
                    rvine::clip_open_unit(second_dynamic[row]);
            }
            out.h_pair_operations += rows;
            continue;
        }

        if (dynamic_edge.dynamics == DynamicRvineKind::ScarOu) {
            PreparedScarOuEvaluator evaluator(
                dynamic_edge.edge.copula,
                pairs,
                observation_rows,
                2,
                dynamic_edge.ou_config,
                dynamic_edge.ou_method);
            const LogLikResult likelihood = evaluator.loglik(
                dynamic_edge.ou_params);
            if (!likelihood.is_ok()) {
                out.status = likelihood.status;
                out.failure = likelihood.failure;
                out.failure.edge = edge_index;
                out.failure.operation = static_cast<int>(operation);
                return out;
            }
            if (!std::isfinite(likelihood.log_likelihood)
                || !std::isfinite(
                    out.log_likelihood + likelihood.log_likelihood)) {
                fail_dynamic_rvine(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    -1,
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            out.log_likelihood += likelihood.log_likelihood;
            const ScarOuVectorResult result = evaluator.mixture_h_pair(
                dynamic_edge.ou_params);
            if (!result.is_ok()) {
                out.status = result.status;
                out.failure = result.failure;
                out.failure.edge = edge_index;
                out.failure.operation = static_cast<int>(operation);
                return out;
            }
            if (result.values.size() != rows * 2U) {
                fail_dynamic_rvine(
                    out,
                    SCAR_INVALID_SIZE,
                    -1,
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            first_dynamic.assign(
                result.values.begin(), result.values.begin() + rows);
            second_dynamic.assign(
                result.values.begin() + rows, result.values.end());
        } else if (dynamic_edge.dynamics == DynamicRvineKind::ScarJacobi) {
            PreparedScarJacobiEvaluator evaluator(
                dynamic_edge.edge.copula,
                pairs,
                observation_rows,
                2,
                dynamic_edge.jacobi_config);
            const JacobiEvaluatorPairResult result = evaluator.mixture_h_pair(
                dynamic_edge.jacobi_params);
            if (!result.is_ok()) {
                out.status = result.status;
                out.failure = result.failure;
                out.failure.edge = edge_index;
                out.failure.operation = static_cast<int>(operation);
                return out;
            }
            const double log_likelihood =
                result.value.diagnostics.log_likelihood;
            if (!std::isfinite(log_likelihood)
                || !std::isfinite(out.log_likelihood + log_likelihood)) {
                fail_dynamic_rvine(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    -1,
                    edge_index,
                    static_cast<int>(operation));
                return out;
            }
            out.log_likelihood += log_likelihood;
            first_dynamic = result.value.first;
            second_dynamic = result.value.second;
        } else {
            fail_dynamic_rvine(
                out,
                SCAR_INVALID_FAMILY,
                -1,
                edge_index,
                static_cast<int>(operation));
            return out;
        }

        if (!store_dynamic_pair(
                plan,
                operation,
                first_dynamic,
                second_dynamic,
                nodes,
                rows,
                node_count,
                out.failure.row)) {
            fail_dynamic_rvine(
                out,
                SCAR_NUMERICAL_FAILURE,
                out.failure.row,
                edge_index,
                static_cast<int>(operation));
            return out;
        }
        out.h_pair_operations += rows;
    }

    out.residuals.resize(observation_count);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            const double value = nodes[
                row * node_count
                + static_cast<std::size_t>(plan.residual_nodes[column])];
            if (!std::isfinite(value)) {
                fail_dynamic_rvine(
                    out,
                    SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(row),
                    -1,
                    static_cast<int>(plan.edge_indices.size()));
                return out;
            }
            out.residuals[row * dimension + column] =
                rvine::clip_open_unit(value);
        }
    }
    if (capture_node_values) {
        out.node_values = std::move(nodes);
    }
    return out;
}

}  // namespace scar
