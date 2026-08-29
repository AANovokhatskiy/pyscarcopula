#include "scar/rvine.hpp"

#include "density_internal.hpp"

#include "scar/core/checked_arithmetic.hpp"

#include <cstddef>

namespace scar::rvine {
RosenblattResult rosenblatt_transform(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads,
    bool capture_node_values) {
    RosenblattResult out;
    out.n_rows = observation_rows;
    out.dimension = plan.dimension;
    out.node_count = plan.node_count;
    out.n_threads_requested = n_threads;
    if (plan.residual_nodes.size()
            != static_cast<std::size_t>(plan.dimension)) {
        out.status = Status::InvalidSize;
        return out;
    }

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
        out.failure.row);
    if (request_status != SCAR_OK) {
        out.status = status_from_int(request_status);
        return out;
    }

    out.residuals.assign(value_count, 0.0);
    if (capture_node_values) {
        std::size_t node_value_count = 0;
        if (!scar_internal::checked_size_mul(
                static_cast<std::size_t>(observation_rows),
                static_cast<std::size_t>(plan.node_count),
                node_value_count)) {
            out.status = Status::InvalidSize;
            return out;
        }
        out.node_values.assign(node_value_count, 0.0);
    }
    std::vector<double> node_workspace;
    std::uint64_t non_finite_rows = 0;
    DensityDiagnostics diagnostics;
    const int status = detail::evaluate_density_plan_rows(
        plan,
        prepared_edges,
        parameters,
        observations,
        observation_rows,
        observation_columns,
        nullptr,
        out.residuals.data(),
        capture_node_values ? out.node_values.data() : nullptr,
        false,
        node_workspace,
        out.failure.row,
        out.failure.edge,
        out.failure.operation,
        non_finite_rows,
        diagnostics);
    out.h_pair_operations = diagnostics.h_pair_operations;
    out.independence_fast_paths = diagnostics.independence_fast_paths;
    if (status != SCAR_OK) {
        out.status = status_from_int(status);
    }
    return out;
}

}  // namespace scar::rvine
