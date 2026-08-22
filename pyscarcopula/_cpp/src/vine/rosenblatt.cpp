#include "scar/rvine.hpp"

#include <cstddef>

namespace scar::rvine {
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
    if (plan.residual_nodes.size()
            != static_cast<std::size_t>(plan.dimension)) {
        out.status = SCAR_INVALID_SIZE;
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
        out.failure_row);
    if (request_status != SCAR_OK) {
        out.status = request_status;
        return out;
    }

    out.residuals.assign(value_count, 0.0);
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
        false,
        node_workspace,
        out.failure_row,
        out.failure_edge,
        out.failure_operation,
        non_finite_rows,
        diagnostics);
    out.h_pair_operations = diagnostics.h_pair_operations;
    out.independence_fast_paths = diagnostics.independence_fast_paths;
    if (status != SCAR_OK) {
        out.status = status;
    }
    return out;
}

}  // namespace scar::rvine
