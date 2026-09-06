#pragma once

#include "scar/rvine.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar::rvine::detail {

double edge_log_pdf(
    const PreparedEdge& edge,
    bool transposed,
    double first,
    double second,
    double parameter);

int prepare_density_plan_request(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads,
    std::vector<PreparedEdge>& prepared_edges,
    std::size_t& value_count,
    std::int64_t& failure_row);

int evaluate_density_plan_rows(
    const RVineDensityPlan& plan,
    const std::vector<PreparedEdge>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    double* log_pdf,
    double* residuals,
    double* node_values,
    bool tolerate_non_finite,
    std::vector<double>& node_workspace,
    std::int64_t& failure_row,
    int& failure_edge,
    int& failure_operation,
    std::uint64_t& non_finite_rows,
    DensityDiagnostics& diagnostics);

}  // namespace scar::rvine::detail
