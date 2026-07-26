#pragma once

#include "scar/copula.hpp"
#include "scar/gas.hpp"
#include "scar/rvine_plan.hpp"
#include "scar/status.hpp"

#include <cstdint>
#include <vector>

namespace scar {

struct GasRvineEdge {
    CopulaSpec copula;
    GasParams gas_params;
    GasConfig gas_config;
    bool dynamic = false;
};

struct GasRvineSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    int status = SCAR_OK;
    std::int64_t failure_row = -1;
    int failure_edge = -1;
};

GasRvineSampleResult gas_rvine_sample(
    const std::vector<GasRvineEdge>& edges,
    const RvineTraversalPlan& plan,
    const double* uniforms,
    std::int64_t n_rows,
    std::int64_t uniform_columns,
    const double* parameter_paths,
    std::int64_t parameter_rows,
    std::int64_t parameter_edges);

}  // namespace scar
