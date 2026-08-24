#pragma once

#include "scar/copula/spec.hpp"
#include "scar/gas.hpp"
#include "scar/gas_rvine/result.hpp"
#include "scar/rvine_plan.hpp"

#include <cstdint>
#include <vector>

namespace scar {

struct GasRvineEdge {
    CopulaSpec copula;
    GasParams gas_params;
    GasConfig gas_config;
    bool dynamic = false;
};

GasRvineSampleResult gas_rvine_sample(
    const std::vector<GasRvineEdge>& edges,
    const RVineTraversalPlan& plan,
    const double* uniforms,
    std::int64_t n_rows,
    std::int64_t uniform_columns,
    const double* parameter_paths,
    std::int64_t parameter_rows,
    std::int64_t parameter_edges);

}  // namespace scar
