#pragma once

#include "scar/copula.hpp"
#include "scar/gas.hpp"
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

struct GasRvinePlan {
    int dimension = 0;
    int node_count = 0;
    int last_uniform_column = -1;
    int last_output_node = -1;

    std::vector<int> output_nodes;
    std::vector<int> column_uniforms;

    std::vector<int> inverse_offsets;
    std::vector<int> inverse_edges;
    std::vector<int> inverse_partner_nodes;
    std::vector<int> inverse_output_nodes;

    std::vector<int> forward_offsets;
    std::vector<int> forward_edges;
    std::vector<int> forward_leaf_nodes;
    std::vector<int> forward_partner_nodes;
    std::vector<int> forward_leaf_output_nodes;
    std::vector<int> forward_partner_output_nodes;

    std::vector<int> update_u1_nodes;
    std::vector<int> update_u2_nodes;
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
    const GasRvinePlan& plan,
    const double* uniforms,
    std::int64_t n_rows,
    std::int64_t uniform_columns,
    const double* parameter_paths,
    std::int64_t parameter_rows,
    std::int64_t parameter_edges);

}  // namespace scar
