#pragma once

#include <vector>

namespace scar {

/// Model-independent execution plan for unconditional R-vine traversal.
struct RVineTraversalPlan {
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
    std::vector<int> inverse_transposed;

    std::vector<int> forward_offsets;
    std::vector<int> forward_edges;
    std::vector<int> forward_leaf_nodes;
    std::vector<int> forward_partner_nodes;
    std::vector<int> forward_leaf_output_nodes;
    std::vector<int> forward_partner_output_nodes;
    std::vector<int> forward_transposed;

    std::vector<int> update_u1_nodes;
    std::vector<int> update_u2_nodes;
};

}  // namespace scar
