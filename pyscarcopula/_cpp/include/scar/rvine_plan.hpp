#pragma once

#include <cstdint>
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

/// Opcodes shared by suffix and arbitrary-conditioning execution plans.
enum class RVineOpcode : int {
    H = 1,
    H_PAIR = 2,
    H_INV = 3,
    COPY = 4,
};

/// Origin of a node value before execution of the conditional program.
enum class RVineNodeSource : int {
    Computed = 0,
    Given = 1,
    Uniform = 2,
};

/// Flat conditional program compiled by an external topology layer.
struct RVineConditionalPlan {
    int dimension = 0;
    int node_count = 0;

    std::vector<int> given_variables;
    std::vector<int> given_nodes;
    std::vector<int> uniform_nodes;
    std::vector<int> node_sources;
    std::vector<int> node_source_indices;

    std::vector<int> opcodes;
    std::vector<int> edge_indices;
    std::vector<int> input1_nodes;
    std::vector<int> input2_nodes;
    std::vector<int> output1_nodes;
    std::vector<int> output2_nodes;
    std::vector<int> transposed;

    std::vector<int> output_nodes;
    std::vector<int> used_edges;
};

/// Flat density traversal compiled by an external R-vine structure layer.
struct RVineDensityPlan {
    int dimension = 0;
    int node_count = 0;

    std::vector<int> input_nodes;
    std::vector<int> edge_indices;
    std::vector<int> input1_nodes;
    std::vector<int> input2_nodes;
    std::vector<int> output1_nodes;
    std::vector<int> output2_nodes;
    std::vector<int> transposed;

    std::vector<int> residual_nodes;
    std::vector<int> used_edges;

    // Static dependency closures for single-coordinate density updates.
    // Each variable owns a topologically ordered slice in the flattened
    // operation/node arrays.  These arrays are compiled and validated once;
    // the MCMC hot path never walks the graph to discover dependencies.
    std::vector<int> affected_operation_offsets;
    std::vector<int> affected_operations;
    std::vector<int> affected_node_offsets;
    std::vector<int> affected_nodes;
};

}  // namespace scar
