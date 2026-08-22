#pragma once

#include "scar/copula.hpp"
#include "scar/rvine_plan.hpp"
#include "scar/status.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar::rvine {

/// Parameter storage used by one edge in a packed R-vine request.
enum class ParameterSource : int {
    None = 0,
    Scalar = 1,
    RowPath = 2,
};

/// Model-independent edge metadata owned by a native execution request.
struct EdgeSpec {
    CopulaSpec copula;
    ParameterSource parameter_source = ParameterSource::None;
    int parameter_index = -1;
    bool parameter_free = false;
};

/// Non-owning parameter buffers kept alive by the pybind11 call frame.
struct ParameterPack {
    DoubleView scalar_parameters;
    DoubleView row_parameters;
    std::int64_t n_rows = 0;
    std::int64_t row_parameter_columns = 0;
};

/// Edge metadata prepared once before entering a row loop.
struct PreparedEdge {
    EdgeSpec edge;
    CopulaSpec transposed_copula;
};

/// Result of one generic unconditional R-vine traversal request.
struct SampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    int status = SCAR_OK;
    std::int64_t failure_row = -1;
    int failure_edge = -1;
    int failure_operation = -1;
    int n_threads_requested = 1;
    int n_threads_used = 1;
    std::uint64_t inverse_operations = 0;
    std::uint64_t forward_operations = 0;
    std::uint64_t independence_fast_paths = 0;
};

/// Result of one suffix or DAG conditional-program request.
struct ConditionalSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    int status = SCAR_OK;
    std::int64_t failure_row = -1;
    int failure_edge = -1;
    int failure_operation = -1;
    int n_threads_requested = 1;
    int n_threads_used = 1;
    std::uint64_t h_operations = 0;
    std::uint64_t h_pair_operations = 0;
    std::uint64_t inverse_operations = 0;
    std::uint64_t copy_operations = 0;
    std::uint64_t independence_fast_paths = 0;
    std::uint64_t row_blocks = 0;
    std::uint64_t max_block_rows = 0;
    std::uint64_t peak_workspace_bytes = 0;
};

bool valid_index(int value, int limit) noexcept;
bool validate_traversal_plan(
    const RVineTraversalPlan& plan,
    std::size_t edge_count);
bool validate_conditional_plan(
    const RVineConditionalPlan& plan,
    std::size_t edge_count);
bool validate_density_plan(
    const RVineDensityPlan& plan,
    std::size_t edge_count);
int prepare_edges(
    const std::vector<EdgeSpec>& edges,
    std::vector<PreparedEdge>& prepared);
int validate_parameter_pack(
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters) noexcept;
double parameter_at(
    const EdgeSpec& edge,
    const ParameterPack& parameters,
    std::int64_t row) noexcept;

double clip_open_unit(double value) noexcept;
double h(
    const PreparedEdge& edge,
    bool transposed,
    double value,
    double partner,
    double parameter);
void h_pair(
    const PreparedEdge& edge,
    bool first_transposed,
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next);
double h_inverse(
    const PreparedEdge& edge,
    bool transposed,
    double quantile,
    double given,
    double parameter);

SampleResult sample(
    const RVineTraversalPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView uniforms,
    std::int64_t uniform_rows,
    std::int64_t uniform_columns,
    int n_threads = 1);

ConditionalSampleResult conditional_sample(
    const RVineConditionalPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView given_values,
    DoubleView uniforms,
    std::int64_t uniform_rows,
    std::int64_t uniform_columns,
    int n_threads = 1);

}  // namespace scar::rvine
