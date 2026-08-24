#pragma once

#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/core/span.hpp"
#include "scar/rvine_plan.hpp"
#include "scar/status.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
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
    PreparedPairKernel kernel;
    PreparedPairKernel transposed_kernel;

    PreparedEdge(EdgeSpec edge_spec, CopulaSpec transposed)
        : edge(std::move(edge_spec)),
          transposed_copula(std::move(transposed)),
          kernel(edge.copula),
          transposed_kernel(transposed_copula) {}
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

/// Diagnostics shared by standalone and MCMC density evaluation.
struct DensityDiagnostics {
    std::uint64_t density_operations = 0;
    std::uint64_t h_pair_operations = 0;
    std::uint64_t independence_fast_paths = 0;
};

/// Result of one fused row-wise R-vine density request.
struct DensityResult {
    std::vector<double> log_pdf;
    std::int64_t n_rows = 0;
    int dimension = 0;
    int status = SCAR_OK;
    std::int64_t failure_row = -1;
    int failure_edge = -1;
    int failure_operation = -1;
    int n_threads_requested = 1;
    int n_threads_used = 1;
    DensityDiagnostics diagnostics;
};

/// Result of one static R-vine Rosenblatt residual extraction request.
struct RosenblattResult {
    std::vector<double> residuals;
    std::int64_t n_rows = 0;
    int dimension = 0;
    int status = SCAR_OK;
    std::int64_t failure_row = -1;
    int failure_edge = -1;
    int failure_operation = -1;
    int n_threads_requested = 1;
    int n_threads_used = 1;
    std::uint64_t h_pair_operations = 0;
    std::uint64_t independence_fast_paths = 0;
};

enum class MCMCDensityAlgorithm : int {
    Auto = 0,
    FullRecompute = 1,
    Incremental = 2,
};

/// Result of one stateful coordinate-wise conditional-MCMC chunk.
struct MCMCResult {
    std::vector<double> state;
    std::vector<double> log_pdf;
    std::vector<std::uint64_t> proposed;
    std::vector<std::uint64_t> accepted;
    std::int64_t n_rows = 0;
    int dimension = 0;
    std::int64_t coordinate_steps = 0;
    int status = SCAR_OK;
    std::int64_t failure_row = -1;
    int failure_edge = -1;
    int failure_operation = -1;
    int n_threads_requested = 1;
    int n_threads_used = 1;
    std::uint64_t non_finite_proposals = 0;
    MCMCDensityAlgorithm density_algorithm =
        MCMCDensityAlgorithm::FullRecompute;
    std::vector<std::uint64_t> affected_operations;
    std::uint64_t affected_operation_evaluations = 0;
    std::uint64_t cache_bytes = 0;
    std::uint64_t peak_workspace_bytes = 0;
    std::uint64_t row_chunks = 0;
    std::uint64_t max_chunk_rows = 0;
    std::uint64_t memory_budget_bytes = 0;
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

DensityResult log_pdf_rows(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads = 1);

RosenblattResult rosenblatt_transform(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads = 1);

MCMCResult mcmc_chunk(
    const RVineDensityPlan& plan,
    const std::vector<EdgeSpec>& edges,
    const ParameterPack& parameters,
    const std::vector<int>& given_indices,
    DoubleView given_values,
    const std::vector<int>& free_indices,
    DoubleView current_state,
    std::int64_t state_rows,
    std::int64_t state_columns,
    DoubleView current_log_pdf,
    std::int64_t global_step_offset,
    DoubleView proposal_uniforms,
    std::int64_t proposal_steps,
    std::int64_t proposal_rows,
    DoubleView acceptance_uniforms,
    std::int64_t acceptance_steps,
    std::int64_t acceptance_rows,
    int n_threads = 1,
    MCMCDensityAlgorithm density_algorithm = MCMCDensityAlgorithm::Auto,
    std::uint64_t memory_budget_bytes = 64U * 1024U * 1024U);

namespace detail {

/// Evaluate one prepared edge density with its requested orientation.
double edge_log_pdf(
    const PreparedEdge& edge,
    bool transposed,
    double first,
    double second,
    double parameter);

/// Validate and prepare inputs shared by density and Rosenblatt requests.
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

/// Internal prepared traversal shared by density, Rosenblatt, and MCMC.
int evaluate_density_plan_rows(
    const RVineDensityPlan& plan,
    const std::vector<PreparedEdge>& edges,
    const ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    double* log_pdf,
    double* residuals,
    bool tolerate_non_finite,
    std::vector<double>& node_workspace,
    std::int64_t& failure_row,
    int& failure_edge,
    int& failure_operation,
    std::uint64_t& non_finite_rows,
    DensityDiagnostics& diagnostics);

}  // namespace detail

}  // namespace scar::rvine
