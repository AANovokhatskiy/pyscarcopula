#pragma once

#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/core/span.hpp"
#include "scar/rvine_plan.hpp"
#include "scar/status.hpp"
#include "scar/vine/result.hpp"

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

/// Non-owning parameter buffers that must outlive the native call.
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

}  // namespace scar::rvine
