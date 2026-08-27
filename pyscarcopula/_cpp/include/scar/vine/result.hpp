#pragma once

#include "scar/core/result.hpp"

#include <cstdint>
#include <vector>

namespace scar::rvine {

/// Result of one generic unconditional R-vine traversal request.
struct SampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int n_threads_used = 1;
    std::uint64_t inverse_operations = 0;
    std::uint64_t forward_operations = 0;
    std::uint64_t independence_fast_paths = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Result of one suffix or DAG conditional-program request.
struct ConditionalSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};
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

    bool is_ok() const noexcept {
        return ok(status);
    }
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
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int n_threads_used = 1;
    DensityDiagnostics diagnostics;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Result of one static R-vine Rosenblatt residual extraction request.
struct RosenblattResult {
    std::vector<double> residuals;
    double log_likelihood = 0.0;
    std::int64_t n_rows = 0;
    int dimension = 0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int n_threads_used = 1;
    std::uint64_t h_pair_operations = 0;
    std::uint64_t independence_fast_paths = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
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
    std::uint64_t proposal_draws_used = 0;
    std::uint64_t acceptance_draws_used = 0;
    Status status = Status::Ok;
    FailureContext failure{};
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

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar::rvine
