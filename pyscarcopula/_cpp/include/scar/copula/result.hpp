#pragma once

#include "scar/copula/grid_values.hpp"
#include "scar/core/result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

/// Per-row multivariate log densities and scalar-parameter derivatives.
struct MultivariateRowsResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_dr;
    Status status = Status::Ok;
    FailureContext failure{};
    std::uint64_t student_ppf_cache_values = 0;
    std::uint64_t student_ppf_exact_values = 0;
    std::uint64_t student_ppf_asymptotic_values = 0;
    std::uint64_t student_workspace_growth_events = 0;
    std::size_t student_workspace_peak_bytes = 0;
    int n_threads_requested = 1;
    int row_parallel_blocks = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct MultivariateGridResult {
    GridValues pdf;
    GridValues d_pdf_dx;
    Status status = Status::Ok;
    FailureContext failure{};
    std::uint64_t student_ppf_cache_values = 0;
    std::uint64_t student_ppf_exact_values = 0;
    std::uint64_t student_ppf_asymptotic_values = 0;
    std::uint64_t student_workspace_growth_events = 0;
    std::size_t student_workspace_peak_bytes = 0;
    int n_threads_requested = 1;
    int student_parallel_blocks = 0;
    int equicorr_parallel_blocks = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Per-row sufficient statistics for an equicorrelation Gaussian copula.
struct EquicorrPreparationResult {
    std::vector<double> sum_z;
    std::vector<double> sum_z2;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_blocks = 0;
    int parallel_axis = 0;  ///< 0 sequential, 1 rows, 2 dimension tiles.
    std::size_t dimension_tiles = 0;
    std::size_t temporary_values = 0;
    std::uint64_t clipping_events = 0;
    std::uint64_t nonfinite_values = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
