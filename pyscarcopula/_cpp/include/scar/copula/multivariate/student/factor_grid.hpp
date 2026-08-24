#pragma once

#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/core/result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

struct FactorStudentGridResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_ddf;
    std::size_t rows = 0;
    std::size_t grid_size = 0;
    std::size_t dimension_tiles = 0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_axis = 0;
    int parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;
    std::size_t partial_workspace_peak_bytes = 0;
    std::uint64_t ppf_exact_values = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

FactorStudentGridResult factor_student_log_pdf_and_dlog_ddf_grid(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    const double* df_grid,
    std::size_t grid_size,
    std::size_t dimension_tile,
    int n_threads = 1);

}  // namespace scar
