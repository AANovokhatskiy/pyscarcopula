#include "scar/copula.hpp"
#include "scar/copula/multivariate/student/density.hpp"

#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/multivariate/batch.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace scar_internal {

using scar::SCAR_INVALID_FAMILY;
using scar::SCAR_INVALID_PARAMETER;
using scar::SCAR_INVALID_SIZE;
using scar::SCAR_NUMERICAL_FAILURE;
using scar::SCAR_OK;

namespace {

constexpr std::int64_t kStudentGridMinRowsPerBlock = 8;
constexpr std::size_t kStudentRowsMinCells = 4096;

struct StudentBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
    StudentWorkspace::Diagnostics diagnostics;
};

bool valid_dense_correlation(const scar::CopulaSpec& spec) {
    std::size_t square = 0;
    if (spec.dim < 2
        || !valid_student_dimension(spec.dim, square)
        || spec.dense_inverse_cholesky().size() != square
        || !std::isfinite(spec.dense_log_determinant())) {
        return false;
    }
    for (int i = 0; i < spec.dim; ++i) {
        for (int j = 0; j < spec.dim; ++j) {
            const double value = spec.dense_inverse_cholesky()[
                static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(spec.dim)
                + static_cast<std::size_t>(j)];
            if (!std::isfinite(value)
                || (j > i && std::abs(value) > 1e-14)) {
                return false;
            }
        }
    }
    return true;
}

int validate_student_batch(
    const scar::CopulaSpec& spec,
    const scar::Observations& observations,
    std::int64_t row_offset) {

    if (observations.empty() || row_offset < 0) {
        return SCAR_INVALID_SIZE;
    }
    if (spec.family != scar::CopulaFamily::Student
        || spec.rotation != scar::Rotation::R0
        || spec.transform != scar::Transform::Softplus
        || !valid_dense_correlation(spec)) {
        return SCAR_INVALID_FAMILY;
    }
    if ((!spec.student_ppf_nodes().empty()
         || !spec.student_ppf_table().empty())
        && (spec.student_ppf_observation_count() <= 0
            || row_offset
                > spec.student_ppf_observation_count()
                    - static_cast<std::int64_t>(observations.size()))) {
        return SCAR_INVALID_SIZE;
    }
    return validate_multivariate_observations(spec, observations);
}

void accumulate_diagnostics(
    const StudentWorkspace::Diagnostics& diagnostics,
    scar::MultivariateRowsResult& out) {

    out.student_ppf_cache_values += diagnostics.ppf_cache_values;
    out.student_ppf_exact_values += diagnostics.ppf_exact_values;
    out.student_ppf_asymptotic_values += diagnostics.ppf_asymptotic_values;
    out.student_workspace_growth_events += diagnostics.growth_events;
    out.student_workspace_peak_bytes = std::max(
        out.student_workspace_peak_bytes,
        diagnostics.peak_capacity_bytes);
}

void accumulate_diagnostics(
    const StudentWorkspace::Diagnostics& diagnostics,
    scar::MultivariateGridResult& out) {

    out.student_ppf_cache_values += diagnostics.ppf_cache_values;
    out.student_ppf_exact_values += diagnostics.ppf_exact_values;
    out.student_ppf_asymptotic_values += diagnostics.ppf_asymptotic_values;
    out.student_workspace_growth_events += diagnostics.growth_events;
    out.student_workspace_peak_bytes = std::max(
        out.student_workspace_peak_bytes,
        diagnostics.peak_capacity_bytes);
}

}  // namespace

scar::MultivariateRowsResult student_log_pdf_and_grad_rows(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& degrees_of_freedom,
    std::int64_t row_offset,
    int n_threads) {

    scar::MultivariateRowsResult out;
    out.n_threads_requested = n_threads;
    out.status = validate_student_batch(spec, observations, row_offset);
    out.log_pdf.assign(
        observations.size(), -std::numeric_limits<double>::infinity());
    out.dlog_dr.assign(
        observations.size(), std::numeric_limits<double>::quiet_NaN());
    if (out.status != SCAR_OK) {
        return out;
    }
    if (!valid_thread_count(n_threads)) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }
    if (degrees_of_freedom.size() != 1
        && degrees_of_freedom.size() != observations.size()) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }

    const PreparedStudentDensity model = prepare_student_density(spec);
    const bool use_threads = n_threads > 1
        && grid_parallel_worthwhile(
            observations.size(),
            static_cast<std::size_t>(spec.dim),
            static_cast<std::size_t>(kStudentGridMinRowsPerBlock),
            kStudentRowsMinCells);
    std::vector<StudentBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    parallel_for_blocks(
        0,
        static_cast<std::int64_t>(observations.size()),
        kStudentGridMinRowsPerBlock,
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            StudentBlockResult& block_result = block_results[block];
            block_result.ran = true;
            StudentWorkspace workspace;
            workspace.reserve_x(static_cast<std::size_t>(spec.dim));
            workspace.reserve_dx_ddf(static_cast<std::size_t>(spec.dim));
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double df = parameter_at(degrees_of_freedom, row);
                double log_pdf = 0.0;
                double derivative = 0.0;
                const bool ok = std::isfinite(df)
                    && student_log_pdf_and_dlog_ddf(
                        model,
                        observations[row].data(),
                        df,
                        row_offset + row_index,
                        log_pdf,
                        derivative,
                        workspace);
                if (!ok) {
                    block_result.failure_index = row_index;
                    block_result.diagnostics = workspace.diagnostics;
                    return;
                }
                out.log_pdf[row] = log_pdf;
                out.dlog_dr[row] = derivative;
            }
            block_result.diagnostics = workspace.diagnostics;
        });

    for (const StudentBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.row_parallel_blocks;
        accumulate_diagnostics(block.diagnostics, out);
        if (block.failure_index >= 0
            && (out.failure_index < 0
                || block.failure_index < out.failure_index)) {
            out.failure_index = block.failure_index;
        }
    }
    if (out.failure_index >= 0) {
        out.status = SCAR_NUMERICAL_FAILURE;
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure_index + 1);
        std::fill(
            out.log_pdf.begin() + first_uncomputed,
            out.log_pdf.end(),
            -std::numeric_limits<double>::infinity());
        std::fill(
            out.dlog_dr.begin() + first_uncomputed,
            out.dlog_dr.end(),
            std::numeric_limits<double>::quiet_NaN());
    }
    return out;
}

scar::MultivariateGridResult student_pdf_and_grad_grid(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& state_grid,
    std::int64_t row_offset,
    int n_threads) {

    scar::MultivariateGridResult out;
    out.n_threads_requested = n_threads;
    initialize_multivariate_grid(
        out, observations.size(), state_grid.size());
    if (out.status != SCAR_OK) {
        return out;
    }
    out.status = validate_student_batch(spec, observations, row_offset);
    if (!valid_thread_count(n_threads)) {
        out.status = SCAR_INVALID_PARAMETER;
    }
    if (out.status != SCAR_OK || state_grid.empty()) {
        if (out.status == SCAR_OK) {
            out.status = SCAR_INVALID_SIZE;
        }
        return out;
    }
    if (!std::all_of(
            state_grid.begin(), state_grid.end(), [](double value) {
                return std::isfinite(value);
            })) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }

    std::vector<double> df_grid(state_grid.size(), 0.0);
    std::vector<double> dpsi_grid(state_grid.size(), 0.0);
    for (std::size_t column = 0; column < state_grid.size(); ++column) {
        df_grid[column] = copula_transform(spec, state_grid[column]);
        dpsi_grid[column] = copula_dtransform(spec, state_grid[column]);
    }

    const PreparedStudentDensity model = prepare_student_density(spec);
    std::vector<StudentBlockResult> block_results(
        static_cast<std::size_t>(n_threads));
    parallel_for_blocks(
        0,
        static_cast<std::int64_t>(observations.size()),
        kStudentGridMinRowsPerBlock,
        n_threads,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            StudentBlockResult& block_result = block_results[block];
            block_result.ran = true;
            StudentWorkspace workspace;
            workspace.reserve_x(static_cast<std::size_t>(spec.dim));
            workspace.reserve_dx_ddf(static_cast<std::size_t>(spec.dim));
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const std::size_t base = row * state_grid.size();
                student_fill_row_with_workspace(
                    model,
                    observations[row].data(),
                    row_offset + row_index,
                    df_grid,
                    dpsi_grid,
                    out.pdf.values.data() + base,
                    out.d_pdf_dx.values.data() + base,
                    workspace);
                for (std::size_t column = 0;
                     column < state_grid.size();
                     ++column) {
                    if (!std::isfinite(out.pdf.values[base + column])
                        || !std::isfinite(
                            out.d_pdf_dx.values[base + column])) {
                        block_result.failure_index = row_index;
                        block_result.diagnostics = workspace.diagnostics;
                        return;
                    }
                }
            }
            block_result.diagnostics = workspace.diagnostics;
        });

    for (const StudentBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.student_parallel_blocks;
        accumulate_diagnostics(block.diagnostics, out);
        if (block.failure_index >= 0
            && (out.failure_index < 0
                || block.failure_index < out.failure_index)) {
            out.failure_index = block.failure_index;
        }
    }
    if (out.failure_index >= 0) {
        out.status = SCAR_NUMERICAL_FAILURE;
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure_index + 1)
            * state_grid.size();
        std::fill(
            out.pdf.values.begin() + first_uncomputed,
            out.pdf.values.end(),
            0.0);
        std::fill(
            out.d_pdf_dx.values.begin() + first_uncomputed,
            out.d_pdf_dx.values.end(),
            0.0);
    }
    return out;
}

}  // namespace scar_internal
