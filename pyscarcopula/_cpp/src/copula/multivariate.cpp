#include "scar/copula.hpp"

#include "scar/detail/copula.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

constexpr std::int64_t kStudentGridMinRowsPerBlock = 8;
constexpr std::int64_t kEquicorrGridMinRowsPerBlock = 64;
constexpr std::size_t kEquicorrGridMinCells = 262144;
constexpr std::size_t kStudentRowsMinCells = 4096;

struct StudentGridBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
    scar_internal::StudentWorkspace::Diagnostics diagnostics;
};

struct EquicorrGridBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
};

struct MultivariateRowsBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
    scar_internal::StudentWorkspace::Diagnostics diagnostics;
};

bool valid_factor(const CopulaSpec& spec) {
    std::size_t square = 0;
    if (spec.dim < 2
        || !scar_internal::valid_student_dimension(spec.dim, square)
        || spec.l_inv.size() != square
        || !std::isfinite(spec.log_det)) {
        return false;
    }
    for (int i = 0; i < spec.dim; ++i) {
        for (int j = 0; j < spec.dim; ++j) {
            const double value = spec.l_inv[
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

int validate(
    const CopulaSpec& spec,
    const Observations& u,
    std::int64_t row_offset) {

    if (u.empty() || row_offset < 0) {
        return SCAR_INVALID_SIZE;
    }
    if (spec.family == CopulaFamily::Student) {
        if (spec.rotation != Rotation::R0
            || spec.transform != Transform::Softplus
            || !valid_factor(spec)) {
            return SCAR_INVALID_FAMILY;
        }
        if ((!spec.ppf_nodes.empty() || !spec.ppf_table.empty())
            && (spec.ppf_n_obs <= 0
                || row_offset
                    > spec.ppf_n_obs
                        - static_cast<std::int64_t>(u.size()))) {
            return SCAR_INVALID_SIZE;
        }
    } else if (spec.family == CopulaFamily::EquicorrGaussian) {
        if (spec.rotation != Rotation::R0
            || spec.transform != Transform::GaussianTanh
            || spec.dim < 2) {
            return SCAR_INVALID_FAMILY;
        }
    } else {
        return SCAR_INVALID_FAMILY;
    }

    for (const auto& row : u) {
        if (row.size() != static_cast<std::size_t>(spec.dim)) {
            return SCAR_INVALID_SIZE;
        }
        if (!std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            return SCAR_INVALID_PARAMETER;
        }
    }
    return SCAR_OK;
}

double parameter_at(const std::vector<double>& r, std::size_t row) {
    return r.size() == 1 ? r[0] : r[row];
}

void initialize_grid(
    MultivariateGridResult& out,
    std::size_t n_obs,
    std::size_t n_grid) {

    out.pdf.n_obs = static_cast<std::int64_t>(n_obs);
    out.pdf.n_grid = static_cast<std::int64_t>(n_grid);
    out.d_pdf_dx.n_obs = out.pdf.n_obs;
    out.d_pdf_dx.n_grid = out.pdf.n_grid;
    std::size_t elements = 0;
    if (!scar_internal::checked_size_mul(n_obs, n_grid, elements)) {
        out.status = SCAR_INVALID_SIZE;
        return;
    }
    out.pdf.values.assign(elements, 0.0);
    out.d_pdf_dx.values.assign(elements, 0.0);
}

bool cholesky_with_jitter(
    const std::vector<double>& matrix,
    std::size_t dimension,
    std::vector<double>& lower,
    double* applied_jitter = nullptr) {

    return scar_internal::linalg::cholesky_symmetric_with_jitter(
        matrix.data(), dimension, lower, applied_jitter);
}

bool solve_spd(
    const std::vector<double>& lower,
    std::size_t dimension,
    const std::vector<double>& rhs,
    std::size_t columns,
    std::vector<double>& solution) {

    return scar_internal::linalg::solve_spd(
        lower.data(), dimension, rhs.data(), columns, solution);
}

struct ConditionalFactors {
    std::vector<double> r_gg;
    std::vector<double> r_gf;
    std::vector<double> r_fg;
    std::vector<double> r_ff;
    std::vector<double> lower_gg;
    std::vector<double> solved_cross;
    std::vector<double> schur_base;
    std::vector<double> prepared_lower_cov;
    bool use_prepared_lower_cov = false;
};

struct ConditionalWorkerWorkspace {
    ConditionalFactors row_factors;
    std::vector<double> solved_given;
    std::vector<double> covariance;
    std::vector<double> lower_cov;
    std::vector<double> given_vector;
    std::vector<double> conditional_mean;
    std::vector<double> innovation;
};

struct ConditionalBlockResult {
    bool ran = false;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
    std::uint64_t correlation_factorizations = 0;
};

bool prepare_conditional_factors(
    const double* correlation,
    std::size_t d,
    const std::vector<int>& given_indices,
    const std::vector<int>& free_indices,
    bool student,
    bool allow_prepared_lower_cov,
    ConditionalFactors& factors) {

    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = free_indices.size();
    factors.r_gg.assign(n_given * n_given, 0.0);
    factors.r_gf.assign(n_given * n_free, 0.0);
    factors.r_fg.assign(n_free * n_given, 0.0);
    factors.r_ff.assign(n_free * n_free, 0.0);
    factors.schur_base.assign(n_free * n_free, 0.0);
    for (std::size_t i = 0; i < n_given; ++i) {
        const std::size_t gi =
            static_cast<std::size_t>(given_indices[i]);
        for (std::size_t j = 0; j < n_given; ++j) {
            const std::size_t gj =
                static_cast<std::size_t>(given_indices[j]);
            factors.r_gg[i * n_given + j] = correlation[gi * d + gj];
        }
        for (std::size_t j = 0; j < n_free; ++j) {
            const std::size_t fj =
                static_cast<std::size_t>(free_indices[j]);
            factors.r_gf[i * n_free + j] = correlation[gi * d + fj];
        }
    }
    for (std::size_t i = 0; i < n_free; ++i) {
        const std::size_t fi =
            static_cast<std::size_t>(free_indices[i]);
        for (std::size_t j = 0; j < n_given; ++j) {
            const std::size_t gj =
                static_cast<std::size_t>(given_indices[j]);
            factors.r_fg[i * n_given + j] = correlation[fi * d + gj];
        }
        for (std::size_t j = 0; j < n_free; ++j) {
            const std::size_t fj =
                static_cast<std::size_t>(free_indices[j]);
            factors.r_ff[i * n_free + j] = correlation[fi * d + fj];
        }
    }
    if (!cholesky_with_jitter(factors.r_gg, n_given, factors.lower_gg)
        || !solve_spd(
            factors.lower_gg,
            n_given,
            factors.r_gf,
            n_free,
            factors.solved_cross)) {
        return false;
    }
    for (std::size_t i = 0; i < n_free; ++i) {
        for (std::size_t j = 0; j < n_free; ++j) {
            double schur = factors.r_ff[i * n_free + j];
            for (std::size_t k = 0; k < n_given; ++k) {
                schur -= factors.r_fg[i * n_given + k]
                    * factors.solved_cross[k * n_free + j];
            }
            factors.schur_base[i * n_free + j] = schur;
        }
    }
    factors.use_prepared_lower_cov = false;
    if (!allow_prepared_lower_cov) {
        return true;
    }
    double applied_jitter = 0.0;
    const bool factorized = cholesky_with_jitter(
        factors.schur_base,
        n_free,
        factors.prepared_lower_cov,
        &applied_jitter);
    if (!student && !factorized) {
        return false;
    }
    factors.use_prepared_lower_cov = student
        ? factorized && applied_jitter == 0.0
        : true;
    return true;
}

ConditionalSampleResult conditional_latent(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    const DoubleView* df,
    DoubleView normal_draws,
    const DoubleView* chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_threads_requested = n_threads;
    if (dimension < 2 || n_rows <= 0 || given_indices.empty()
        || given_indices.size() >= static_cast<std::size_t>(dimension)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    if (n_threads < 1 || n_threads > 256) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }
    const std::size_t d = static_cast<std::size_t>(dimension);
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = d - n_given;
    out.n_free = static_cast<std::int64_t>(n_free);
    if (correlation_rows != 1 && correlation_rows != n_rows) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    const std::size_t corr_rows =
        static_cast<std::size_t>(correlation_rows);
    if (correlations.size() != corr_rows * d * d
        || normal_draws.size() != rows * n_free
        || (given_latent.size() != n_given
            && given_latent.size() != rows * n_given)
        || (df != nullptr && df->size() != rows)
        || (chi_square_draws != nullptr
            && chi_square_draws->size() != rows)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }

    std::vector<bool> is_given(d, false);
    for (int index : given_indices) {
        if (index < 0 || index >= dimension
            || is_given[static_cast<std::size_t>(index)]) {
            out.status = SCAR_INVALID_PARAMETER;
            return out;
        }
        is_given[static_cast<std::size_t>(index)] = true;
    }
    std::vector<int> free_indices;
    free_indices.reserve(n_free);
    for (int index = 0; index < dimension; ++index) {
        if (!is_given[static_cast<std::size_t>(index)]) {
            free_indices.push_back(index);
        }
    }

    out.values.assign(rows * n_free, 0.0);
    ConditionalFactors common_factors;
    if (correlation_rows == 1) {
        if (!prepare_conditional_factors(
                correlations.data(),
                d,
                given_indices,
                free_indices,
                df != nullptr,
                true,
                common_factors)) {
            out.status = SCAR_NUMERICAL_FAILURE;
            out.failure_index = 0;
            return out;
        }
        out.correlation_factorizations = 1;
    }

    constexpr std::size_t min_rows = 32;
    constexpr std::size_t min_work = 65536;
    const bool use_threads = n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            rows, d * d, min_rows, min_work);
    std::vector<ConditionalBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        n_rows,
        static_cast<std::int64_t>(min_rows),
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            ConditionalBlockResult& block_result = block_results[block];
            block_result.ran = true;
            ConditionalWorkerWorkspace workspace;
            workspace.given_vector.assign(n_given, 0.0);
            workspace.covariance.assign(n_free * n_free, 0.0);
            workspace.conditional_mean.assign(n_free, 0.0);
            workspace.innovation.assign(n_free, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const ConditionalFactors* factors = &common_factors;
                if (correlation_rows != 1) {
                    if (!prepare_conditional_factors(
                            correlations.data() + row * d * d,
                            d,
                            given_indices,
                            free_indices,
                            df != nullptr,
                            false,
                            workspace.row_factors)) {
                        block_result.status = SCAR_NUMERICAL_FAILURE;
                        block_result.failure_index = row_index;
                        return;
                    }
                    ++block_result.correlation_factorizations;
                    factors = &workspace.row_factors;
                }
                for (std::size_t i = 0; i < n_given; ++i) {
                    workspace.given_vector[i] = given_latent[
                        given_latent.size() == n_given
                            ? i : row * n_given + i];
                }
                if (!solve_spd(
                        factors->lower_gg,
                        n_given,
                        workspace.given_vector,
                        1,
                        workspace.solved_given)) {
                    block_result.status = SCAR_NUMERICAL_FAILURE;
                    block_result.failure_index = row_index;
                    return;
                }

                double covariance_scale = 1.0;
                double radial_scale = 1.0;
                if (df != nullptr) {
                    const double degrees = (*df)[row];
                    const double chi_square = (*chi_square_draws)[row];
                    double delta = 0.0;
                    for (std::size_t i = 0; i < n_given; ++i) {
                        delta += workspace.given_vector[i]
                            * workspace.solved_given[i];
                    }
                    const double conditional_df =
                        degrees + static_cast<double>(n_given);
                    if (!(degrees > 2.0) || !(conditional_df > 0.0)
                        || !(chi_square > 0.0)
                        || !std::isfinite(delta)) {
                        block_result.status = SCAR_INVALID_PARAMETER;
                        block_result.failure_index = row_index;
                        return;
                    }
                    covariance_scale =
                        (degrees + delta) / conditional_df;
                    radial_scale = std::sqrt(
                        conditional_df / chi_square);
                }

                if (!factors->use_prepared_lower_cov) {
                    for (std::size_t i = 0; i < n_free; ++i) {
                        for (std::size_t j = 0; j < n_free; ++j) {
                            workspace.covariance[i * n_free + j] =
                                covariance_scale
                                * factors->schur_base[i * n_free + j];
                        }
                    }
                    if (!cholesky_with_jitter(
                            workspace.covariance,
                            n_free,
                            workspace.lower_cov)) {
                        block_result.status = SCAR_NUMERICAL_FAILURE;
                        block_result.failure_index = row_index;
                        return;
                    }
                }

                const std::vector<double>& innovation_factor =
                    factors->use_prepared_lower_cov
                    ? factors->prepared_lower_cov
                    : workspace.lower_cov;
                const double prepared_factor_scale =
                    factors->use_prepared_lower_cov && df != nullptr
                    ? std::sqrt(covariance_scale)
                    : 1.0;
                scar_internal::linalg::row_major_matvec(
                    factors->r_fg.data(),
                    n_free,
                    n_given,
                    workspace.solved_given.data(),
                    workspace.conditional_mean.data());
                scar_internal::linalg::lower_triangular_matvec(
                    innovation_factor.data(),
                    n_free,
                    normal_draws.data() + row * n_free,
                    workspace.innovation.data());
                for (std::size_t i = 0; i < n_free; ++i) {
                    const double value = workspace.conditional_mean[i]
                        + radial_scale * prepared_factor_scale
                            * workspace.innovation[i];
                    if (!std::isfinite(value)) {
                        block_result.status = SCAR_NUMERICAL_FAILURE;
                        block_result.failure_index = row_index;
                        return;
                    }
                    out.values[row * n_free + i] = value;
                }
            }
        });

    for (const ConditionalBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        out.correlation_factorizations += block.correlation_factorizations;
        if (block.failure_index >= 0
            && (out.failure_index < 0
                || block.failure_index < out.failure_index)) {
            out.failure_index = block.failure_index;
            out.status = block.status;
        }
    }
    if (out.failure_index >= 0) {
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure_index + 1) * n_free;
        std::fill(
            out.values.begin() + first_uncomputed,
            out.values.end(),
            0.0);
    }
    return out;
}

}  // namespace

MultivariateRowsResult multivariate_log_pdf_and_grad(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r,
    std::int64_t row_offset,
    int n_threads) {

    MultivariateRowsResult out;
    out.n_threads_requested = n_threads;
    out.status = validate(spec, u, row_offset);
    out.log_pdf.assign(
        u.size(), -std::numeric_limits<double>::infinity());
    out.dlog_dr.assign(
        u.size(), std::numeric_limits<double>::quiet_NaN());
    if (out.status != SCAR_OK) {
        return out;
    }
    if (n_threads < 1 || n_threads > 256) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }
    if (r.size() != 1 && r.size() != u.size()) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    const std::size_t min_rows = spec.family == CopulaFamily::Student
        ? static_cast<std::size_t>(kStudentGridMinRowsPerBlock)
        : static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock);
    const std::size_t min_cells = spec.family == CopulaFamily::Student
        ? kStudentRowsMinCells
        : kEquicorrGridMinCells;
    const bool use_threads = n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            u.size(), static_cast<std::size_t>(spec.dim),
            min_rows, min_cells);
    std::vector<MultivariateRowsBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(u.size()),
        static_cast<std::int64_t>(min_rows),
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            MultivariateRowsBlockResult& block_result = block_results[block];
            block_result.ran = true;
            scar_internal::StudentWorkspace student_workspace;
            if (spec.family == CopulaFamily::Student) {
                student_workspace.reserve_x(
                    static_cast<std::size_t>(spec.dim));
                student_workspace.reserve_dx_ddf(
                    static_cast<std::size_t>(spec.dim));
            }
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t i =
                    static_cast<std::size_t>(row_index);
                const double parameter = parameter_at(r, i);
                double log_pdf = 0.0;
                double dlog = 0.0;
                bool ok = std::isfinite(parameter);
                if (ok && spec.family == CopulaFamily::Student) {
                    ok = scar_internal::student_log_pdf_and_dlog_ddf(
                        spec,
                        u[i].data(),
                        parameter,
                        row_offset + row_index,
                        log_pdf,
                        dlog,
                        student_workspace);
                } else if (ok) {
                    log_pdf = scar_internal::equicorr_log_pdf(
                        spec, u[i].data(), parameter, &dlog);
                    ok = std::isfinite(log_pdf) && std::isfinite(dlog);
                }
                if (!ok) {
                    block_result.failure_index = row_index;
                    block_result.diagnostics =
                        student_workspace.diagnostics;
                    return;
                }
                out.log_pdf[i] = log_pdf;
                out.dlog_dr[i] = dlog;
            }
            block_result.diagnostics = student_workspace.diagnostics;
        });

    for (const MultivariateRowsBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.row_parallel_blocks;
        out.student_ppf_cache_values +=
            block.diagnostics.ppf_cache_values;
        out.student_ppf_exact_values +=
            block.diagnostics.ppf_exact_values;
        out.student_ppf_asymptotic_values +=
            block.diagnostics.ppf_asymptotic_values;
        out.student_workspace_growth_events +=
            block.diagnostics.growth_events;
        out.student_workspace_peak_bytes = std::max(
            out.student_workspace_peak_bytes,
            block.diagnostics.peak_capacity_bytes);
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

MultivariateGridResult multivariate_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid,
    std::int64_t row_offset,
    int n_threads) {

    MultivariateGridResult out;
    out.n_threads_requested = n_threads;
    initialize_grid(out, u.size(), x_grid.size());
    if (out.status != SCAR_OK) {
        return out;
    }
    out.status = validate(spec, u, row_offset);
    if (n_threads < 1 || n_threads > 256) {
        out.status = SCAR_INVALID_PARAMETER;
    }
    if (out.status != SCAR_OK || x_grid.empty()) {
        if (out.status == SCAR_OK) {
            out.status = SCAR_INVALID_SIZE;
        }
        return out;
    }
    if (!std::all_of(x_grid.begin(), x_grid.end(), [](double value) {
            return std::isfinite(value);
        })) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }

    std::vector<double> parameter_grid(x_grid.size(), 0.0);
    std::vector<double> dpsi_grid(x_grid.size(), 0.0);
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        parameter_grid[j] =
            scar_internal::copula_transform(spec, x_grid[j]);
        dpsi_grid[j] =
            scar_internal::copula_dtransform(spec, x_grid[j]);
    }

    if (spec.family == CopulaFamily::Student) {
        std::vector<StudentGridBlockResult> block_results(
            static_cast<std::size_t>(n_threads));
        scar_internal::parallel_for_blocks(
            0,
            static_cast<std::int64_t>(u.size()),
            kStudentGridMinRowsPerBlock,
            n_threads,
            [&](std::int64_t begin,
                std::int64_t end,
                std::size_t block) {
                StudentGridBlockResult& block_result = block_results[block];
                block_result.ran = true;
                scar_internal::StudentWorkspace workspace;
                workspace.reserve_x(static_cast<std::size_t>(spec.dim));
                workspace.reserve_dx_ddf(static_cast<std::size_t>(spec.dim));
                for (std::int64_t row_index = begin;
                     row_index < end;
                     ++row_index) {
                    const std::size_t i =
                        static_cast<std::size_t>(row_index);
                    const std::size_t base = i * x_grid.size();
                    scar_internal::student_fill_row_with_workspace(
                        spec,
                        u[i].data(),
                        row_offset + row_index,
                        parameter_grid,
                        dpsi_grid,
                        out.pdf.values.data() + base,
                        out.d_pdf_dx.values.data() + base,
                        workspace);
                    for (std::size_t j = 0; j < x_grid.size(); ++j) {
                        if (!std::isfinite(out.pdf.values[base + j])
                            || !std::isfinite(
                                out.d_pdf_dx.values[base + j])) {
                            block_result.failure_index = row_index;
                            block_result.diagnostics = workspace.diagnostics;
                            return;
                        }
                    }
                }
                block_result.diagnostics = workspace.diagnostics;
            });

        for (const StudentGridBlockResult& block : block_results) {
            if (!block.ran) {
                continue;
            }
            ++out.student_parallel_blocks;
            out.student_ppf_cache_values +=
                block.diagnostics.ppf_cache_values;
            out.student_ppf_exact_values +=
                block.diagnostics.ppf_exact_values;
            out.student_ppf_asymptotic_values +=
                block.diagnostics.ppf_asymptotic_values;
            out.student_workspace_growth_events +=
                block.diagnostics.growth_events;
            out.student_workspace_peak_bytes = std::max(
                out.student_workspace_peak_bytes,
                block.diagnostics.peak_capacity_bytes);
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
                * x_grid.size();
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

    out.student_parallel_blocks = 0;
    const bool use_equicorr_threads =
        n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            u.size(),
            x_grid.size(),
            static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock),
            kEquicorrGridMinCells);
    std::vector<EquicorrGridBlockResult> block_results(
        static_cast<std::size_t>(use_equicorr_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(u.size()),
        kEquicorrGridMinRowsPerBlock,
        use_equicorr_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            EquicorrGridBlockResult& block_result = block_results[block];
            block_result.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t i =
                    static_cast<std::size_t>(row_index);
                const std::size_t base = i * x_grid.size();
                scar_internal::EquicorrStats stats;
                if (!scar_internal::equicorr_sufficient_statistics(
                        spec, u[i].data(), stats)) {
                    block_result.failure_index = row_index;
                    return;
                }
                for (std::size_t j = 0; j < x_grid.size(); ++j) {
                    double dlog = 0.0;
                    const double log_pdf =
                        scar_internal::equicorr_log_pdf_from_stats(
                            spec, stats, parameter_grid[j], &dlog);
                    const double pdf = std::exp(log_pdf);
                    out.pdf.values[base + j] = pdf;
                    out.d_pdf_dx.values[base + j] =
                        pdf * dlog * dpsi_grid[j];
                }
                for (std::size_t j = 0; j < x_grid.size(); ++j) {
                    if (!std::isfinite(out.pdf.values[base + j])
                        || !std::isfinite(
                            out.d_pdf_dx.values[base + j])) {
                        block_result.failure_index = row_index;
                        return;
                    }
                }
            }
        });

    for (const EquicorrGridBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.equicorr_parallel_blocks;
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
            * x_grid.size();
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

ConditionalSampleResult multivariate_gaussian_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return multivariate_gaussian_conditional(
        {correlations.data(), correlations.size()},
        correlation_rows,
        dimension,
        given_indices,
        {given_latent.data(), given_latent.size()},
        {normal_draws.data(), normal_draws.size()},
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_gaussian_conditional(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return conditional_latent(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_latent,
        nullptr,
        normal_draws,
        nullptr,
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_student_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& df,
    const std::vector<double>& normal_draws,
    const std::vector<double>& chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return multivariate_student_conditional(
        {correlations.data(), correlations.size()},
        correlation_rows,
        dimension,
        given_indices,
        {given_latent.data(), given_latent.size()},
        {df.data(), df.size()},
        {normal_draws.data(), normal_draws.size()},
        {chi_square_draws.data(), chi_square_draws.size()},
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_student_conditional(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return conditional_latent(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_latent,
        &df,
        normal_draws,
        &chi_square_draws,
        n_rows,
        n_threads);
}

}  // namespace scar
