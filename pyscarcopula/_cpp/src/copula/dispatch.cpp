#include "scar/copula/pair/gaussian.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/dispatch.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/factor.hpp"
#include "scar/math/normal.hpp"

#include <cmath>
#include <limits>

namespace scar_internal {

namespace {

constexpr std::int64_t kEquicorrGridMinRowsPerBlock = 64;
constexpr std::size_t kEquicorrGridMinCells = 262144;

bool gaussian_cache_available(
    const scar::CopulaSpec& spec,
    std::int64_t row_index) {

    return row_index >= 0
        && spec.pair_gaussian_first_scores().size()
            == spec.pair_gaussian_second_scores().size()
        && static_cast<std::size_t>(row_index)
            < spec.pair_gaussian_first_scores().size();
}

void gaussian_fill_row(
    const scar::CopulaSpec& spec,
    double first,
    double second,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    double rotated_first = 0.0;
    double rotated_second = 0.0;
    scar::copula::apply_rotation(
        first,
        second,
        static_cast<int>(spec.rotation),
        rotated_first,
        rotated_second);
    const double z1 = scar::math::normal_quantile(rotated_first);
    const double z2 = scar::math::normal_quantile(rotated_second);
    scar::copula::pair::gaussian_fill_grid_row_from_stats(
        z1 * z1 + z2 * z2,
        z1 * z2,
        r_grid,
        dpsi_grid,
        fi_row,
        dfi_dx_row);
}

void equicorr_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    EquicorrStats stats;
    const bool cache_available =
        row_index >= 0
        && spec.equicorr_sum_scores().size()
            == spec.equicorr_sum_squares().size()
        && static_cast<std::size_t>(row_index)
            < spec.equicorr_sum_scores().size();
    if (cache_available) {
        const std::size_t index = static_cast<std::size_t>(row_index);
        stats.sum = spec.equicorr_sum_scores()[index];
        stats.sum_squares = spec.equicorr_sum_squares()[index];
    } else if (!equicorr_sufficient_statistics(spec, row, stats)) {
        std::fill(
            fi_row,
            fi_row + r_grid.size(),
            std::numeric_limits<double>::quiet_NaN());
        if (dfi_dx_row != nullptr) {
            std::fill(
                dfi_dx_row,
                dfi_dx_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
        }
        return;
    }
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        double dlog_dr = 0.0;
        const double log_pdf = equicorr_log_pdf_from_stats(
            spec,
            stats,
            r_grid[j],
            dfi_dx_row == nullptr ? nullptr : &dlog_dr);
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

}  // namespace

bool copula_is_supported(const scar::CopulaSpec& spec) {
    const scar::PreparedPairKernel pair_kernel(spec);
    if (pair_kernel.is_registered()) {
        return pair_kernel.is_supported();
    }
    if (!scar::copula::is_valid_rotation(static_cast<int>(spec.rotation))) {
        return false;
    }
    if (!std::isfinite(spec.offset) || spec.offset < 0.0) {
        return false;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return spec.rotation == scar::Rotation::R0
            && spec.transform == scar::Transform::GaussianTanh
            && spec.dim >= 2;
    }
    if (spec.family == scar::CopulaFamily::Student) {
        std::size_t expected = 0;
        if (!valid_student_dimension(spec.dim, expected)) {
            return false;
        }
        if (spec.correlation_kind == scar::CorrelationKind::Factor) {
            return spec.rotation == scar::Rotation::R0
                && spec.transform == scar::Transform::Softplus
                && spec.offset >= 2.0
                && spec.dim >= 2
                && spec.factor_operator() != nullptr
                && spec.factor_operator()->dimension()
                    == static_cast<std::size_t>(spec.dim)
                && std::isfinite(
                    spec.factor_operator()->logdet());
        }
        const bool valid_values = std::all_of(
            spec.dense_inverse_cholesky().begin(), spec.dense_inverse_cholesky().end(), [](double value) {
                return std::isfinite(value);
            });
        bool lower_triangular =
            spec.dim >= 2 && spec.dense_inverse_cholesky().size() == expected;
        if (lower_triangular) {
            for (int i = 0; i < spec.dim && lower_triangular; ++i) {
                for (int j = i + 1; j < spec.dim; ++j) {
                    const std::size_t index =
                        static_cast<std::size_t>(i)
                            * static_cast<std::size_t>(spec.dim)
                        + static_cast<std::size_t>(j);
                    if (std::abs(spec.dense_inverse_cholesky()[index]) > 1e-14) {
                        lower_triangular = false;
                        break;
                    }
                }
            }
        }
        return spec.rotation == scar::Rotation::R0
            && spec.transform == scar::Transform::Softplus
            && spec.offset >= 2.0
            && spec.dim >= 2
            && spec.dense_inverse_cholesky().size() == expected
            && std::isfinite(spec.dense_log_determinant())
            && valid_values
            && lower_triangular;
    }
    return false;
}

bool copula_is_supported_for_ou(const scar::CopulaSpec& spec) {
    return copula_is_supported(spec);
}

double copula_log_pdf_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        return kernel.log_pdf_unrotated(u1, u2, r);
    }
    return -std::numeric_limits<double>::infinity();
}

double copula_dlog_pdf_dr_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        return kernel.dlog_pdf_dparameter_unrotated(u1, u2, r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_pdf_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        return kernel.pdf_unrotated(u1, u2, x);
    }
    const double r = copula_transform(spec, x);
    return std::exp(copula_log_pdf_unrotated(spec, u1, u2, r));
}

void copula_pdf_and_grad_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        kernel.pdf_and_gradient_unrotated(u1, u2, x, pdf, d_pdf_dx);
        return;
    }

    const double r = copula_transform(spec, x);
    const double d_r_dx = copula_dtransform(spec, x);
    const double log_pdf = copula_log_pdf_unrotated(spec, u1, u2, r);
    pdf = std::exp(log_pdf);
    d_pdf_dx = pdf
        * copula_dlog_pdf_dr_unrotated(spec, u1, u2, r)
        * d_r_dx;
}

void copula_prepare_grid_transform(
    const scar::CopulaSpec& spec,
    const std::vector<double>& x_grid,
    std::vector<double>& r_grid,
    std::vector<double>& dpsi_grid) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        kernel.prepare_parameter_grid(x_grid, r_grid, dpsi_grid);
        return;
    }
    r_grid.resize(x_grid.size());
    dpsi_grid.resize(x_grid.size());
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        r_grid[j] = copula_transform(spec, x_grid[j]);
        dpsi_grid[j] = copula_dtransform(spec, x_grid[j]);
    }
}

void copula_pdf_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    double* fi_row) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        kernel.fill_grid_row(u1, u2, r_grid, fi_row);
        return;
    }

    double v1 = 0.0;
    double v2 = 0.0;
    scar::copula::apply_rotation(
        u1, u2, static_cast<int>(spec.rotation), v1, v2);
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        fi_row[j] = std::exp(copula_log_pdf_unrotated(spec, v1, v2, r_grid[j]));
    }
}

void copula_pdf_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    double* fi_row,
    double* log_scale) {

    if (log_scale != nullptr) {
        *log_scale = 0.0;
    }
    if (spec.family == scar::CopulaFamily::Student
        && spec.correlation_kind == scar::CorrelationKind::Factor
        && spec.factor_operator() != nullptr) {
        const std::size_t row_offset =
            static_cast<std::size_t>(t)
            * static_cast<std::size_t>(spec.dim);
        const scar::FactorStudentGridResult result =
            scar::factor_student_log_pdf_and_dlog_ddf_grid(
                *spec.factor_operator(),
                u + row_offset,
                1,
                r_grid.data(),
                r_grid.size(),
                spec.factor_dimension_tile(),
                1);
        if (result.failure.index >= 0
            || result.log_pdf.size() != r_grid.size()) {
            std::fill(
                fi_row,
                fi_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const double row_scale = *std::max_element(
            result.log_pdf.begin(), result.log_pdf.end());
        if (!std::isfinite(row_scale)) {
            std::fill(
                fi_row,
                fi_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
            return;
        }
        for (std::size_t j = 0; j < r_grid.size(); ++j) {
            fi_row[j] = std::exp(result.log_pdf[j] - row_scale);
        }
        if (log_scale != nullptr) {
            *log_scale = row_scale;
        }
        return;
    }

    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        static const std::vector<double> no_dpsi;
        const bool cache_available =
            t >= 0
            && spec.equicorr_sum_scores().size()
                == spec.equicorr_sum_squares().size()
            && static_cast<std::size_t>(t)
                < spec.equicorr_sum_scores().size();
        const double* row = cache_available || u == nullptr
            ? nullptr
            : u + static_cast<std::size_t>(t)
                * static_cast<std::size_t>(spec.dim);
        equicorr_fill_row(
            spec, row, t, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    const int stride =
        spec.family == scar::CopulaFamily::Student ? spec.dim : 2;
    const double* row =
        u + static_cast<std::size_t>(t) * static_cast<std::size_t>(stride);
    if (spec.family == scar::CopulaFamily::Student) {
        static const std::vector<double> no_dpsi;
        student_fill_row(spec, row, t, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian
        && gaussian_cache_available(spec, t)) {
        static const std::vector<double> no_dpsi;
        const std::size_t index = static_cast<std::size_t>(t);
        const double z1 = spec.pair_gaussian_first_scores()[index];
        const double z2 = spec.pair_gaussian_second_scores()[index];
        scar::copula::pair::gaussian_fill_grid_row_from_stats(
            z1 * z1 + z2 * z2,
            z1 * z2,
            r_grid,
            no_dpsi,
            fi_row,
            nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        static const std::vector<double> no_dpsi;
        gaussian_fill_row(
            spec,
            row[0],
            row[1],
            r_grid,
            no_dpsi,
            fi_row,
            nullptr);
        return;
    }
    copula_pdf_row_precomputed(spec, row[0], row[1], r_grid, fi_row);
}

void copula_pdf_and_grad_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        kernel.fill_grid_row_with_gradient(
            u1,
            u2,
            r_grid,
            dpsi_grid,
            fi_row,
            dfi_dx_row);
        return;
    }

    double v1 = 0.0;
    double v2 = 0.0;
    scar::copula::apply_rotation(
        u1, u2, static_cast<int>(spec.rotation), v1, v2);
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double log_pdf = copula_log_pdf_unrotated(spec, v1, v2, r_grid[j]);
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        dfi_dx_row[j] = pdf
            * copula_dlog_pdf_dr_unrotated(spec, v1, v2, r_grid[j])
            * dpsi_grid[j];
    }
}

void copula_pdf_and_grad_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    double* log_scale) {

    if (log_scale != nullptr) {
        *log_scale = 0.0;
    }
    if (spec.family == scar::CopulaFamily::Student
        && spec.correlation_kind == scar::CorrelationKind::Factor
        && spec.factor_operator() != nullptr) {
        const std::size_t row_offset =
            static_cast<std::size_t>(t)
            * static_cast<std::size_t>(spec.dim);
        const scar::FactorStudentGridResult result =
            scar::factor_student_log_pdf_and_dlog_ddf_grid(
                *spec.factor_operator(),
                u + row_offset,
                1,
                r_grid.data(),
                r_grid.size(),
                spec.factor_dimension_tile(),
                1);
        if (result.failure.index >= 0
            || result.log_pdf.size() != r_grid.size()
            || result.dlog_ddf.size() != r_grid.size()) {
            std::fill(
                fi_row,
                fi_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
            std::fill(
                dfi_dx_row,
                dfi_dx_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const double row_scale = *std::max_element(
            result.log_pdf.begin(), result.log_pdf.end());
        if (!std::isfinite(row_scale)) {
            std::fill(
                fi_row,
                fi_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
            std::fill(
                dfi_dx_row,
                dfi_dx_row + r_grid.size(),
                std::numeric_limits<double>::quiet_NaN());
            return;
        }
        for (std::size_t j = 0; j < r_grid.size(); ++j) {
            const double density = std::exp(result.log_pdf[j] - row_scale);
            fi_row[j] = density;
            dfi_dx_row[j] =
                density * result.dlog_ddf[j] * dpsi_grid[j];
        }
        if (log_scale != nullptr) {
            *log_scale = row_scale;
        }
        return;
    }

    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        const bool cache_available =
            t >= 0
            && spec.equicorr_sum_scores().size()
                == spec.equicorr_sum_squares().size()
            && static_cast<std::size_t>(t)
                < spec.equicorr_sum_scores().size();
        const double* row = cache_available || u == nullptr
            ? nullptr
            : u + static_cast<std::size_t>(t)
                * static_cast<std::size_t>(spec.dim);
        equicorr_fill_row(
            spec, row, t, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    const int stride =
        spec.family == scar::CopulaFamily::Student ? spec.dim : 2;
    const double* row =
        u + static_cast<std::size_t>(t) * static_cast<std::size_t>(stride);
    if (spec.family == scar::CopulaFamily::Student) {
        student_fill_row(
            spec, row, t, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian
        && gaussian_cache_available(spec, t)) {
        const std::size_t index = static_cast<std::size_t>(t);
        const double z1 = spec.pair_gaussian_first_scores()[index];
        const double z2 = spec.pair_gaussian_second_scores()[index];
        scar::copula::pair::gaussian_fill_grid_row_from_stats(
            z1 * z1 + z2 * z2,
            z1 * z2,
            r_grid,
            dpsi_grid,
            fi_row,
            dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        gaussian_fill_row(
            spec,
            row[0],
            row[1],
            r_grid,
            dpsi_grid,
            fi_row,
            dfi_dx_row);
        return;
    }
    copula_pdf_and_grad_row_precomputed(
        spec, row[0], row[1], r_grid, dpsi_grid, fi_row, dfi_dx_row);
}

void copula_pdf_and_grad_grid_precomputed(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t n_obs,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    std::vector<double>& fi,
    std::vector<double>& dfi_dx,
    int n_threads,
    double* log_scale_sum,
    std::int64_t first_row,
    double* row_log_scales) {

    if (log_scale_sum != nullptr) {
        *log_scale_sum = 0.0;
    }

    const std::size_t K = r_grid.size();
    std::size_t n_obs_size = 0;
    std::size_t elements = 0;
    if (first_row < 0 || n_obs < 0
        || n_obs > std::numeric_limits<std::int64_t>::max() - first_row
        || K == 0 || dpsi_grid.size() != K
        || !checked_nonnegative_size(n_obs, n_obs_size)
        || !checked_size_mul(n_obs_size, K, elements)) {
        fi.clear();
        dfi_dx.clear();
        return;
    }
    const std::int64_t end_row = first_row + n_obs;
    std::size_t observation_elements = 0;
    if (spec.dim < 1 || !checked_size_mul(
            static_cast<std::size_t>(end_row),
            static_cast<std::size_t>(spec.dim), observation_elements)) {
        fi.clear();
        dfi_dx.clear();
        return;
    }
    if (row_log_scales != nullptr) {
        std::fill(row_log_scales, row_log_scales + n_obs_size, 0.0);
    }
    fi.assign(elements, 0.0);
    dfi_dx.assign(elements, 0.0);
    if (spec.family == scar::CopulaFamily::Student
        && spec.correlation_kind == scar::CorrelationKind::Factor
        && spec.factor_operator() != nullptr) {
        const scar::FactorStudentGridResult result =
            scar::factor_student_log_pdf_and_dlog_ddf_grid(
                *spec.factor_operator(),
                u == nullptr ? nullptr
                    : u + static_cast<std::size_t>(first_row)
                        * static_cast<std::size_t>(spec.dim),
                n_obs_size,
                r_grid.data(),
                K,
                spec.factor_dimension_tile(),
                n_threads);
        if (result.failure.index >= 0
            || result.log_pdf.size() != elements
            || result.dlog_ddf.size() != elements) {
            std::fill(
                fi.begin(),
                fi.end(),
                std::numeric_limits<double>::quiet_NaN());
            std::fill(
                dfi_dx.begin(),
                dfi_dx.end(),
                std::numeric_limits<double>::quiet_NaN());
            return;
        }
        double total_scale = 0.0;
        for (std::size_t row = 0; row < n_obs_size; ++row) {
            const std::size_t offset = row * K;
            const auto row_begin =
                result.log_pdf.begin()
                + static_cast<std::ptrdiff_t>(offset);
            const double row_scale = *std::max_element(
                row_begin,
                row_begin + static_cast<std::ptrdiff_t>(K));
            if (!std::isfinite(row_scale)) {
                std::fill(
                    fi.begin(),
                    fi.end(),
                    std::numeric_limits<double>::quiet_NaN());
                std::fill(
                    dfi_dx.begin(),
                    dfi_dx.end(),
                    std::numeric_limits<double>::quiet_NaN());
                return;
            }
            total_scale += row_scale;
            if (row_log_scales != nullptr) {
                row_log_scales[row] = row_scale;
            }
            for (std::size_t grid = 0; grid < K; ++grid) {
                const std::size_t index = offset + grid;
                const double density =
                    std::exp(result.log_pdf[index] - row_scale);
                fi[index] = density;
                dfi_dx[index] = density
                    * result.dlog_ddf[index]
                    * dpsi_grid[grid];
            }
        }
        if (log_scale_sum != nullptr) {
            *log_scale_sum = total_scale;
        }
        return;
    }
    if (spec.family == scar::CopulaFamily::Student
        && spec.dim == 2
        && student_fill_grid_bivariate(
            spec,
            n_obs,
            r_grid,
            dpsi_grid,
            fi.data(),
            dfi_dx.data(),
            n_threads,
            first_row)) {
        return;
    }
    if (spec.family == scar::CopulaFamily::Student) {
        constexpr std::int64_t min_rows_per_block = 8;
        parallel_for_blocks(
            first_row,
            end_row,
            min_rows_per_block,
            n_threads,
            [&](std::int64_t begin,
                std::int64_t end,
                std::size_t) {
                StudentWorkspace workspace;
                workspace.reserve_x(static_cast<std::size_t>(spec.dim));
                workspace.reserve_dx_ddf(
                    static_cast<std::size_t>(spec.dim));
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::size_t output_row =
                        static_cast<std::size_t>(t - first_row) * K;
                    const double* observation_row =
                        u + static_cast<std::size_t>(t)
                            * static_cast<std::size_t>(spec.dim);
                    student_fill_row_with_workspace(
                        spec,
                        observation_row,
                        t,
                        r_grid,
                        dpsi_grid,
                        fi.data() + output_row,
                        dfi_dx.data() + output_row,
                        workspace);
                }
            });
        return;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian
        && n_threads > 1
        && grid_parallel_worthwhile(
            n_obs_size,
            K,
            static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock),
            kEquicorrGridMinCells)) {
        parallel_for_blocks(
            first_row,
            end_row,
            kEquicorrGridMinRowsPerBlock,
            n_threads,
            [&](std::int64_t begin,
                std::int64_t end,
                std::size_t) {
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::size_t output_row =
                        static_cast<std::size_t>(t - first_row) * K;
                    const bool cache_available =
                        spec.equicorr_sum_scores().size()
                            == spec.equicorr_sum_squares().size()
                        && static_cast<std::size_t>(t)
                            < spec.equicorr_sum_scores().size();
                    const double* observation_row =
                        cache_available || u == nullptr
                        ? nullptr
                        : u + static_cast<std::size_t>(t)
                            * static_cast<std::size_t>(spec.dim);
                    equicorr_fill_row(
                        spec,
                        observation_row,
                        t,
                        r_grid,
                        dpsi_grid,
                        fi.data() + output_row,
                        dfi_dx.data() + output_row);
                }
            });
        return;
    }
    const scar::PreparedPairKernel pair_kernel(spec);
    if (pair_kernel.is_registered()) {
        if (spec.family == scar::CopulaFamily::Gaussian) {
            const bool cache_available =
                spec.pair_gaussian_first_scores().size()
                    == spec.pair_gaussian_second_scores().size()
                && spec.pair_gaussian_first_scores().size()
                    >= static_cast<std::size_t>(end_row);
            for (std::int64_t t = first_row; t < end_row; ++t) {
                const std::size_t observation = static_cast<std::size_t>(t);
                const std::size_t output_row =
                    static_cast<std::size_t>(t - first_row) * K;
                if (!cache_available) {
                    const double* observation_row = u + observation * 2;
                    gaussian_fill_row(
                        spec,
                        observation_row[0],
                        observation_row[1],
                        r_grid,
                        dpsi_grid,
                        fi.data() + output_row,
                        dfi_dx.data() + output_row);
                    continue;
                }
                const double z1 = spec.pair_gaussian_first_scores()[observation];
                const double z2 = spec.pair_gaussian_second_scores()[observation];
                scar::copula::pair::gaussian_fill_grid_row_from_stats(
                    z1 * z1 + z2 * z2,
                    z1 * z2,
                    r_grid,
                    dpsi_grid,
                    fi.data() + output_row,
                    dfi_dx.data() + output_row);
            }
        } else {
            for (std::int64_t t = first_row; t < end_row; ++t) {
                const std::size_t observation =
                    static_cast<std::size_t>(t);
                const std::size_t output_row =
                    static_cast<std::size_t>(t - first_row) * K;
                const double* observation_row = u + observation * 2;
                pair_kernel.fill_grid_row_with_gradient(
                    observation_row[0],
                    observation_row[1],
                    r_grid,
                    dpsi_grid,
                    fi.data() + output_row,
                    dfi_dx.data() + output_row);
            }
        }
        return;
    }
    for (std::int64_t t = first_row; t < end_row; ++t) {
        const std::size_t row = static_cast<std::size_t>(t - first_row) * K;
        copula_pdf_and_grad_row_precomputed_flat(
            spec,
            u,
            t,
            r_grid,
            dpsi_grid,
            fi.data() + row,
            dfi_dx.data() + row);
    }
}

double copula_h_rotated(
    const scar::CopulaSpec& spec,
    double u,
    double v,
    double r) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        return kernel.h(u, v, r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_h_inverse_rotated(
    const scar::CopulaSpec& spec,
    double q,
    double given,
    double r) {

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        return kernel.inverse_h(q, given, r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void copula_fi_row_on_grid(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& x_grid,
    std::vector<double>& fi_row) {

    const int stride =
        (spec.family == scar::CopulaFamily::Student
         || spec.family == scar::CopulaFamily::EquicorrGaussian)
        ? spec.dim
        : 2;
    const double* row =
        u + static_cast<std::size_t>(t) * static_cast<std::size_t>(stride);
    if (spec.family == scar::CopulaFamily::Student) {
        student_fill_row_from_x_grid(
            spec, row, t, x_grid, fi_row.data());
        return;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        std::vector<double> r_grid;
        std::vector<double> dpsi_grid;
        copula_prepare_grid_transform(
            spec, x_grid, r_grid, dpsi_grid);
        equicorr_fill_row(
            spec, row, t, r_grid, dpsi_grid, fi_row.data(), nullptr);
        return;
    }

    const scar::PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        double v1 = 0.0;
        double v2 = 0.0;
        scar::copula::apply_rotation(
            row[0],
            row[1],
            static_cast<int>(spec.rotation),
            v1,
            v2);
        for (std::size_t j = 0; j < x_grid.size(); ++j) {
            fi_row[j] = kernel.pdf_unrotated(v1, v2, x_grid[j]);
        }
    }
}

}  // namespace scar_internal
