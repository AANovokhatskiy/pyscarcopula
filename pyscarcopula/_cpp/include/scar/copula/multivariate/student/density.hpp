#pragma once

#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/copula/spec.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {
struct MultivariateGridResult;
struct MultivariateRowsResult;
}

namespace scar_internal {

struct PreparedStudentDensity {
    int dimension = 0;
    const scar::copula::multivariate::correlation::DenseCorrelation* dense =
        nullptr;
    const scar::FactorCorrelationOperator* factor = nullptr;
    const scar::copula::multivariate::student::PpfCache* ppf_cache = nullptr;
    bool valid = false;
};

PreparedStudentDensity prepare_student_density(
    const scar::CopulaSpec& spec);

struct StudentWorkspace {
    struct Diagnostics {
        std::uint64_t ppf_cache_values = 0;
        std::uint64_t ppf_exact_values = 0;
        std::uint64_t ppf_asymptotic_values = 0;
        std::uint64_t growth_events = 0;
        std::size_t peak_capacity_bytes = 0;
    } diagnostics;
    std::vector<double> x;
    std::vector<double> dx_ddf;
    std::vector<double> precision_x;
    std::vector<double> factor_small;

    void update_peak_capacity() noexcept {
        diagnostics.peak_capacity_bytes = std::max(
            diagnostics.peak_capacity_bytes,
            (x.capacity()
             + dx_ddf.capacity()
             + precision_x.capacity()
             + factor_small.capacity())
                * sizeof(double));
    }

    void reserve_x(std::size_t size) {
        if (size > x.capacity()) {
            ++diagnostics.growth_events;
            x.reserve(size);
        }
        update_peak_capacity();
    }

    void reserve_dx_ddf(std::size_t size) {
        if (size > dx_ddf.capacity()) {
            ++diagnostics.growth_events;
            dx_ddf.reserve(size);
        }
        update_peak_capacity();
    }

    void resize_x(std::size_t size) {
        if (size > x.capacity()) {
            ++diagnostics.growth_events;
        }
        x.resize(size);
        update_peak_capacity();
    }

    void resize_dx_ddf(std::size_t size) {
        if (size > dx_ddf.capacity()) {
            ++diagnostics.growth_events;
        }
        dx_ddf.resize(size);
        update_peak_capacity();
    }

    void resize_precision_x(std::size_t size) {
        if (size > precision_x.capacity()) {
            ++diagnostics.growth_events;
        }
        precision_x.resize(size);
        update_peak_capacity();
    }

    void resize_factor_small(std::size_t size) {
        if (size > factor_small.capacity()) {
            ++diagnostics.growth_events;
        }
        factor_small.resize(size);
        update_peak_capacity();
    }
};

double student_log_pdf(
    const PreparedStudentDensity& model,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace);
double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index);
double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace);
bool student_log_pdf_and_dlog_ddf(
    const PreparedStudentDensity& model,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf,
    StudentWorkspace& workspace);
bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf);
bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf,
    StudentWorkspace& workspace);
bool student_log_pdf_from_quantiles(
    const double* quantiles,
    const double* quantile_derivatives,
    std::size_t dimension,
    double df,
    double logdet,
    double quadratic_form,
    double quadratic_form_derivative,
    double& log_pdf,
    double* dlog_ddf);
bool student_marginal_log_pdf_from_quantile(
    double quantile,
    double quantile_derivative,
    double df,
    double marginal_constant,
    double marginal_constant_derivative,
    double& log_pdf,
    double& dlog_ddf);
bool student_marginal_log_pdf_constants(
    double df,
    double& marginal_constant,
    double& marginal_constant_derivative);
bool student_log_pdf_from_summaries(
    std::size_t dimension,
    double df,
    double logdet,
    double quadratic_form,
    double quadratic_form_derivative,
    double marginal_log_pdf,
    double marginal_dlog_ddf,
    double& log_pdf,
    double* dlog_ddf);
bool student_precision_matrix(
    const scar::CopulaSpec& spec,
    std::vector<double>& precision);
bool student_corr_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    double* scores);
bool student_corr_directional_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    const std::vector<double>& direction,
    double* scores);
void student_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace::Diagnostics* diagnostics = nullptr);
void student_fill_row_with_workspace(
    const PreparedStudentDensity& model,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace& workspace);
void student_fill_row_with_workspace(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace& workspace);
void student_fill_row_from_x_grid(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& x_grid,
    double* fi_row);
bool student_fill_grid_bivariate(
    const scar::CopulaSpec& spec,
    std::int64_t n_obs,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi,
    double* dfi_dx,
    int n_threads = 1);

scar::MultivariateRowsResult student_log_pdf_and_grad_rows(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& degrees_of_freedom,
    std::int64_t row_offset,
    int n_threads);
scar::MultivariateGridResult student_pdf_and_grad_grid(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& state_grid,
    std::int64_t row_offset,
    int n_threads);

}  // namespace scar_internal
