#pragma once

#include "scar/copula.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar_internal {

bool is_valid_rotation(int rotation);
scar::CopulaSpec transposed_copula_spec(const scar::CopulaSpec& spec);
using ConditionalKernel = double (*)(double, double, double);
double softplus(double x);
double d_softplus(double x);
double log1mexp(double x);
double logsumexp(double a, double b);
double normal_quantile(double p);
double normal_quantile_refined(double p);
struct EquicorrStats {
    double sum = 0.0;
    double sum_squares = 0.0;
};
bool equicorr_sufficient_statistics(
    const scar::CopulaSpec& spec,
    const double* row,
    EquicorrStats& stats);
double equicorr_transform(const scar::CopulaSpec& spec, double x);
double equicorr_inverse_transform(const scar::CopulaSpec& spec, double rho);
double equicorr_dtransform(const scar::CopulaSpec& spec, double x);
double equicorr_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double rho,
    double* dlog_drho);
double equicorr_log_pdf_from_stats(
    const scar::CopulaSpec& spec,
    const EquicorrStats& stats,
    double rho,
    double* dlog_drho);
double copula_transform(const scar::CopulaSpec& spec, double x);
double copula_inverse_transform(const scar::CopulaSpec& spec, double r);
double copula_dtransform(const scar::CopulaSpec& spec, double x);
double copula_tau_to_param(const scar::CopulaSpec& spec, double tau);
double copula_param_to_tau(const scar::CopulaSpec& spec, double r);
void apply_rotation(double u1, double u2, int rotation, double& v1, double& v2);
double evaluate_rotated_conditional(
    double first,
    double second,
    double parameter,
    int rotation,
    ConditionalKernel kernel);

double clayton_log_pdf_unrotated(double u1, double u2, double r);
double clayton_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void clayton_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx);
double clayton_h_rotated(double u, double v, double r, int rotation);
double clayton_h_inverse_rotated(double q, double given, double r, int rotation);
double gumbel_log_pdf_unrotated(double u1, double u2, double r);
double gumbel_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void gumbel_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx);
double gumbel_h_rotated(double u, double v, double r, int rotation);
double gumbel_h_inverse_rotated(double q, double given, double r, int rotation);
double frank_log_pdf_unrotated(double u1, double u2, double r);
double frank_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void frank_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx);
double frank_h_rotated(double u, double v, double r, int rotation);
double frank_h_inverse_rotated(double q, double given, double r, int rotation);
double joe_log_pdf_unrotated(double u1, double u2, double r);
double joe_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void joe_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx);
double joe_h_rotated(double u, double v, double r, int rotation);
double joe_h_inverse_rotated(double q, double given, double r, int rotation);
double gaussian_log_pdf_unrotated(double u1, double u2, double rho);
double gaussian_dlog_pdf_dr_unrotated(double u1, double u2, double rho);
void gaussian_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx);
double gaussian_h_from_quantiles(double z_u, double z_v, double rho);
double gaussian_h_rotated(double u, double v, double rho, int rotation);
double gaussian_h_inverse_rotated(double q, double given, double rho, int rotation);
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
double student_quantile_value(double p, double df);
double student_quantile_for_observation(
    const scar::CopulaSpec& spec,
    double p,
    double df,
    std::int64_t row_index,
    int column);
double student_cdf_value(double value, double df);
void student_quantile_value_and_derivative(
    double p, double df, double& value, double& derivative);
void student_quantile_large_df_value_and_derivative(
    double p, double df, double& value, double& derivative);
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

bool copula_is_supported(const scar::CopulaSpec& spec);
bool copula_is_supported_for_ou(const scar::CopulaSpec& spec);
double copula_log_pdf_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r);
double copula_dlog_pdf_dr_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r);
double copula_pdf_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x);
void copula_pdf_and_grad_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx);
void copula_prepare_grid_transform(
    const scar::CopulaSpec& spec,
    const std::vector<double>& x_grid,
    std::vector<double>& r_grid,
    std::vector<double>& dpsi_grid);
void copula_pdf_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    double* fi_row);
void copula_pdf_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    double* fi_row,
    double* log_scale = nullptr);
void copula_pdf_and_grad_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row);
void copula_pdf_and_grad_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    double* log_scale = nullptr);
void copula_pdf_and_grad_grid_precomputed(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t n_obs,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    std::vector<double>& fi,
    std::vector<double>& dfi_dx,
    int n_threads = 1,
    double* log_scale_sum = nullptr);
double copula_h_rotated(
    const scar::CopulaSpec& spec,
    double u,
    double v,
    double r);
double copula_h_inverse_rotated(
    const scar::CopulaSpec& spec,
    double q,
    double given,
    double r);

}  // namespace scar_internal
