#pragma once

#include "scar/copula.hpp"
#include "scar/detail/safety.hpp"

#include <cstdint>
#include <vector>

namespace scar_internal {

bool is_valid_rotation(int rotation);
using ConditionalKernel = double (*)(double, double, double);
double softplus(double x);
double d_softplus(double x);
double log1mexp(double x);
double logsumexp(double a, double b);
double normal_quantile(double p);
double normal_quantile_refined(double p);
double equicorr_transform(const scar::CopulaSpec& spec, double x);
double equicorr_inverse_transform(const scar::CopulaSpec& spec, double rho);
double equicorr_dtransform(const scar::CopulaSpec& spec, double x);
double equicorr_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
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
double gaussian_h_rotated(double u, double v, double rho, int rotation);
double gaussian_h_inverse_rotated(double q, double given, double rho, int rotation);
double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index);
bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf);
double student_quantile_value(double p, double df);
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
void student_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row);
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
    double* dfi_dx);

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
    double* fi_row);
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
    double* dfi_dx_row);
void copula_pdf_and_grad_grid_precomputed(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t n_obs,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    std::vector<double>& fi,
    std::vector<double>& dfi_dx);
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
