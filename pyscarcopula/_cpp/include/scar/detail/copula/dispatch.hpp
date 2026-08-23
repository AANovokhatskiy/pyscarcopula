#pragma once

#include "scar/copula/spec.hpp"

#include <cstdint>
#include <vector>

namespace scar_internal {

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

void copula_fi_row_on_grid(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& x_grid,
    std::vector<double>& fi_row);

}  // namespace scar_internal
