#pragma once

#include "scar/copula/spec.hpp"

#include <cstdint>
#include <vector>

namespace scar {
struct MultivariateGridResult;
struct MultivariateRowsResult;
}

namespace scar_internal {

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

scar::MultivariateRowsResult equicorr_log_pdf_and_grad_rows(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& correlations,
    std::int64_t row_offset,
    int n_threads);
scar::MultivariateGridResult equicorr_pdf_and_grad_grid(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& state_grid,
    std::int64_t row_offset,
    int n_threads);

}  // namespace scar_internal
