#pragma once

#include "gradient_workspace.hpp"
#include "scar/copula/spec.hpp"
#include "scar/copula/rotation.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar::spectral_detail {

inline bool prepare_gaussian_spectral_terms(
    const CopulaSpec& copula,
    const std::vector<double>& r_grid,
    ScarOuSpectralGradientWorkspace& workspace) {

    if (copula.family != CopulaFamily::Gaussian) return false;

    const std::size_t grid_size = r_grid.size();
    workspace.gaussian_r2.resize(grid_size);
    workspace.gaussian_omr2.resize(grid_size);
    workspace.gaussian_log_norm.resize(grid_size);
    workspace.gaussian_dlog_det.resize(grid_size);
    workspace.gaussian_omr2_squared.resize(grid_size);
    for (std::size_t j = 0; j < grid_size; ++j) {
        const double r = r_grid[j];
        const double r2 = r * r;
        const double omr2 = 1.0 - r2;
        workspace.gaussian_r2[j] = r2;
        workspace.gaussian_omr2[j] = omr2;
        workspace.gaussian_log_norm[j] = -0.5 * std::log(omr2);
        workspace.gaussian_dlog_det[j] = r / omr2;
        workspace.gaussian_omr2_squared[j] = omr2 * omr2;
    }
    return true;
}

inline void gaussian_spectral_pdf_and_grad_row(
    const CopulaSpec& copula,
    std::int64_t row,
    const double* observations,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    const ScarOuSpectralGradientWorkspace& workspace,
    double* fi_row,
    double* dfi_dx_row,
    double& log_scale) {

    const std::size_t row_index = static_cast<std::size_t>(row);
    double z1, z2;
    if (row_index < copula.pair_gaussian_first_scores().size() &&
        row_index < copula.pair_gaussian_second_scores().size()) {
        z1 = copula.pair_gaussian_first_scores()[row_index];
        z2 = copula.pair_gaussian_second_scores()[row_index];
    } else {
        double u1, u2;
        scar::copula::apply_rotation(observations[2 * row_index],
            observations[2 * row_index + 1], static_cast<int>(copula.rotation), u1, u2);
        z1 = scar::math::normal_quantile(u1);
        z2 = scar::math::normal_quantile(u2);
    }
    const double sum_squares = z1 * z1 + z2 * z2;
    const double cross_product = z1 * z2;
    log_scale = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double r2 = workspace.gaussian_r2[j];
        const double omr2 = workspace.gaussian_omr2[j];
        const double numerator =
            r2 * sum_squares - 2.0 * r * cross_product;
        const double log_pdf =
            workspace.gaussian_log_norm[j] - 0.5 * numerator / omr2;
        fi_row[j] = log_pdf;
        log_scale = std::max(log_scale, log_pdf);

        if (dfi_dx_row == nullptr) continue;
        const double derivative_numerator =
            (2.0 * r * sum_squares - 2.0 * cross_product) * omr2
            + 2.0 * r * numerator;
        const double derivative_quadratic = derivative_numerator
            / workspace.gaussian_omr2_squared[j];
        const double derivative_log_pdf =
            workspace.gaussian_dlog_det[j]
            - 0.5 * derivative_quadratic;
        dfi_dx_row[j] = derivative_log_pdf * dpsi_grid[j];
    }
    // Match PreparedDynamicEmission's scaled-density convention: derivatives
    // are exp(-log_scale) * d(pdf), with the row scale held fixed.
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        fi_row[j] = std::exp(fi_row[j] - log_scale);
        if (dfi_dx_row) dfi_dx_row[j] *= fi_row[j];
    }
}

}  // namespace scar::spectral_detail
