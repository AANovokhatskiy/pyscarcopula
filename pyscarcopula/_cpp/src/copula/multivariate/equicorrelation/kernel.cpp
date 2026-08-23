#include "scar/copula/multivariate/equicorrelation/kernel.hpp"

#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {

using scar::math::normal_quantile_refined;

double equicorr_transform(const scar::CopulaSpec& spec, double x) {
    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    return rho_min
        + 0.5 * (1.0 - rho_min) * (1.0 + std::tanh(x));
}

double equicorr_inverse_transform(
    const scar::CopulaSpec& spec,
    double rho) {

    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    double scaled =
        2.0 * (rho - rho_min) / (1.0 - rho_min) - 1.0;
    scaled = std::clamp(scaled, -0.999999, 0.999999);
    return std::atanh(scaled);
}

double equicorr_dtransform(const scar::CopulaSpec& spec, double x) {
    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    const double th = std::tanh(x);
    return 0.5 * (1.0 - rho_min) * (1.0 - th * th);
}

double equicorr_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double rho,
    double* dlog_drho) {

    EquicorrStats stats;
    if (!equicorr_sufficient_statistics(spec, row, stats)) {
        return -std::numeric_limits<double>::infinity();
    }
    return equicorr_log_pdf_from_stats(spec, stats, rho, dlog_drho);
}

bool equicorr_sufficient_statistics(
    const scar::CopulaSpec& spec,
    const double* row,
    EquicorrStats& stats) {

    if (row == nullptr) {
        return false;
    }
    stats = EquicorrStats{};
    for (int j = 0; j < spec.dim; ++j) {
        const double z = normal_quantile_refined(row[j]);
        stats.sum_squares += z * z;
        stats.sum += z;
    }
    return std::isfinite(stats.sum) && std::isfinite(stats.sum_squares);
}

double equicorr_log_pdf_from_stats(
    const scar::CopulaSpec& spec,
    const EquicorrStats& stats,
    double rho,
    double* dlog_drho) {

    const double one_minus_rho = 1.0 - rho;
    const double common_eigenvalue =
        1.0 + static_cast<double>(spec.dim - 1) * rho;
    if (one_minus_rho <= 0.0 || common_eigenvalue <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double square_sum = stats.sum * stats.sum;
    const double log_det =
        static_cast<double>(spec.dim - 1) * std::log(one_minus_rho)
        + std::log(common_eigenvalue);
    const double diagonal_term = rho / one_minus_rho;
    const double common_term =
        -rho / (one_minus_rho * common_eigenvalue);

    if (dlog_drho != nullptr) {
        const double dlog_det =
            -static_cast<double>(spec.dim - 1) / one_minus_rho
            + static_cast<double>(spec.dim - 1) / common_eigenvalue;
        const double ddiagonal =
            1.0 / (one_minus_rho * one_minus_rho);
        const double dcommon =
            -(common_eigenvalue
              - rho * one_minus_rho
                  * static_cast<double>(spec.dim - 1))
            / std::pow(one_minus_rho * common_eigenvalue, 2.0);
        *dlog_drho =
            -0.5 * dlog_det
            -0.5 * (
                ddiagonal * stats.sum_squares + dcommon * square_sum);
    }
    return -0.5 * log_det
        -0.5 * (
            diagonal_term * stats.sum_squares + common_term * square_sum);
}

}  // namespace scar_internal
