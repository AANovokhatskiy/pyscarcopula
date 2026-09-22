#include "scar/copula/multivariate/equicorrelation/kernel.hpp"

#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {

using scar::math::normal_quantile_refined;

double equicorr_transform(const scar::CopulaSpec& spec, double x) {
    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    const double tail = std::exp(-2.0 * std::abs(x));
    const double distance = (1.0 - rho_min) * tail / (1.0 + tail);
    const double rho = x >= 0.0 ? 1.0 - distance : rho_min + distance;
    return std::clamp(rho, std::nextafter(rho_min, 1.0),
                      std::nextafter(1.0, rho_min));
}

double equicorr_inverse_transform(
    const scar::CopulaSpec& spec,
    double rho) {

    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    return 0.5 * (std::log(rho - rho_min) - std::log1p(-rho));
}

double equicorr_dtransform(const scar::CopulaSpec& spec, double x) {
    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    const double tail = std::exp(-2.0 * std::abs(x));
    const double distance = (1.0 - rho_min) * tail / (1.0 + tail);
    const double rho = x >= 0.0 ? 1.0 - distance : rho_min + distance;
    if (rho <= rho_min || rho >= 1.0) {
        return 0.0;
    }
    return 2.0 * (1.0 - rho_min) * tail / ((1.0 + tail) * (1.0 + tail));
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
    double mean = 0.0;
    stats.centered_squares = 0.0;
    for (int j = 0; j < spec.dim; ++j) {
        const double z = normal_quantile_refined(row[j]);
        stats.sum_squares += z * z;
        stats.sum += z;
        const double delta = z - mean;
        mean += delta / static_cast<double>(j + 1);
        stats.centered_squares += delta * (z - mean);
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
        std::fma(static_cast<double>(spec.dim - 1), rho, 1.0);
    if (one_minus_rho <= 0.0 || common_eigenvalue <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double dimension = static_cast<double>(spec.dim);
    const double mean_component = (stats.sum / dimension) * stats.sum;
    double centered = stats.centered_squares;
    if (!std::isfinite(centered)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (centered >= 0.0) {
        // Raw sums and Welford residuals follow different reduction paths.
        // Allow their dimension-dependent rounding error, not inconsistent
        // externally supplied prepared statistics.
        const double reconstructed = std::max(0.0, std::fma(
            -stats.sum / dimension, stats.sum, stats.sum_squares));
        const double consistency_tolerance = 64.0
            * std::numeric_limits<double>::epsilon() * dimension
            * std::max({1.0, stats.sum_squares, mean_component});
        if (std::abs(centered - reconstructed) > consistency_tolerance) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    if (centered < 0.0) {
        centered = std::fma(-stats.sum / dimension, stats.sum,
                            stats.sum_squares);
        // Legacy prepared data cannot resolve variance below the rounding
        // uncertainty of its two sums. Reject appreciable amplification rather
        // than erase a small, possibly real positive variance.
        const double uncertainty = 8.0 * std::numeric_limits<double>::epsilon()
            * std::max(stats.sum_squares, mean_component);
        if (centered <= uncertainty
            && 0.5 * std::abs(rho) / one_minus_rho * uncertainty > 1e-6) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        centered = std::max(0.0, centered);
    }
    const double log_det =
        static_cast<double>(spec.dim - 1) * std::log1p(-rho)
        + std::log(common_eigenvalue);

    if (dlog_drho != nullptr) {
        const double dlog_det =
            -(dimension - 1.0) * dimension * rho
            / (one_minus_rho * common_eigenvalue);
        *dlog_drho =
            -0.5 * dlog_det
            -0.5 * (
                centered / (one_minus_rho * one_minus_rho)
                - (dimension - 1.0) * mean_component
                    / (common_eigenvalue * common_eigenvalue));
    }
    return -0.5 * log_det
        -0.5 * (
            rho / one_minus_rho * centered
            - (dimension - 1.0) * rho / common_eigenvalue * mean_component);
}

}  // namespace scar_internal
