#include "scar/copula/pair/gaussian.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {

using scar::math::normal_quantile;

namespace {

double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

}  // namespace

double gaussian_tau_to_parameter(double tau) {
    if (!std::isfinite(tau) || tau <= -1.0 || tau >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sin(0.5 * kPi * tau);
}

double gaussian_parameter_to_tau(double parameter) {
    if (!std::isfinite(parameter)
        || parameter <= -1.0
        || parameter >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 2.0 * std::asin(parameter) / kPi;
}

double gaussian_log_pdf_unrotated(double u1, double u2, double rho) {
    const double v1 = clip_pseudo_observation(u1);
    const double v2 = clip_pseudo_observation(u2);
    const double x1 = normal_quantile(v1);
    const double x2 = normal_quantile(v2);
    const double r2 = rho * rho;
    const double omr2 = 1.0 - r2;
    return -0.5 * std::log(omr2)
        - 0.5 * (r2 * (x1 * x1 + x2 * x2) - 2.0 * rho * x1 * x2) / omr2;
}

double gaussian_dlog_pdf_dr_unrotated(double u1, double u2, double rho) {
    const double v1 = clip_pseudo_observation(u1);
    const double v2 = clip_pseudo_observation(u2);
    const double x1 = normal_quantile(v1);
    const double x2 = normal_quantile(v2);
    const double r2 = rho * rho;
    const double omr2 = 1.0 - r2;
    const double s1 = x1 * x1 + x2 * x2;
    const double s12 = x1 * x2;
    const double dlog_det = rho / omr2;
    const double num =
        (2.0 * rho * s1 - 2.0 * s12) * omr2
        + 2.0 * rho * (r2 * s1 - 2.0 * rho * s12);
    const double dquad = num / (omr2 * omr2);
    return dlog_det - 0.5 * dquad;
}

void gaussian_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = clip_pseudo_observation(u1);
    const double v2 = clip_pseudo_observation(u2);
    const double th = std::tanh(x / 4.0);
    const double rho = 0.9999 * th;
    const double x1 = normal_quantile(v1);
    const double x2 = normal_quantile(v2);
    const double r2 = rho * rho;
    const double omr2 = 1.0 - r2;
    const double s1 = x1 * x1 + x2 * x2;
    const double s12 = x1 * x2;

    const double log_pdf =
        -0.5 * std::log(omr2)
        - 0.5 * (r2 * s1 - 2.0 * rho * s12) / omr2;
    pdf = std::exp(log_pdf);

    const double dlog_det = rho / omr2;
    const double num =
        (2.0 * rho * s1 - 2.0 * s12) * omr2
        + 2.0 * rho * (r2 * s1 - 2.0 * rho * s12);
    const double dquad = num / (omr2 * omr2);
    const double dlog_dr = dlog_det - 0.5 * dquad;
    const double dr_dx = 0.9999 * 0.25 * (1.0 - th * th);
    d_pdf_dx = pdf * dlog_dr * dr_dx;
}

void gaussian_pair_pdf_and_gradient(
    double u1,
    double u2,
    double value,
    double,
    double,
    double& pdf,
    double& d_pdf_dx) {

    gaussian_pdf_and_grad_x_unrotated(
        u1, u2, value, pdf, d_pdf_dx);
}

void gaussian_fill_grid_row_from_stats_impl(
    double sum_squares,
    double cross_product,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    for (std::size_t j = 0; j < parameter_grid.size(); ++j) {
        const double parameter = parameter_grid[j];
        const double parameter2 = parameter * parameter;
        const double one_minus_parameter2 = 1.0 - parameter2;
        const double log_pdf =
            -0.5 * std::log(one_minus_parameter2)
            - 0.5
                * (parameter2 * sum_squares
                   - 2.0 * parameter * cross_product)
                / one_minus_parameter2;
        const double pdf = std::exp(log_pdf);
        pdf_row[j] = pdf;
        if (gradient_row != nullptr) {
            const double dlog_det =
                parameter / one_minus_parameter2;
            const double numerator =
                (2.0 * parameter * sum_squares
                 - 2.0 * cross_product)
                    * one_minus_parameter2
                + 2.0 * parameter
                    * (parameter2 * sum_squares
                       - 2.0 * parameter * cross_product);
            const double dquadratic = numerator
                / (one_minus_parameter2 * one_minus_parameter2);
            const double dlog_dparameter =
                dlog_det - 0.5 * dquadratic;
            gradient_row[j] =
                pdf * dlog_dparameter * derivative_grid[j];
        }
    }
}

void gaussian_fill_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    const double v1 = clip_pseudo_observation(u1);
    const double v2 = clip_pseudo_observation(u2);
    const double x1 = normal_quantile(v1);
    const double x2 = normal_quantile(v2);
    gaussian_fill_grid_row_from_stats_impl(
        x1 * x1 + x2 * x2,
        x1 * x2,
        parameter_grid,
        derivative_grid,
        pdf_row,
        gradient_row);
}

void gaussian_fill_density_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    double* pdf_row) {

    static const std::vector<double> no_derivatives;
    gaussian_fill_grid_row(
        u1, u2, parameter_grid, no_derivatives, pdf_row, nullptr);
}

void gaussian_fill_density_gradient_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    for (std::size_t index = 0; index < parameter_grid.size(); ++index) {
        const double parameter = parameter_grid[index];
        const double density = std::exp(
            gaussian_log_pdf_unrotated(u1, u2, parameter));
        pdf_row[index] = density;
        if (gradient_row != nullptr) {
            gradient_row[index] = density
                * gaussian_dlog_pdf_dr_unrotated(u1, u2, parameter)
                * derivative_grid[index];
        }
    }
}

double gaussian_h_from_quantiles(double z_u, double z_v, double rho);

double gaussian_h_unrotated(double u, double v, double rho) {
    const double u_clipped = clip_pseudo_observation(u);
    const double v_clipped = clip_pseudo_observation(v);
    return gaussian_h_from_quantiles(
        normal_quantile(u_clipped), normal_quantile(v_clipped), rho);
}

double gaussian_h_from_quantiles(double z_u, double z_v, double rho) {
    const double z = (z_u - rho * z_v) / std::sqrt(1.0 - rho * rho);
    return norm_cdf(z);
}

void gaussian_h_pair_from_quantiles(
    double z_first,
    double z_second,
    double rho,
    double& first_next,
    double& second_next) {
    const double scale = std::sqrt(1.0 - rho * rho);
    first_next = norm_cdf((z_first - rho * z_second) / scale);
    second_next = norm_cdf((z_second - rho * z_first) / scale);
}

double gaussian_h_inverse_unrotated(double q, double given, double rho) {
    const double q_clipped = clip_pseudo_observation(q);
    const double given_clipped = clip_pseudo_observation(given);
    const double rho_clipped = std::min(std::max(rho, -0.999999), 0.999999);
    const double z =
        normal_quantile(q_clipped) * std::sqrt(1.0 - rho_clipped * rho_clipped)
        + rho_clipped * normal_quantile(given_clipped);
    return norm_cdf(z);
}

}  // namespace scar_internal

namespace scar::copula::pair {

const PairKernelFunctions& gaussian_kernel() noexcept {
    static const PairKernelFunctions functions = {
        scar_internal::gaussian_tau_to_parameter,
        scar_internal::gaussian_parameter_to_tau,
        scar_internal::gaussian_log_pdf_unrotated,
        scar_internal::gaussian_dlog_pdf_dr_unrotated,
        scar_internal::gaussian_pair_pdf_and_gradient,
        scar_internal::gaussian_h_unrotated,
        scar_internal::gaussian_h_inverse_unrotated,
        scar_internal::gaussian_fill_density_grid_row,
        scar_internal::gaussian_fill_density_gradient_grid_row,
    };
    return functions;
}

double gaussian_h_from_quantiles(double z_u, double z_v, double rho) {
    return scar_internal::gaussian_h_from_quantiles(z_u, z_v, rho);
}

void gaussian_h_pair_from_quantiles(
    double z_first,
    double z_second,
    double rho,
    double& first_next,
    double& second_next) {

    scar_internal::gaussian_h_pair_from_quantiles(
        z_first,
        z_second,
        rho,
        first_next,
        second_next);
}

void gaussian_fill_grid_row_from_stats(
    double sum_squares,
    double cross_product,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    scar_internal::gaussian_fill_grid_row_from_stats_impl(
        sum_squares,
        cross_product,
        parameter_grid,
        derivative_grid,
        pdf_row,
        gradient_row);
}

}  // namespace scar::copula::pair
