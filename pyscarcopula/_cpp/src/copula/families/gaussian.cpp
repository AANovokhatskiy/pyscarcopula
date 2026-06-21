#include "scar/detail/copula.hpp"

#include <algorithm>
#include <cmath>

namespace scar_internal {

namespace {

double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

}  // namespace

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

double gaussian_h_unrotated(double u, double v, double rho) {
    const double u_clipped = clip_pseudo_observation(u);
    const double v_clipped = clip_pseudo_observation(v);
    const double z = (normal_quantile(u_clipped) - rho * normal_quantile(v_clipped))
        / std::sqrt(1.0 - rho * rho);
    return norm_cdf(z);
}

double gaussian_h_rotated(double u, double v, double rho, int rotation) {
    if (rotation == 0) {
        return gaussian_h_unrotated(u, v, rho);
    }
    return std::numeric_limits<double>::quiet_NaN();
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

double gaussian_h_inverse_rotated(double q, double given, double rho, int rotation) {
    if (rotation == 0) {
        return gaussian_h_inverse_unrotated(q, given, rho);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace scar_internal
