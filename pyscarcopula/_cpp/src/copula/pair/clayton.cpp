#include "scar/copula/pair/clayton.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {

double clayton_tau_to_parameter(double tau) {
    if (!std::isfinite(tau) || tau <= 0.0 || tau >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 2.0 * tau / (1.0 - tau);
}

double clayton_parameter_to_tau(double parameter) {
    return std::isfinite(parameter) && parameter > 0.0
        ? parameter / (parameter + 2.0)
        : std::numeric_limits<double>::quiet_NaN();
}

double clayton_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);

    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double a = -r * log_v1;
    const double b = -r * log_v2;
    const double log_max = std::max(a, b);
    const double log_min = std::min(a, b);
    const double correction = std::exp(log_min - log_max) - std::exp(-log_max);
    const double log_s = log_max + std::log1p(correction);

    return std::log1p(r)
        + (-r - 1.0) * log_v1
        + (-r - 1.0) * log_v2
        + (-2.0 - 1.0 / r) * log_s;
}

double clayton_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);

    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double a = -r * log_v1;
    const double b = -r * log_v2;
    const double log_max = std::max(a, b);
    const double log_min = std::min(a, b);
    const double correction = std::exp(log_min - log_max) - std::exp(-log_max);
    const double log_s = log_max + std::log1p(correction);

    const double log_abs_logv1 = std::log(-log_v1);
    const double log_abs_logv2 = std::log(-log_v2);
    const double p = log_abs_logv1 + a;
    const double q = log_abs_logv2 + b;
    const double pq_max = std::max(p, q);
    const double pq_min = std::min(p, q);
    const double log_ds = pq_max + std::log1p(std::exp(pq_min - pq_max));
    const double ds_over_s = std::exp(log_ds - log_s);

    return 1.0 / (1.0 + r)
        - log_v1
        - log_v2
        + log_s / (r * r)
        + (-2.0 - 1.0 / r) * ds_over_s;
}

void clayton_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double r,
    double d_r_dx,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double a = -r * log_v1;
    const double b = -r * log_v2;
    const double log_max = std::max(a, b);
    const double log_min = std::min(a, b);
    const double correction = std::exp(log_min - log_max) - std::exp(-log_max);
    const double log_s = log_max + std::log1p(correction);

    const double log_pdf =
        std::log1p(r)
        + (-r - 1.0) * log_v1
        + (-r - 1.0) * log_v2
        + (-2.0 - 1.0 / r) * log_s;
    pdf = std::exp(log_pdf);

    const double log_abs_logv1 = std::log(-log_v1);
    const double log_abs_logv2 = std::log(-log_v2);
    const double p = log_abs_logv1 + a;
    const double q = log_abs_logv2 + b;
    const double pq_max = std::max(p, q);
    const double pq_min = std::min(p, q);
    const double log_ds = pq_max + std::log1p(std::exp(pq_min - pq_max));
    const double ds_over_s = std::exp(log_ds - log_s);
    const double dlog_dr =
        1.0 / (1.0 + r)
        - log_v1
        - log_v2
        + log_s / (r * r)
        + (-2.0 - 1.0 / r) * ds_over_s;
    d_pdf_dx = pdf * dlog_dr * d_r_dx;
}

void clayton_pair_pdf_and_gradient(
    double u1,
    double u2,
    double,
    double parameter,
    double d_parameter_dx,
    double& pdf,
    double& d_pdf_dx) {

    clayton_pdf_and_grad_x_unrotated(
        u1,
        u2,
        parameter,
        d_parameter_dx,
        pdf,
        d_pdf_dx);
}

void clayton_fill_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_abs_logv1 = std::log(-log_v1);
    const double log_abs_logv2 = std::log(-log_v2);

    for (std::size_t j = 0; j < parameter_grid.size(); ++j) {
        const double parameter = parameter_grid[j];
        const double a = -parameter * log_v1;
        const double b = -parameter * log_v2;
        const double log_max = std::max(a, b);
        const double log_min = std::min(a, b);
        const double correction =
            std::exp(log_min - log_max) - std::exp(-log_max);
        const double log_s = log_max + std::log1p(correction);
        const double log_pdf =
            std::log1p(parameter)
            + (-parameter - 1.0) * log_v1
            + (-parameter - 1.0) * log_v2
            + (-2.0 - 1.0 / parameter) * log_s;
        const double pdf = std::exp(log_pdf);
        pdf_row[j] = pdf;
        if (gradient_row != nullptr) {
            const double p = log_abs_logv1 + a;
            const double q = log_abs_logv2 + b;
            const double pq_max = std::max(p, q);
            const double pq_min = std::min(p, q);
            const double log_ds =
                pq_max + std::log1p(std::exp(pq_min - pq_max));
            const double ds_over_s = std::exp(log_ds - log_s);
            const double dlog_dparameter =
                1.0 / (1.0 + parameter)
                - log_v1
                - log_v2
                + log_s / (parameter * parameter)
                + (-2.0 - 1.0 / parameter) * ds_over_s;
            gradient_row[j] =
                pdf * dlog_dparameter * derivative_grid[j];
        }
    }
}

void clayton_fill_density_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    double* pdf_row) {

    static const std::vector<double> no_derivatives;
    clayton_fill_grid_row(
        u1, u2, parameter_grid, no_derivatives, pdf_row, nullptr);
}

void clayton_fill_density_gradient_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    clayton_fill_grid_row(
        u1,
        u2,
        parameter_grid,
        derivative_grid,
        pdf_row,
        gradient_row);
}

double clayton_h_unrotated(double u, double v, double r) {
    const double u_clipped = std::min(std::max(u, kHEps), 1.0 - kHEps);
    const double v_clipped = std::min(std::max(v, kHEps), 1.0 - kHEps);

    if (r < 1e-8) {
        return u_clipped;
    }

    const double log_u = std::log(u_clipped);
    const double log_v = std::log(v_clipped);
    const double a = -r * log_u;
    const double b = -r * log_v;
    const double log_max = std::max(a, b);
    const double log_min = std::min(a, b);
    const double correction = std::exp(log_min - log_max) - std::exp(-log_max);
    if (correction <= -1.0) {
        return u_clipped;
    }

    const double log_s = log_max + std::log1p(correction);
    const double log_h = (-r - 1.0) * log_v + (-1.0 - 1.0 / r) * log_s;
    const double log_eps = std::log(kHEps);
    const double log_one_minus_eps = std::log(1.0 - kHEps);
    if (log_h <= log_eps) {
        return kHEps;
    }
    if (log_h >= log_one_minus_eps) {
        return 1.0 - kHEps;
    }
    return std::exp(log_h);
}

void clayton_h_pair_unrotated(
    double u,
    double v,
    double r,
    double& h_uv,
    double& h_vu) {

    const double u_clipped = std::min(std::max(u, kHEps), 1.0 - kHEps);
    const double v_clipped = std::min(std::max(v, kHEps), 1.0 - kHEps);

    if (r < 1e-8) {
        h_uv = u_clipped;
        h_vu = v_clipped;
        return;
    }

    const double log_u = std::log(u_clipped);
    const double log_v = std::log(v_clipped);
    const double a = -r * log_u;
    const double b = -r * log_v;
    const double log_max = std::max(a, b);
    const double log_min = std::min(a, b);
    const double correction = std::exp(log_min - log_max) - std::exp(-log_max);
    if (correction <= -1.0) {
        h_uv = u_clipped;
        h_vu = v_clipped;
        return;
    }

    const double log_s = log_max + std::log1p(correction);
    const double common = (-1.0 - 1.0 / r) * log_s;
    const double log_eps = std::log(kHEps);
    const double log_one_minus_eps = std::log(1.0 - kHEps);

    const double log_h_uv = (-r - 1.0) * log_v + common;
    if (log_h_uv <= log_eps) {
        h_uv = kHEps;
    } else if (log_h_uv >= log_one_minus_eps) {
        h_uv = 1.0 - kHEps;
    } else {
        h_uv = std::exp(log_h_uv);
    }

    const double log_h_vu = (-r - 1.0) * log_u + common;
    if (log_h_vu <= log_eps) {
        h_vu = kHEps;
    } else if (log_h_vu >= log_one_minus_eps) {
        h_vu = 1.0 - kHEps;
    } else {
        h_vu = std::exp(log_h_vu);
    }
}

double clayton_h_inverse_unrotated(double q, double given, double r) {
    const double q_clipped = std::min(std::max(q, kHEps), 1.0 - kHEps);
    const double given_clipped = std::min(std::max(given, kHEps), 1.0 - kHEps);

    if (r < 1e-8) {
        return q_clipped;
    }

    const double a = q_clipped * std::pow(given_clipped, r + 1.0);
    if (a < kPdfEps) {
        return kHEps;
    }
    const double base =
        std::pow(a, -r / (1.0 + r)) + 1.0 - std::pow(given_clipped, -r);
    if (base < kPdfEps) {
        return kHEps;
    }
    const double value = std::pow(base, -1.0 / r);
    return std::min(std::max(value, kHEps), 1.0 - kHEps);
}

double clayton_psi(double t, double r) {
    return std::pow(1.0 + t * r, -1.0 / r);
}

}  // namespace scar_internal

namespace scar::copula::pair {

const PairKernelFunctions& clayton_kernel() noexcept {
    static const PairKernelFunctions functions = {
        scar_internal::clayton_tau_to_parameter,
        scar_internal::clayton_parameter_to_tau,
        scar_internal::clayton_log_pdf_unrotated,
        scar_internal::clayton_dlog_pdf_dr_unrotated,
        scar_internal::clayton_pair_pdf_and_gradient,
        scar_internal::clayton_h_unrotated,
        scar_internal::clayton_h_inverse_unrotated,
        scar_internal::clayton_fill_density_grid_row,
        scar_internal::clayton_fill_density_gradient_grid_row,
    };
    return functions;
}

}  // namespace scar::copula::pair
