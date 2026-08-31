#include "scar/copula/pair/joe.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace scar_internal {

namespace {

double digamma_positive(double x) {
    double result = 0.0;
    while (x < 12.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x)
        - 0.5 * inv
        - inv2 * (
            1.0 / 12.0
            - inv2 * (
                1.0 / 120.0
                - inv2 * (
                    1.0 / 252.0
                    - inv2 * (
                        1.0 / 240.0
                        - inv2 * 5.0 / 660.0))));
    return result;
}

double digamma_quotient_near_one(double epsilon) {
    constexpr std::array<double, 13> coefficients = {
        1.6449340668482264365,
        -1.2020569031595942854,
        1.0823232337111381915,
        -1.0369277551433699263,
        1.0173430619844491397,
        -1.0083492773819228268,
        1.0040773561979443394,
        -1.0020083928260822144,
        1.0009945751278180853,
        -1.0004941886041194646,
        1.0002460865533080483,
        -1.0001227133475784891,
        1.0000612481350587048,
    };
    double value = coefficients.back();
    for (std::size_t index = coefficients.size() - 1; index > 0; --index) {
        value = value * epsilon + coefficients[index - 1];
    }
    return value;
}

double joe_tau(double parameter) {
    if (!std::isfinite(parameter) || parameter < 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (parameter == 1.0) {
        return 0.0;
    }
    const double euler_gamma =
        0.577215664901532860606512090082402431;
    const double a = 2.0 / parameter;
    double f_value = 0.0;
    const double epsilon = a - 1.0;
    if (std::abs(epsilon) <= 0.01) {
        f_value = digamma_quotient_near_one(epsilon)
            - (digamma_positive(a + 1.0) + euler_gamma) / a;
    } else {
        f_value =
            (digamma_positive(a) + euler_gamma) / epsilon
            - (digamma_positive(a + 1.0) + euler_gamma) / a;
    }
    return std::clamp(
        1.0 - 4.0 * f_value / (parameter * parameter),
        0.0,
        1.0 - 1e-15);
}

template <typename Function>
double invert_positive_tau(
    double tau,
    double lower,
    double upper,
    Function function) {

    while (function(upper) <= tau && upper <= 1e12) {
        upper *= 2.0;
    }
    if (upper > 1e12 || !std::isfinite(function(upper))) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    for (int iteration = 0; iteration < 120; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        if (function(midpoint) < tau) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    return 0.5 * (lower + upper);
}

double joe_log_B(double log_q0, double log_q1) {
    double q0 = 0.0;
    if (log_q0 > -745.0) {
        q0 = std::exp(log_q0);
    }
    const double q0_clipped = std::min(std::max(q0, 0.0), 1.0);
    const double log_one_minus_q0 = std::log1p(-q0_clipped);
    return logsumexp(log_q0, log_one_minus_q0 + log_q1);
}

}  // namespace

double joe_tau_to_parameter(double tau) {
    if (!std::isfinite(tau) || tau <= 0.0 || tau >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return invert_positive_tau(
        tau,
        1.0,
        std::max(2.0, 2.0 / (1.0 - tau)),
        joe_tau);
}

double joe_parameter_to_tau(double parameter) {
    return joe_tau(parameter);
}

double joe_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::min(std::max(1.0 - v1, kPdfEps), 1.0 - kPdfEps);
    const double q2 = std::min(std::max(1.0 - v2, kPdfEps), 1.0 - kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);
    const double log_t1 = r * log_q1 + log1mexp(-r * log_q2);
    const double log_t2 = r * log_q2;
    const double log_B = logsumexp(log_t1, log_t2);
    const double B = std::exp(log_B);

    double log_rp = 0.0;
    if (B > r - 1.0) {
        log_rp = log_B + std::log1p((r - 1.0) / B);
    } else {
        log_rp = std::log(r - 1.0) + std::log1p(B / (r - 1.0));
    }

    return (r - 1.0) * (log_q1 + log_q2)
        + log_rp
        - (2.0 - 1.0 / r) * log_B;
}

double joe_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::max(1.0 - v1, kPdfEps);
    const double q2 = std::max(1.0 - v2, kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);
    const double q1r = std::pow(q1, r);
    const double q2r = std::pow(q2, r);
    const double B = std::max(q1r + q2r - q1r * q2r, kPdfEps);
    const double dB =
        q1r * log_q1 * (1.0 - q2r)
        + q2r * log_q2 * (1.0 - q1r);

    return log_q1 + log_q2
        + (1.0 + dB) / (r - 1.0 + B)
        - std::log(B) / (r * r)
        - (2.0 - 1.0 / r) * dB / B;
}

void joe_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double r,
    double d_r_dx,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::min(std::max(1.0 - v1, kPdfEps), 1.0 - kPdfEps);
    const double q2 = std::min(std::max(1.0 - v2, kPdfEps), 1.0 - kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);
    const double log_t1 = r * log_q1 + log1mexp(-r * log_q2);
    const double log_t2 = r * log_q2;
    const double log_B = logsumexp(log_t1, log_t2);
    const double B_for_log = std::exp(log_B);

    double log_rp = 0.0;
    if (B_for_log > r - 1.0) {
        log_rp = log_B + std::log1p((r - 1.0) / B_for_log);
    } else {
        log_rp = std::log(r - 1.0) + std::log1p(B_for_log / (r - 1.0));
    }

    const double log_pdf =
        (r - 1.0) * (log_q1 + log_q2)
        + log_rp
        - (2.0 - 1.0 / r) * log_B;
    pdf = std::exp(log_pdf);

    const double q1r = std::exp(r * log_q1);
    const double q2r = std::exp(r * log_q2);
    const double B = std::max(q1r + q2r - q1r * q2r, kPdfEps);
    const double dB =
        q1r * log_q1 * (1.0 - q2r)
        + q2r * log_q2 * (1.0 - q1r);
    const double dlog_dr =
        log_q1 + log_q2
        + (1.0 + dB) / (r - 1.0 + B)
        - std::log(B) / (r * r)
        - (2.0 - 1.0 / r) * dB / B;
    d_pdf_dx = pdf * dlog_dr * d_r_dx;
}

void joe_pair_pdf_and_gradient(
    double u1,
    double u2,
    double,
    double parameter,
    double d_parameter_dx,
    double& pdf,
    double& d_pdf_dx) {

    joe_pdf_and_grad_x_unrotated(
        u1,
        u2,
        parameter,
        d_parameter_dx,
        pdf,
        d_pdf_dx);
}

void joe_fill_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 =
        std::min(std::max(1.0 - v1, kPdfEps), 1.0 - kPdfEps);
    const double q2 =
        std::min(std::max(1.0 - v2, kPdfEps), 1.0 - kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);

    for (std::size_t j = 0; j < parameter_grid.size(); ++j) {
        const double parameter = parameter_grid[j];
        const double log_t1 =
            parameter * log_q1 + log1mexp(-parameter * log_q2);
        const double log_t2 = parameter * log_q2;
        const double log_B = logsumexp(log_t1, log_t2);
        const double B_for_log = std::exp(log_B);
        double log_rp = 0.0;
        if (B_for_log > parameter - 1.0) {
            log_rp = log_B
                + std::log1p((parameter - 1.0) / B_for_log);
        } else {
            log_rp = std::log(parameter - 1.0)
                + std::log1p(B_for_log / (parameter - 1.0));
        }
        const double log_pdf =
            (parameter - 1.0) * (log_q1 + log_q2)
            + log_rp
            - (2.0 - 1.0 / parameter) * log_B;
        const double pdf = std::exp(log_pdf);
        pdf_row[j] = pdf;
        if (gradient_row != nullptr) {
            const double q1r = std::pow(q1, parameter);
            const double q2r = std::pow(q2, parameter);
            const double B =
                std::max(q1r + q2r - q1r * q2r, kPdfEps);
            const double dB =
                q1r * log_q1 * (1.0 - q2r)
                + q2r * log_q2 * (1.0 - q1r);
            const double dlog_dparameter =
                log_q1 + log_q2
                + (1.0 + dB) / (parameter - 1.0 + B)
                - std::log(B) / (parameter * parameter)
                - (2.0 - 1.0 / parameter) * dB / B;
            gradient_row[j] =
                pdf * dlog_dparameter * derivative_grid[j];
        }
    }
}

void joe_fill_density_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    double* pdf_row) {

    static const std::vector<double> no_derivatives;
    joe_fill_grid_row(
        u1, u2, parameter_grid, no_derivatives, pdf_row, nullptr);
}

void joe_fill_density_gradient_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    joe_fill_grid_row(
        u1,
        u2,
        parameter_grid,
        derivative_grid,
        pdf_row,
        gradient_row);
}

double joe_h_unrotated(double u, double v, double r) {
    const double u_clipped = std::min(std::max(u, kHEps), 1.0 - kHEps);
    const double v_clipped = std::min(std::max(v, kHEps), 1.0 - kHEps);
    if (r < 1.0 + 1e-8) {
        return u_clipped;
    }

    const double log_1mu = std::log(1.0 - u_clipped);
    const double log_1mv = std::log(1.0 - v_clipped);
    const double log_qu = r * log_1mu;
    const double log_qv = r * log_1mv;
    const double log_B = joe_log_B(log_qu, log_qv);
    if (!std::isfinite(log_B)) {
        return u_clipped;
    }

    const double log_h =
        log1mexp(-log_qu)
        + (1.0 / r - 1.0) * log_B
        + log_qv
        - log_1mv;
    const double log_eps = std::log(kHEps);
    const double log_one_minus_eps = std::log(1.0 - kHEps);
    if (log_h <= log_eps) {
        return kHEps;
    }
    if (log_h >= log_one_minus_eps) {
        return 1.0 - kHEps;
    }
    if (!std::isfinite(log_h)) {
        return u_clipped;
    }
    return std::exp(log_h);
}

double joe_h_inverse_with_options(
    double q, double given, double r,
    const scar::HInverseOptions& options) {
    const double q_clipped = std::min(std::max(q, kHEps), 1.0 - kHEps);
    const double given_clipped = std::min(std::max(given, kHEps), 1.0 - kHEps);
    if (r < 1.0 + 1e-8) {
        return q_clipped;
    }

    double lo = kHEps;
    double hi = 1.0 - kHEps;
    double t = q_clipped;
    for (int j = 0; j < options.max_iterations; ++j) {
        t = std::min(std::max(t, lo), hi);
        const double h_val = joe_h_unrotated(t, given_clipped, r);
        const double err = h_val - q_clipped;
        if (std::abs(err) < options.tolerance) {
            break;
        }
        if (err > 0.0) {
            hi = t;
        } else {
            lo = t;
        }

        const double dt = std::max(t * 1e-7, 1e-12);
        const double t_p = std::min(t + dt, 1.0 - kHEps);
        const double dh_dt = (joe_h_unrotated(t_p, given_clipped, r) - h_val)
            / std::max(t_p - t, 1e-300);
        const double newton = t - err / dh_dt;
        if (std::isfinite(newton) && std::abs(dh_dt) > 1e-300
            && newton > lo && newton < hi) {
            t = newton;
        } else {
            t = 0.5 * (lo + hi);
        }
    }
    return std::min(std::max(t, kHEps), 1.0 - kHEps);
}

double joe_h_inverse_unrotated(double q, double given, double r) {
    return joe_h_inverse_with_options(q, given, r, {1e-10, 50});
}

}  // namespace scar_internal

namespace scar::copula::pair {

const PairKernelFunctions& joe_kernel() noexcept {
    static const PairKernelFunctions functions = {
        scar_internal::joe_tau_to_parameter,
        scar_internal::joe_parameter_to_tau,
        scar_internal::joe_log_pdf_unrotated,
        scar_internal::joe_dlog_pdf_dr_unrotated,
        scar_internal::joe_pair_pdf_and_gradient,
        scar_internal::joe_h_unrotated,
        scar_internal::joe_h_inverse_unrotated,
        scar_internal::joe_fill_density_grid_row,
        scar_internal::joe_fill_density_gradient_grid_row,
        scar_internal::joe_h_inverse_with_options,
    };
    return functions;
}

}  // namespace scar::copula::pair
