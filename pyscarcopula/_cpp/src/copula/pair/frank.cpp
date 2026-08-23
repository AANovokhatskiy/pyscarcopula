#include "scar/copula/pair/frank.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {

namespace {

double frank_tau(double parameter) {
    if (!std::isfinite(parameter) || parameter <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (parameter < 1e-4) {
        const double parameter2 = parameter * parameter;
        return parameter / 9.0
            - parameter * parameter2 / 900.0
            + parameter * parameter2 * parameter2 / 52920.0;
    }

    const long double value = static_cast<long double>(parameter);
    long double integral = 0.0L;
    if (parameter <= 0.25) {
        const long double value2 = value * value;
        integral =
            value
            - value2 / 4.0L
            + value * value2 / 36.0L
            - value * value2 * value2 / 3600.0L
            + value * value2 * value2 * value2 / 211680.0L
            - value * std::pow(value2, 4) / 10886400.0L
            + value * std::pow(value2, 5) / 526901760.0L
            - 691.0L * value * std::pow(value2, 6)
                / 16999766784000.0L
            + value * std::pow(value2, 7) / 490497638400.0L
            - 3617.0L * value * std::pow(value2, 8)
                / 35568742809600000.0L;
    } else {
        integral = static_cast<long double>(kPi) * kPi / 6.0L;
        for (int k = 1; k < 100000; ++k) {
            const long double kd = static_cast<long double>(k);
            const long double term =
                std::exp(-kd * value)
                * (value / kd + 1.0L / (kd * kd));
            integral -= term;
            if (term < 1e-19L) {
                break;
            }
        }
    }
    const long double tau =
        1.0L - 4.0L / value + 4.0L * integral / (value * value);
    return static_cast<double>(tau);
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

}  // namespace

double frank_tau_to_parameter(double tau) {
    if (!std::isfinite(tau) || tau <= 0.0 || tau >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return invert_positive_tau(
        tau,
        0.0,
        std::max(1.0, 4.0 / (1.0 - tau)),
        frank_tau);
}

double frank_parameter_to_tau(double parameter) {
    return frank_tau(parameter);
}

double frank_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);

    const double a = r * v1;
    const double b = r * v2;
    const double log_num = std::log(r) + log1mexp(r) - a - b;
    const double log_t1 = -a + log1mexp(b);
    const double log_t2 = -b + log1mexp(r - b);
    return log_num - 2.0 * logsumexp(log_t1, log_t2);
}

double frank_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    if (std::abs(r) < 1e-10) {
        return 0.0;
    }

    const double emr = std::exp(-r);
    const double emrv1 = std::exp(-r * v1);
    const double emrv2 = std::exp(-r * v2);
    const double emr1v2 = std::exp(-r * (1.0 - v2));
    const double A = emrv1 * (1.0 - emrv2);
    const double B = emrv2 * (1.0 - emr1v2);
    const double D = std::max(A + B, kPdfEps);
    const double dA = emrv1 * (-v1 * (1.0 - emrv2) + v2 * emrv2);
    const double dB = emrv2 * (-v2 * (1.0 - emr1v2) + (1.0 - v2) * emr1v2);

    return 1.0 / r
        + emr / (1.0 - emr)
        - (v1 + v2)
        - 2.0 * (dA + dB) / D;
}

void frank_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double r,
    double d_r_dx,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double a = r * v1;
    const double b = r * v2;
    const double log_num = std::log(r) + log1mexp(r) - a - b;
    const double log_t1 = -a + log1mexp(b);
    const double log_t2 = -b + log1mexp(r - b);
    pdf = std::exp(log_num - 2.0 * logsumexp(log_t1, log_t2));

    const double emr = std::exp(-r);
    const double emrv1 = std::exp(-r * v1);
    const double emrv2 = std::exp(-r * v2);
    const double emr1v2 = std::exp(-r * (1.0 - v2));
    const double A = emrv1 * (1.0 - emrv2);
    const double B = emrv2 * (1.0 - emr1v2);
    const double D = std::max(A + B, kPdfEps);
    const double dA = emrv1 * (-v1 * (1.0 - emrv2) + v2 * emrv2);
    const double dB = emrv2 * (-v2 * (1.0 - emr1v2) + (1.0 - v2) * emr1v2);
    const double dlog_dr =
        1.0 / r
        + emr / (1.0 - emr)
        - (v1 + v2)
        - 2.0 * (dA + dB) / D;
    d_pdf_dx = pdf * dlog_dr * d_r_dx;
}

void frank_pair_pdf_and_gradient(
    double u1,
    double u2,
    double,
    double parameter,
    double d_parameter_dx,
    double& pdf,
    double& d_pdf_dx) {

    frank_pdf_and_grad_x_unrotated(
        u1,
        u2,
        parameter,
        d_parameter_dx,
        pdf,
        d_pdf_dx);
}

void frank_fill_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    for (std::size_t j = 0; j < parameter_grid.size(); ++j) {
        const double parameter = parameter_grid[j];
        const double a = parameter * v1;
        const double b = parameter * v2;
        const double log_num =
            std::log(parameter) + log1mexp(parameter) - a - b;
        const double log_t1 = -a + log1mexp(b);
        const double log_t2 = -b + log1mexp(parameter - b);
        const double pdf =
            std::exp(log_num - 2.0 * logsumexp(log_t1, log_t2));
        pdf_row[j] = pdf;
        if (gradient_row != nullptr) {
            const double emr = std::exp(-parameter);
            const double emrv1 = std::exp(-parameter * v1);
            const double emrv2 = std::exp(-parameter * v2);
            const double emr1v2 =
                std::exp(-parameter * (1.0 - v2));
            const double A = emrv1 * (1.0 - emrv2);
            const double B = emrv2 * (1.0 - emr1v2);
            const double D = std::max(A + B, kPdfEps);
            const double dA = emrv1
                * (-v1 * (1.0 - emrv2) + v2 * emrv2);
            const double dB = emrv2
                * (-v2 * (1.0 - emr1v2) + (1.0 - v2) * emr1v2);
            const double dlog_dparameter =
                1.0 / parameter
                + emr / (1.0 - emr)
                - (v1 + v2)
                - 2.0 * (dA + dB) / D;
            gradient_row[j] =
                pdf * dlog_dparameter * derivative_grid[j];
        }
    }
}

void frank_fill_density_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    double* pdf_row) {

    static const std::vector<double> no_derivatives;
    frank_fill_grid_row(
        u1, u2, parameter_grid, no_derivatives, pdf_row, nullptr);
}

void frank_fill_density_gradient_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    frank_fill_grid_row(
        u1,
        u2,
        parameter_grid,
        derivative_grid,
        pdf_row,
        gradient_row);
}

double frank_h_unrotated(double u, double v, double r) {
    const double u_clipped =
        std::min(std::max(u, kPseudoObsEps), 1.0 - kPseudoObsEps);
    const double v_clipped =
        std::min(std::max(v, kPseudoObsEps), 1.0 - kPseudoObsEps);
    if (std::abs(r) < 1e-8) {
        return u_clipped;
    }

    const double ru = r * u_clipped;
    const double rv = r * v_clipped;
    const double log_numer = -rv + log1mexp(ru);
    const double log_A = -ru + log1mexp(rv);
    const double log_B = -rv + log1mexp(r * (1.0 - v_clipped));
    const double log_h = log_numer - logsumexp(log_A, log_B);

    if (log_h < -700.0) {
        return kPseudoObsEps;
    }
    if (log_h > -kPseudoObsEps) {
        return 1.0 - kPseudoObsEps;
    }
    return std::exp(log_h);
}

double frank_h_inverse_unrotated(double q, double given, double r) {
    const double q_clipped =
        std::min(std::max(q, kPseudoObsEps), 1.0 - kPseudoObsEps);
    const double given_clipped =
        std::min(std::max(given, kPseudoObsEps), 1.0 - kPseudoObsEps);
    if (std::abs(r) < 1e-8) {
        return q_clipped;
    }

    const double x3 = std::exp(-r);
    const double log_Q =
        std::log1p(-q_clipped) - std::log(q_clipped) - r * given_clipped;

    double t = q_clipped;
    if (log_Q > 50.0) {
        const double one_minus_arg =
            (1.0 - x3) / (std::exp(log_Q) + 1.0);
        if (one_minus_arg <= 0.0) {
            t = kPseudoObsEps;
        } else {
            t = -std::log1p(-one_minus_arg) / r;
        }
    } else if (log_Q < -745.0) {
        t = 1.0;
    } else {
        const double Q = std::exp(log_Q);
        const double denom = Q + 1.0;
        if (denom <= 0.0 || !std::isfinite(denom)) {
            return q_clipped;
        }
        const double arg = (Q + x3) / denom;
        if (arg <= 0.0) {
            t = 1.0;
        } else if (arg >= 1.0 - kPseudoObsEps) {
            const double one_minus_arg = (1.0 - x3) / denom;
            if (one_minus_arg <= 0.0) {
                t = kPseudoObsEps;
            } else {
                t = -std::log1p(-one_minus_arg) / r;
            }
        } else {
            t = -std::log(arg) / r;
        }
    }
    return std::min(
        std::max(t, kPseudoObsEps), 1.0 - kPseudoObsEps);
}

}  // namespace scar_internal

namespace scar::copula::pair {

const PairKernelFunctions& frank_kernel() noexcept {
    static const PairKernelFunctions functions = {
        scar_internal::frank_tau_to_parameter,
        scar_internal::frank_parameter_to_tau,
        scar_internal::frank_log_pdf_unrotated,
        scar_internal::frank_dlog_pdf_dr_unrotated,
        scar_internal::frank_pair_pdf_and_gradient,
        scar_internal::frank_h_unrotated,
        scar_internal::frank_h_inverse_unrotated,
        scar_internal::frank_fill_density_grid_row,
        scar_internal::frank_fill_density_gradient_grid_row,
    };
    return functions;
}

}  // namespace scar::copula::pair
