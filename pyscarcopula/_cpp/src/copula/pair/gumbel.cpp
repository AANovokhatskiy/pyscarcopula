#include "scar/copula/pair/gumbel.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace scar_internal {

double gumbel_tau_to_parameter(double tau) {
    return std::isfinite(tau) && tau > 0.0 && tau < 1.0
        ? 1.0 / (1.0 - tau)
        : std::numeric_limits<double>::quiet_NaN();
}

double gumbel_parameter_to_tau(double parameter) {
    return std::isfinite(parameter) && parameter >= 1.0
        ? 1.0 - 1.0 / parameter
        : std::numeric_limits<double>::quiet_NaN();
}

double gumbel_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_p1 = std::log(-log_v1);
    const double log_p2 = std::log(-log_v2);

    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);
    const double delta = r * (log_min - log_max);
    const double S = r * log_max + std::log1p(std::exp(delta));
    const double A = std::exp(S / r);

    return (r - 1.0) * (log_p1 + log_p2)
        + (1.0 / r - 2.0) * S
        + std::log(r - 1.0 + A)
        - A
        - log_v1
        - log_v2;
}

double gumbel_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_p1 = std::log(std::max(-log_v1, kPdfEps));
    const double log_p2 = std::log(std::max(-log_v2, kPdfEps));

    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);
    const double delta = r * (log_min - log_max);
    const double exp_delta = std::exp(delta);
    const double S = r * log_max + std::log1p(exp_delta);
    const double sig = exp_delta / (1.0 + exp_delta);
    const double dS_dr = log_max + (log_min - log_max) * sig;
    const double A = std::exp(S / r);
    const double dlogA_dr = (dS_dr * r - S) / (r * r);
    const double dA_dr = A * dlogA_dr;

    return (log_p1 + log_p2)
        - S / (r * r)
        + (1.0 / r - 2.0) * dS_dr
        + (1.0 + dA_dr) / (r - 1.0 + A)
        - dA_dr;
}

void gumbel_pdf_and_grad_x_unrotated(
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
    const double log_p1 = std::log(std::max(-log_v1, kPdfEps));
    const double log_p2 = std::log(std::max(-log_v2, kPdfEps));

    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);
    const double delta = r * (log_min - log_max);
    const double exp_delta = std::exp(delta);
    const double S = r * log_max + std::log1p(exp_delta);
    const double sig = exp_delta / (1.0 + exp_delta);
    const double dS_dr = log_max + (log_min - log_max) * sig;
    const double A = std::exp(S / r);

    const double log_pdf =
        (r - 1.0) * (log_p1 + log_p2)
        + (1.0 / r - 2.0) * S
        + std::log(r - 1.0 + A)
        - A
        - log_v1
        - log_v2;
    pdf = std::exp(log_pdf);

    const double dlogA_dr = (dS_dr * r - S) / (r * r);
    const double dA_dr = A * dlogA_dr;
    const double dlog_dr =
        (log_p1 + log_p2)
        - S / (r * r)
        + (1.0 / r - 2.0) * dS_dr
        + (1.0 + dA_dr) / (r - 1.0 + A)
        - dA_dr;
    d_pdf_dx = pdf * dlog_dr * d_r_dx;
}

void gumbel_pair_pdf_and_gradient(
    double u1,
    double u2,
    double,
    double parameter,
    double d_parameter_dx,
    double& pdf,
    double& d_pdf_dx) {

    gumbel_pdf_and_grad_x_unrotated(
        u1,
        u2,
        parameter,
        d_parameter_dx,
        pdf,
        d_pdf_dx);
}

void gumbel_fill_grid_row(
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
    const double log_p1 = std::log(std::max(-log_v1, kPdfEps));
    const double log_p2 = std::log(std::max(-log_v2, kPdfEps));
    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);

    for (std::size_t j = 0; j < parameter_grid.size(); ++j) {
        const double parameter = parameter_grid[j];
        const double delta = parameter * (log_min - log_max);
        const double exp_delta = std::exp(delta);
        const double S = parameter * log_max + std::log1p(exp_delta);
        const double A = std::exp(S / parameter);
        const double log_pdf =
            (parameter - 1.0) * (log_p1 + log_p2)
            + (1.0 / parameter - 2.0) * S
            + std::log(parameter - 1.0 + A)
            - A
            - log_v1
            - log_v2;
        const double pdf = std::exp(log_pdf);
        pdf_row[j] = pdf;
        if (gradient_row != nullptr) {
            const double sig = exp_delta / (1.0 + exp_delta);
            const double dS_dparameter =
                log_max + (log_min - log_max) * sig;
            const double dlogA_dparameter =
                (dS_dparameter * parameter - S)
                / (parameter * parameter);
            const double dA_dparameter = A * dlogA_dparameter;
            const double dlog_dparameter =
                (log_p1 + log_p2)
                - S / (parameter * parameter)
                + (1.0 / parameter - 2.0) * dS_dparameter
                + (1.0 + dA_dparameter) / (parameter - 1.0 + A)
                - dA_dparameter;
            gradient_row[j] =
                pdf * dlog_dparameter * derivative_grid[j];
        }
    }
}

void gumbel_fill_density_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    double* pdf_row) {

    static const std::vector<double> no_derivatives;
    gumbel_fill_grid_row(
        u1, u2, parameter_grid, no_derivatives, pdf_row, nullptr);
}

void gumbel_fill_density_gradient_grid_row(
    double u1,
    double u2,
    const std::vector<double>& parameter_grid,
    const std::vector<double>& derivative_grid,
    double* pdf_row,
    double* gradient_row) {

    gumbel_fill_grid_row(
        u1,
        u2,
        parameter_grid,
        derivative_grid,
        pdf_row,
        gradient_row);
}

namespace {

double gumbel_log_expm1(double value) {
    return value > 0.6931471805599453
        ? value + std::log1p(-std::exp(-value))
        : std::log(std::expm1(value));
}

}  // namespace

double gumbel_log_h_unrotated(
    double u, double v, double r, bool reflected = false) {
    if (!std::isfinite(r) || r < 1.0 || !std::isfinite(u)
        || !std::isfinite(v) || u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (u == 0.0 || u == 1.0 || r == 1.0) {
        return reflected ? std::log1p(-u) : std::log(u);
    }
    if (v == 0.0) {
        return 0.0;
    }
    if (v == 1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    const double x = reflected ? -std::log1p(-u) : -std::log(u);
    const double y = -std::log(v);
    const double delta = std::abs(x - y) < 0.5 * y
        ? std::log1p((x - y) / y) : std::log(x) - std::log(y);
    const double scaled = r * delta;
    if (scaled < -36.0) {
        // Keep a representable survival tail even when exp(scaled) underflows.
        return -std::exp(scaled + std::log(y / r + (r - 1.0) / r));
    }
    const double log_sum = scaled > 0.0
        ? scaled + std::log1p(std::exp(-scaled))
        : std::log1p(std::exp(scaled));
    // log_sum/r must be formed without an overflowing r*delta. The
    // remaining log term may legitimately tend to infinity (h tends to 0).
    const double log_ratio = scaled > 0.0
        ? delta + std::log1p(std::exp(-scaled)) / r : log_sum / r;
    const double log_h = -y * std::expm1(log_ratio)
        - ((r - 1.0) / r) * log_sum;
    return std::min(log_h, 0.0);
}

double gumbel_h_unrotated(double u, double v, double r) {
    return std::exp(gumbel_log_h_unrotated(u, v, r));
}

double gumbel_h_reflected(double u, double v, double r) {
    return -std::expm1(gumbel_log_h_unrotated(u, v, r, true));
}

double gumbel_inverse_log_value(
    double log_q, double given, double r,
    const scar::HInverseOptions& options) {
    const double alpha = 1.0 / r;
    const double beta = (r - 1.0) / r;
    const double y = -std::log(given);
    const double target = -log_q;
    if (target < 1e-100) {
        // x itself can be below denorm_min while the returned tail gap is
        // representable. The linearized equation has relative error O(target).
        const double log_x = std::log(target) - std::log(alpha * y + beta);
        return -y * std::exp(alpha * log_x);
    }
    // x = r*log(A/y). The equation has a nonnegative root and two
    // independent analytic upper bounds; neither requires powers y^r.
    double lo = 0.0;
    double hi = std::min(std::log1p(target / y) / alpha, target / beta);
    double x = hi;
    const double tolerance = std::min(options.tolerance, 8e-15);
    bool converged = false;
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        const double ax = alpha * x;
        const double residual = y * std::expm1(ax) + beta * x - target;
        if (std::abs(residual) <= tolerance * target) {
            converged = true;
            break;
        }
        if (residual > 0.0) {
            hi = x;
        } else {
            lo = x;
        }
        const double derivative = alpha * y * std::exp(ax) + beta;
        double candidate = x - residual / derivative;
        if (!(candidate > lo && candidate < hi) || !std::isfinite(candidate)) {
            candidate = lo + 0.5 * (hi - lo);
        }
        if (candidate == x) {
            // A collapsed bracket without the requested residual is failure.
            throw std::runtime_error(
                "Gumbel conditional inverse did not converge at floating-point precision");
        }
        x = candidate;
    }
    if (!converged) {
        throw std::runtime_error(
            "Gumbel conditional inverse did not converge within max_iterations");
    }
    return -y * std::exp(alpha * gumbel_log_expm1(x));
}

double gumbel_h_inverse_impl(
    double q, double given, double r,
    const scar::HInverseOptions& options, bool reflected) {
    const double failure = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(r) || r < 1.0 || !std::isfinite(q)
        || !std::isfinite(given) || q < 0.0 || q > 1.0
        || given < 0.0 || given > 1.0 || options.max_iterations <= 0
        || !std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
        return failure;
    }
    if (q == 0.0 || q == 1.0 || r == 1.0) {
        return q;
    }
    if (given == 0.0 || given == 1.0) {
        return reflected ? 1.0 - given : given;
    }
    const double log_value = gumbel_inverse_log_value(
        reflected ? std::log1p(-q) : std::log(q), given, r, options);
    return reflected ? -std::expm1(log_value) : std::exp(log_value);
}

double gumbel_h_inverse_with_options(
    double q, double given, double r,
    const scar::HInverseOptions& options) {
    return gumbel_h_inverse_impl(q, given, r, options, false);
}

double gumbel_h_inverse_reflected_with_options(
    double q, double given, double r,
    const scar::HInverseOptions& options) {
    return gumbel_h_inverse_impl(q, given, r, options, true);
}

double gumbel_h_inverse_unrotated(double q, double given, double r) {
    return gumbel_h_inverse_with_options(q, given, r, {8e-15, 80});
}

double gumbel_h_inverse_reflected(double q, double given, double r) {
    return gumbel_h_inverse_reflected_with_options(q, given, r, {8e-15, 80});
}

}  // namespace scar_internal

namespace scar::copula::pair {

const PairKernelFunctions& gumbel_kernel() noexcept {
    static const PairKernelFunctions functions = {
        scar_internal::gumbel_tau_to_parameter,
        scar_internal::gumbel_parameter_to_tau,
        scar_internal::gumbel_log_pdf_unrotated,
        scar_internal::gumbel_dlog_pdf_dr_unrotated,
        scar_internal::gumbel_pair_pdf_and_gradient,
        scar_internal::gumbel_h_unrotated,
        scar_internal::gumbel_h_inverse_unrotated,
        scar_internal::gumbel_fill_density_grid_row,
        scar_internal::gumbel_fill_density_gradient_grid_row,
        scar_internal::gumbel_h_inverse_with_options,
        scar_internal::gumbel_h_reflected,
        scar_internal::gumbel_h_inverse_reflected,
        scar_internal::gumbel_h_inverse_reflected_with_options,
    };
    return functions;
}

}  // namespace scar::copula::pair
