#include "scar/copula/pair/joe.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

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
            // Preserve the legacy grid derivative's arithmetic. pow(q, r)
            // rounds differently and can change latent optimizer trajectories.
            const double q1r = std::exp(parameter * log_q1);
            const double q2r = std::exp(parameter * log_q2);
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

namespace {

bool joe_conditional_domain(double first, double second, double r) {
    return std::isfinite(first) && std::isfinite(second) && std::isfinite(r)
        && first >= 0.0 && first <= 1.0 && second >= 0.0 && second <= 1.0
        && r >= 1.0;
}







struct JoeConditionalValue {
    double log_negative_log_h;
    double derivative;
};

// x = -log(1-u), y = -log(1-v), t = theta*x, s = theta*y.
// log(h) = log(1-exp(-t)) - beta*softplus(s-t+log(1-exp(-s))).
// Work with log(-log(h)) so both probability tails survive underflow.
JoeConditionalValue joe_conditional_value(double z, double y, double r, double exact_x = -1.0) {
    const double log_t = std::log(r) + z;
    const double x = exact_x >= 0.0 ? exact_x : std::exp(z);
    const double t = r * x;
    const double beta = (r - 1.0) / r;
    const double delta = r * (y - x) + log1mexp(r * y);
    const double first = log_t < -36.0 ? log_t : log1mexp(t);
    const double log_first = t > 36.0 ? -t : std::log(-first);
    const double log_second = std::log(beta)
        + (delta < -36.0 ? delta : std::log(logsumexp(0.0, delta)));
    const double value = logsumexp(log_first, log_second);
    const double log_first_derivative = log_t < -36.0
        ? 0.0 : log_t - t - first;
    const double log_second_derivative = std::log(beta)
        + log_t - logsumexp(0.0, -delta);
    const double derivative = -std::exp(
        logsumexp(log_first_derivative, log_second_derivative) - value);
    return {value, derivative};
}

double joe_inverse_output(double z, bool reflected) {
    const double x = std::exp(z);
    return reflected ? std::exp(-x) : -std::expm1(-x);
}

double joe_inverse_impl(double q, double given, double r,
                        const scar::HInverseOptions& options, bool reflected) {
    if (!joe_conditional_domain(q, given, r)
        || !std::isfinite(options.tolerance) || options.tolerance <= 0.0
        || options.max_iterations <= 0) return std::numeric_limits<double>::quiet_NaN();
    if (q == 0.0 || q == 1.0 || r == 1.0) return q;
    if (given == 1.0) return reflected ? 0.0 : 1.0;
    const double log_q = reflected ? std::log1p(-q) : std::log(q);
    const double log_survival = reflected ? std::log(q) : std::log1p(-q);
    if (given == 0.0) {
        return reflected ? std::exp(log_survival / r)
                         : -std::expm1(log_survival / r);
    }
    const double y = -std::log1p(-given);
    const double target = std::log(-log_q);
    // h <= 1-exp(-t), and h <= (1+exp(s-t)*(1-exp(-s)))^(-beta).
    // These give two rigorous lower bounds. The upper bound ensures h >= q.
    double lo = std::log(-log_survival) - std::log(r);
    const double upper_x = y + (0.6931471805599453094 - log_survival) / r;
    double hi = std::log(upper_x);
    const double c = -log_q / ((r - 1.0) / r);
    const double lower_x = y + (log1mexp(r * y)
        - (c + log1mexp(c))) / r;
    if (lower_x > 0.0) lo = std::max(lo, std::log(lower_x));
    if (lo >= hi) {
        // The certified bounds coincide at floating-point resolution.
        if (lower_x == y && upper_x == y) return reflected ? 1.0 - given : given;
        return joe_inverse_output(hi, reflected);
    }
    const double guess = y + (log_q - log_survival) / r;
    double z = guess > 0.0 ? std::log(guess) : lo;
    z = std::clamp(z, lo, hi);
    const double tolerance = std::min(options.tolerance, 2e-14);
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        const auto value = joe_conditional_value(z, y, r);
        const double error = value.log_negative_log_h - target;
        const double output = joe_inverse_output(z, reflected);
        if (std::abs(error) <= tolerance) return output;
        if (error < 0.0) hi = z;
        else lo = z;
        if (std::nextafter(lo, std::numeric_limits<double>::infinity()) >= hi) {
            return output;
        }
        const double output_lo = joe_inverse_output(lo, reflected);
        const double output_hi = joe_inverse_output(hi, reflected);
        if (std::nextafter(std::min(output_lo, output_hi), 1.0)
            >= std::max(output_lo, output_hi)) return output;

        double next = z - error / value.derivative;
        if (std::isfinite(next) && std::abs(next - z) < 1e-12 * std::max(1.0, std::abs(z))) {
            // A large theta can make the target fall between adjacent roots;
            // certify that bracket instead of demanding an impossible residual.
            // Near z=0 many adjacent z values exponentiate to the same x, so
            // certify adjacency in x as well as in the logarithmic coordinate.
            const double x = std::exp(z);
            const double x_before = std::nextafter(x, 0.0);
            const double x_after = std::nextafter(x, std::numeric_limits<double>::infinity());
            if (x_before > 0.0
                && joe_conditional_value(std::log(x_after), y, r, x_after).log_negative_log_h <= target
                && joe_conditional_value(std::log(x_before), y, r, x_before).log_negative_log_h >= target) return output;
            const double before = std::nextafter(z, -std::numeric_limits<double>::infinity());
            const double after = std::nextafter(z, std::numeric_limits<double>::infinity());
            if (joe_conditional_value(after, y, r).log_negative_log_h <= target
                && joe_conditional_value(before, y, r).log_negative_log_h >= target) return output;
        }
        if (!std::isfinite(next) || next <= lo || next >= hi) next = lo + 0.5 * (hi - lo);
        z = next;
    }
    throw std::runtime_error("Joe conditional inverse did not converge within max_iterations");
}

}  // namespace

double joe_h_unrotated(double u, double v, double r) {
    if (!joe_conditional_domain(u, v, r)) return std::numeric_limits<double>::quiet_NaN();
    if (u == 0.0 || u == 1.0 || r == 1.0) return u;
    if (v == 1.0) return 0.0;
    if (v == 0.0) return -std::expm1(r * std::log1p(-u));
    const double x = -std::log1p(-u);
    const double value = joe_conditional_value(
        std::log(x), -std::log1p(-v), r, x).log_negative_log_h;
    return std::exp(-std::exp(value));
}

double joe_h_reflected(double u, double v, double r) {
    if (!joe_conditional_domain(u, v, r)) return std::numeric_limits<double>::quiet_NaN();
    if (u == 0.0 || u == 1.0 || r == 1.0) return u;
    if (v == 1.0) return 1.0;
    if (v == 0.0) return std::exp(r * std::log(u));
    const double x = -std::log(u);
    const double value = joe_conditional_value(
        std::log(x), -std::log1p(-v), r, x).log_negative_log_h;
    return -std::expm1(-std::exp(value));
}

double joe_h_inverse_with_options(double q, double given, double r,
                                  const scar::HInverseOptions& options) {
    return joe_inverse_impl(q, given, r, options, false);
}

double joe_h_inverse_unrotated(double q, double given, double r) {
    return joe_inverse_impl(q, given, r, {1e-10, 50}, false);
}

double joe_h_inverse_reflected_with_options(double q, double given, double r,
                                            const scar::HInverseOptions& options) {
    return joe_inverse_impl(q, given, r, options, true);
}

double joe_h_inverse_reflected(double q, double given, double r) {
    return joe_inverse_impl(q, given, r, {1e-10, 50}, true);
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
        scar_internal::joe_h_reflected,
        scar_internal::joe_h_inverse_reflected,
        scar_internal::joe_h_inverse_reflected_with_options,
    };
    return functions;
}

}  // namespace scar::copula::pair
