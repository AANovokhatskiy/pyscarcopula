#include "scar/detail/copula.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {
namespace {

double digamma_positive(double x) {
    double result = 0.0;
    while (x < 8.0) {
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

double frank_tau(double theta) {
    if (!std::isfinite(theta) || theta <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (theta < 1e-4) {
        const double theta2 = theta * theta;
        return theta / 9.0
            - theta * theta2 / 900.0
            + theta * theta2 * theta2 / 52920.0;
    }

    const long double value = static_cast<long double>(theta);
    long double integral = 0.0L;
    if (theta <= 2.0) {
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

double joe_tau(double theta) {
    if (!std::isfinite(theta) || theta < 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (theta == 1.0) {
        return 0.0;
    }
    const double euler_gamma = 0.577215664901532860606512090082402431;
    const double a = 2.0 / theta;
    double f_value = 0.0;
    if (std::abs(a - 1.0) < 1e-10) {
        f_value = kPi * kPi / 6.0 - 1.0;
    } else {
        f_value =
            (digamma_positive(a) + euler_gamma) / (a - 1.0)
            - (digamma_positive(a + 1.0) + euler_gamma) / a;
    }
    return std::clamp(
        1.0 - 4.0 * f_value / (theta * theta),
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

double copula_tau_to_param(const scar::CopulaSpec& spec, double tau) {
    if (!std::isfinite(tau)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        if (tau <= -1.0 || tau >= 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sin(0.5 * kPi * tau);
    }
    if (tau <= 0.0 || tau >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (spec.family == scar::CopulaFamily::Clayton) {
        return 2.0 * tau / (1.0 - tau);
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        return 1.0 / (1.0 - tau);
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return invert_positive_tau(
            tau,
            1e-12,
            std::max(1.0, 4.0 / (1.0 - tau)),
            frank_tau);
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        return invert_positive_tau(
            tau,
            1.0,
            std::max(2.0, 2.0 / (1.0 - tau)),
            joe_tau);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_param_to_tau(const scar::CopulaSpec& spec, double r) {
    if (!std::isfinite(r)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        if (r <= -1.0 || r >= 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 2.0 * std::asin(r) / kPi;
    }
    if (spec.family == scar::CopulaFamily::Clayton) {
        return r > 0.0
            ? r / (r + 2.0)
            : std::numeric_limits<double>::quiet_NaN();
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        return r >= 1.0
            ? 1.0 - 1.0 / r
            : std::numeric_limits<double>::quiet_NaN();
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return frank_tau(r);
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        return joe_tau(r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace scar_internal
