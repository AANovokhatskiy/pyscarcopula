#include "scar/math/beta.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar::math {
namespace {

double continued_fraction(
    double a, double b, double x, double requested_tolerance) noexcept {

    constexpr int maximum_iterations = 400;
    constexpr double floor = 1e-300;
    const double tolerance = std::min(3e-14, requested_tolerance);
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < floor) d = floor;
    d = 1.0 / d;
    double value = d;
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        const double m = static_cast<double>(iteration);
        const double m2 = 2.0 * m;
        double coefficient = m * (b - m) * x
            / ((qam + m2) * (a + m2));
        d = 1.0 + coefficient * d;
        if (std::abs(d) < floor) d = floor;
        c = 1.0 + coefficient / c;
        if (std::abs(c) < floor) c = floor;
        d = 1.0 / d;
        value *= d * c;

        coefficient = -(a + m) * (qab + m) * x
            / ((a + m2) * (qap + m2));
        d = 1.0 + coefficient * d;
        if (std::abs(d) < floor) d = floor;
        c = 1.0 + coefficient / c;
        if (std::abs(c) < floor) c = floor;
        d = 1.0 / d;
        const double change = d * c;
        value *= change;
        if (std::abs(change - 1.0) <= tolerance) break;
    }
    return value;
}

}  // namespace

double regularized_beta(
    double x,
    double a,
    double b,
    double tolerance,
    double log_normalization,
    bool has_log_normalization) noexcept {

    if (!std::isfinite(x) || !std::isfinite(a) || !std::isfinite(b)
        || !(a > 0.0) || !(b > 0.0) || !(tolerance > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (!has_log_normalization) {
        log_normalization = std::lgamma(a + b)
            - std::lgamma(a) - std::lgamma(b);
    }
    const double scale = std::exp(
        log_normalization + a * std::log(x) + b * std::log1p(-x));
    const double value = x < (a + 1.0) / (a + b + 2.0)
        ? scale * continued_fraction(a, b, x, tolerance) / a
        : 1.0 - scale * continued_fraction(b, a, 1.0 - x, tolerance) / b;
    return std::clamp(value, 0.0, 1.0);
}

double beta_quantile(double probability, double a, double b) noexcept {
    if (!std::isfinite(probability) || !std::isfinite(a) || !std::isfinite(b)
        || probability < 0.0 || probability > 1.0
        || !(a > 0.0) || !(b > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (probability == 0.0) return 0.0;
    if (probability == 1.0) return 1.0;

    // Reflect upper-tail queries so the continued fraction always resolves
    // the smaller probability and retains accuracy for boundary shapes.
    const bool reflected = probability > 0.5;
    const double target = reflected ? 1.0 - probability : probability;
    const double left_shape = reflected ? b : a;
    const double right_shape = reflected ? a : b;
    double lower = 0.0;
    double upper = 1.0;
    double value = left_shape / (left_shape + right_shape);
    for (int iteration = 0; iteration < 128; ++iteration) {
        const double cdf = regularized_beta(
            value, left_shape, right_shape, 2e-15);
        if (!std::isfinite(cdf)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (cdf < target) lower = value;
        else upper = value;
        if (std::abs(cdf - target)
            <= 4e-15 * std::max(target, 1e-300)) {
            break;
        }
        const double midpoint = 0.5 * (lower + upper);
        if (upper - lower <= 4.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, midpoint)) {
            value = midpoint;
            break;
        }
        value = midpoint;
    }
    const double result = reflected ? 1.0 - value : value;
    return std::clamp(result, 0.0, 1.0);
}

}  // namespace scar::math
