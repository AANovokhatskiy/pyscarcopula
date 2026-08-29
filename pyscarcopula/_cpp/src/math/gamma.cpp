#include "scar/math/gamma.hpp"

#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar::math {

double log_gamma(double value) noexcept {
#if defined(__GLIBC__)
    int sign = 0;
    return ::lgamma_r(value, &sign);
#else
    return std::lgamma(value);
#endif
}

namespace {

constexpr int kMaximumIterations = 10000;
constexpr double kTolerance = 8.0 * std::numeric_limits<double>::epsilon();
constexpr double kFloor = 1e-300;

double gamma_series(double a, double x) noexcept {
    double term = 1.0 / a;
    double sum = term;
    double denominator = a;
    for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
        denominator += 1.0;
        term *= x / denominator;
        sum += term;
        if (std::abs(term) <= std::abs(sum) * kTolerance) break;
    }
    return sum * std::exp(-x + a * std::log(x) - log_gamma(a));
}

double gamma_continued_fraction_q(double a, double x) noexcept {
    double b = x + 1.0 - a;
    double c = 1.0 / kFloor;
    double d = 1.0 / std::max(std::abs(b), kFloor);
    if (b < 0.0) d = -d;
    double value = d;
    for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
        const double i = static_cast<double>(iteration);
        const double coefficient = -i * (i - a);
        b += 2.0;
        d = coefficient * d + b;
        if (std::abs(d) < kFloor) d = std::copysign(kFloor, d);
        c = b + coefficient / c;
        if (std::abs(c) < kFloor) c = std::copysign(kFloor, c);
        d = 1.0 / d;
        const double change = d * c;
        value *= change;
        if (std::abs(change - 1.0) <= kTolerance) break;
    }
    return std::exp(-x + a * std::log(x) - log_gamma(a)) * value;
}

}  // namespace

double regularized_gamma_p(double a, double x) noexcept {
    if (!std::isfinite(a) || !(a > 0.0) || std::isnan(x) || x < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) return 0.0;
    if (std::isinf(x)) return 1.0;
    const double value = x < a + 1.0
        ? gamma_series(a, x)
        : 1.0 - gamma_continued_fraction_q(a, x);
    return std::clamp(value, 0.0, 1.0);
}

double chi_square_quantile(double probability, double df) noexcept {
    if (!std::isfinite(probability) || !std::isfinite(df)
        || probability < 0.0 || probability > 1.0 || !(df > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (probability == 0.0) return 0.0;
    if (probability == 1.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double z = normal_quantile_refined(probability);
    const double correction = 1.0 - 2.0 / (9.0 * df)
        + z * std::sqrt(2.0 / (9.0 * df));
    const double initial = correction > 0.0
        ? df * correction * correction * correction
        : std::numeric_limits<double>::min();
    double lower = 0.0;
    double upper = std::max({initial, df, 1.0});
    while (regularized_gamma_p(0.5 * df, 0.5 * upper) < probability) {
        lower = upper;
        upper *= 2.0;
        if (!std::isfinite(upper)) {
            return std::numeric_limits<double>::infinity();
        }
    }
    double value = std::clamp(initial, lower, upper);
    for (int iteration = 0; iteration < 128; ++iteration) {
        const double cdf = regularized_gamma_p(0.5 * df, 0.5 * value);
        if (!std::isfinite(cdf)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (cdf < probability) lower = value;
        else upper = value;

        if (std::abs(cdf - probability)
            <= 8e-15 * std::max(probability, 1.0 - probability)) {
            break;
        }

        double candidate = 0.5 * (lower + upper);
        if (value > 0.0) {
            const double log_density = (0.5 * df - 1.0) * std::log(value)
                - 0.5 * value - 0.5 * df * std::log(2.0)
                - log_gamma(0.5 * df);
            const double density = std::exp(log_density);
            if (std::isfinite(density) && density > 0.0) {
                const double newton = value - (cdf - probability) / density;
                if (newton > lower && newton < upper) candidate = newton;
            }
        }
        if (upper - lower <= 8.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, candidate)) {
            value = candidate;
            break;
        }
        value = candidate;
    }
    return value;
}

}  // namespace scar::math
