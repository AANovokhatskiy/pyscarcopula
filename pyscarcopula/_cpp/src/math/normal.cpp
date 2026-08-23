#include "scar/math/normal.hpp"

#include "scar/numerical_constants.hpp"

#include <algorithm>
#include <cmath>

namespace scar::math {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double clip_probability(double probability) noexcept {
    constexpr double epsilon = numerical::kPseudoObservationEps;
    return std::clamp(probability, epsilon, 1.0 - epsilon);
}

}  // namespace

double normal_cdf(double value) noexcept {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double normal_quantile(double p) {
    const double a[] = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    };
    const double b[] = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    };
    const double c[] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    };
    const double d[] = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    };

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;
    p = clip_probability(p);

    double x = 0.0;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p > phigh) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    return x;
}

double normal_quantile_refined(double p) {
    p = clip_probability(p);
    const double x = normal_quantile(p);
    const double cdf = normal_cdf(x);
    const double pdf =
        std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
    return x - (cdf - p) / pdf;
}

}  // namespace scar::math
