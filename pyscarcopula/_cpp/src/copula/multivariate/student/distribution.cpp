#include "scar/copula/multivariate/student/distribution.hpp"

#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {
namespace {

struct DualValue {
    double value;
    double derivative;
};

DualValue operator+(DualValue lhs, DualValue rhs) {
    return {
        lhs.value + rhs.value,
        lhs.derivative + rhs.derivative,
    };
}

DualValue operator-(DualValue lhs, DualValue rhs) {
    return {
        lhs.value - rhs.value,
        lhs.derivative - rhs.derivative,
    };
}

DualValue operator-(DualValue value) {
    return {-value.value, -value.derivative};
}

DualValue operator*(DualValue lhs, DualValue rhs) {
    return {
        lhs.value * rhs.value,
        lhs.derivative * rhs.value + lhs.value * rhs.derivative,
    };
}

DualValue operator/(DualValue lhs, DualValue rhs) {
    const double inverse = 1.0 / rhs.value;
    return {
        lhs.value * inverse,
        (lhs.derivative - lhs.value * inverse * rhs.derivative) * inverse,
    };
}

DualValue dual(double value, double derivative = 0.0) {
    return {value, derivative};
}

double betacf(
    double a,
    double b,
    double x,
    double requested_tolerance = 3e-14) {
    constexpr int max_iter = 200;
    constexpr double eps = 3e-14;
    const double convergence_tolerance = std::min(
        eps, requested_tolerance);
    constexpr double fpmin = 1e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < fpmin) {
        d = fpmin;
    }
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= max_iter; ++m) {
        const int m2 = 2 * m;
        double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x
            / ((qam + static_cast<double>(m2)) * (a + static_cast<double>(m2)));
        d = 1.0 + aa * d;
        if (std::abs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x
            / ((a + static_cast<double>(m2)) * (qap + static_cast<double>(m2)));
        d = 1.0 + aa * d;
        if (std::abs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::abs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < convergence_tolerance) {
            break;
        }
    }
    return h;
}

DualValue betacf_dual(DualValue a, DualValue b, DualValue x) {
    constexpr int max_iter = 200;
    constexpr double eps = 3e-14;
    constexpr double fpmin = 1e-300;

    const DualValue one = dual(1.0);
    const DualValue qab = a + b;
    const DualValue qap = a + one;
    const DualValue qam = a - one;
    DualValue c = one;
    DualValue d = one - qab * x / qap;
    if (std::abs(d.value) < fpmin) {
        d = dual(fpmin);
    }
    d = one / d;
    DualValue h = d;

    for (int m = 1; m <= max_iter; ++m) {
        const double m_value = static_cast<double>(m);
        const double m2 = static_cast<double>(2 * m);
        DualValue aa = (
            dual(m_value) * (b - dual(m_value)) * x
            / ((qam + dual(m2)) * (a + dual(m2))));
        d = one + aa * d;
        if (std::abs(d.value) < fpmin) {
            d = dual(fpmin);
        }
        c = one + aa / c;
        if (std::abs(c.value) < fpmin) {
            c = dual(fpmin);
        }
        d = one / d;
        h = h * d * c;

        aa = -(
            (a + dual(m_value)) * (qab + dual(m_value)) * x
            / ((a + dual(m2)) * (qap + dual(m2))));
        d = one + aa * d;
        if (std::abs(d.value) < fpmin) {
            d = dual(fpmin);
        }
        c = one + aa / c;
        if (std::abs(c.value) < fpmin) {
            c = dual(fpmin);
        }
        d = one / d;
        const DualValue change = d * c;
        h = h * change;
        if (std::abs(change.value - 1.0) < eps) {
            break;
        }
    }
    return h;
}

double regularized_beta(
    double x,
    double a,
    double b,
    double tolerance = 3e-14) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    const double bt = std::exp(
        student_log_gamma(a + b)
        - student_log_gamma(a)
        - student_log_gamma(b)
        + a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x, tolerance) / a;
    }
    return 1.0 - bt * betacf(b, a, 1.0 - x, tolerance) / b;
}

DualValue regularized_beta_dual(
    DualValue x,
    DualValue a,
    DualValue b) {

    if (x.value <= 0.0) {
        return dual(0.0);
    }
    if (x.value >= 1.0) {
        return dual(1.0);
    }
    const double log_bt =
        student_log_gamma(a.value + b.value)
        - student_log_gamma(a.value)
        - student_log_gamma(b.value)
        + a.value * std::log(x.value)
        + b.value * std::log1p(-x.value);
    const double log_bt_derivative =
        student_digamma_positive(a.value + b.value)
            * (a.derivative + b.derivative)
        - student_digamma_positive(a.value) * a.derivative
        - student_digamma_positive(b.value) * b.derivative
        + a.derivative * std::log(x.value)
        + a.value * x.derivative / x.value
        + b.derivative * std::log1p(-x.value)
        - b.value * x.derivative / (1.0 - x.value);
    const double bt_value = std::exp(log_bt);
    const DualValue bt = dual(
        bt_value, bt_value * log_bt_derivative);
    if (x.value < (a.value + 1.0) / (a.value + b.value + 2.0)) {
        return bt * betacf_dual(a, b, x) / a;
    }
    return dual(1.0)
        - bt * betacf_dual(b, a, dual(1.0) - x) / b;
}

}  // namespace

double student_log_gamma(double value) {
#if defined(__GLIBC__)
    int sign = 0;
    return ::lgamma_r(value, &sign);
#else
    return std::lgamma(value);
#endif
}

double student_digamma_positive(double x) {
    double result = 0.0;
    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv
        - inv2 * (
            1.0 / 12.0
            - inv2 * (
                1.0 / 120.0
                - inv2 * (
                    1.0 / 252.0
                    - inv2 * (
                        1.0 / 240.0
                        - inv2 * (5.0 / 660.0)))));
    return result;
}

double student_pdf_value(double value, double df) {
    const double log_pdf =
        student_log_gamma(0.5 * (df + 1.0))
        - student_log_gamma(0.5 * df)
        - 0.5 * std::log(df * kPi)
        - 0.5 * (df + 1.0) * std::log1p((value * value) / df);
    return std::exp(log_pdf);
}

double student_survival_positive_value(double value, double df) {
    const double x = df / (df + value * value);
    return 0.5 * regularized_beta(x, 0.5 * df, 0.5);
}

void student_survival_positive_df_value_and_derivative(
    double value,
    double df,
    double& survival,
    double& derivative) {

    const double value_squared = value * value;
    const double denominator = df + value_squared;
    const DualValue x = dual(
        df / denominator,
        value_squared / (denominator * denominator));
    const DualValue result = dual(0.5) * regularized_beta_dual(
        x, dual(0.5 * df, 0.5), dual(0.5));
    survival = result.value;
    derivative = result.derivative;
}

double student_cdf_value(double value, double df) {
    if (!std::isfinite(value) || !std::isfinite(df) || df <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double cdf = value >= 0.0
        ? 1.0 - student_survival_positive_value(value, df)
        : student_survival_positive_value(-value, df);
    return std::clamp(cdf, 0.0, 1.0);
}

double student_cdf_refined_value(double value, double df) {
    if (!std::isfinite(value) || !std::isfinite(df) || df <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double magnitude = std::abs(value);
    const double x = df / (df + magnitude * magnitude);
    const double survival = 0.5 * regularized_beta(
        x,
        0.5 * df,
        0.5,
        4.0 * std::numeric_limits<double>::epsilon());
    const double cdf = value >= 0.0 ? 1.0 - survival : survival;
    return std::clamp(cdf, 0.0, 1.0);
}

}  // namespace scar_internal
