#include "scar/copula/multivariate/student/distribution.hpp"

#include "scar/detail/safety.hpp"
#include "scar/math/beta.hpp"
#include "scar/math/gamma.hpp"

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
    return scar::math::log_gamma(value);
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

double student_log_pdf_normalization(
    std::size_t dimension, double df, double* derivative) {
    const double half_dimension = 0.5 * static_cast<double>(dimension);
    if (df < 32.0) {
        if (derivative != nullptr) {
            *derivative = 0.5 * (
                student_digamma_positive(0.5 * df + half_dimension)
                - student_digamma_positive(0.5 * df)) - half_dimension / df;
        }
        return student_log_gamma(0.5 * df + half_dimension)
            - student_log_gamma(0.5 * df)
            - half_dimension * (std::log(df) + std::log(kPi));
    }

    double value = -half_dimension * std::log(2.0 * kPi);
    double slope = 0.0;
    if (dimension % 2 != 0) {
        // Stirling's log-gamma ratio for a half-step, through df^-11.
        // At df >= 32 the first omitted term is below 3e-18. Keeping the
        // half-step separate avoids losing it when df+1 rounds back to df.
        const double inverse = 1.0 / df;
        const double inverse2 = inverse * inverse;
        value += inverse * (-0.25 + inverse2 * (1.0 / 24.0
            + inverse2 * (-1.0 / 20.0 + inverse2 * (17.0 / 112.0
            + inverse2 * (-31.0 / 36.0 + inverse2 * 691.0 / 88.0)))));
        slope = inverse2 * (0.25 + inverse2 * (-1.0 / 8.0
            + inverse2 * (0.25 + inverse2 * (-17.0 / 16.0
            + inverse2 * (31.0 / 4.0 - inverse2 * 691.0 / 8.0)))));
    }
    // Gamma recurrence gives exact integer steps in log1p form. Its cost
    // is linear in d, bounded by the density's existing marginal reduction.
    for (std::size_t index = dimension % 2; index < dimension; index += 2) {
        const double offset = static_cast<double>(index);
        value += std::log1p(offset / df);
        slope -= (offset / df) / (df + offset);
    }
    if (derivative != nullptr) {
        *derivative = slope;
    }
    return value;
}

double student_pdf_value(double value, double df) {
    const double log_pdf = student_log_pdf_normalization(1, df)
        - 0.5 * (df + 1.0) * std::log1p((value * value) / df);
    return std::exp(log_pdf);
}

double student_survival_positive_value(double value, double df) {
    const double x = df / (df + value * value);
    return 0.5 * scar::math::regularized_beta(x, 0.5 * df, 0.5);
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
    return student_cdf_refined_value(value, student_distribution_parameters(df));
}

StudentDistributionParameters student_distribution_parameters(double df) {
    if (df < 32.0) {
        // Preserve the beta normalization's rounding in iterative CDF
        // inversion. Converting via log-PDF and back changes Newton stopping.
        const double log_beta = student_log_gamma(0.5 * (df + 1.0))
            - student_log_gamma(0.5 * df) - student_log_gamma(0.5);
        return {df, log_beta, log_beta - 0.5 * std::log(df)};
    }
    const double log_pdf = student_log_pdf_normalization(1, df);
    return {df, log_pdf + 0.5 * std::log(df), log_pdf};
}

double student_pdf_value(double value, const StudentDistributionParameters& params) {
    return std::exp(params.log_pdf_normalization
        - 0.5 * (params.df + 1.0) * std::log1p(value * value / params.df));
}

double student_cdf_refined_value(double value, const StudentDistributionParameters& params) {
    const double df = params.df;
    if (!std::isfinite(value) || !std::isfinite(df) || df <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double magnitude = std::abs(value);
    if (magnitude < 1.0) {
        // Form the small beta argument directly. Subtracting df/(df+t*t)
        // from one loses the significant bits needed by quantiles near 1/2.
        const double squared = magnitude * magnitude;
        const double central_mass = 0.5 * scar::math::regularized_beta(
            squared / (df + squared), 0.5, 0.5 * df,
            4.0 * std::numeric_limits<double>::epsilon(),
            params.log_beta_normalization, true);
        return value >= 0.0 ? 0.5 + central_mass : 0.5 - central_mass;
    }
    const double x = df / (df + magnitude * magnitude);
    const double survival = 0.5 * scar::math::regularized_beta(
        x,
        0.5 * df,
        0.5,
        4.0 * std::numeric_limits<double>::epsilon(),
        params.log_beta_normalization, true);
    const double cdf = value >= 0.0 ? 1.0 - survival : survival;
    return std::clamp(cdf, 0.0, 1.0);
}

}  // namespace scar_internal
