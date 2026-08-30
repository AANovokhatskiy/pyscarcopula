#include "scar/copula/multivariate/student/quantile.hpp"

#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {
namespace {

constexpr double kStudentNormalAsymptoticDf = 1000.0;

void student_quantile_large_df_refined(
    double p, double df, double& value, double* derivative);

double student_quantile_initial(double p, double df) {
    const double z = scar::math::normal_quantile(p);
    const double z2 = z * z;
    const double z3 = z * z2;
    const double z5 = z3 * z2;
    const double z7 = z5 * z2;
    const double inv_df = 1.0 / df;
    const double inv_df2 = inv_df * inv_df;
    const double inv_df3 = inv_df2 * inv_df;
    return z
        + 0.25 * (z3 + z) * inv_df
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) * inv_df2 / 96.0
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z)
            * inv_df3 / 384.0;
}

double student_quantile(
    double p,
    double df,
    double tolerance = 2e-13,
    int maximum_iterations = 50,
    bool refined_cdf = false,
    const StudentDistributionParameters* prepared = nullptr,
    double initial_quantile = std::numeric_limits<double>::quiet_NaN()) {
    p = clip_pseudo_observation(p);
    if (df >= kStudentNormalAsymptoticDf) {
        double value = 0.0;
        student_quantile_large_df_refined(p, df, value, nullptr);
        return value;
    }
    if (p == 0.5) {
        return 0.0;
    }

    const bool negative = p < 0.5;
    const double tail_probability = negative ? p : 1.0 - p;
    const double initial_probability = 1.0 - tail_probability;
    const double initial = std::isfinite(initial_quantile)
        ? std::abs(initial_quantile)
        : std::abs(student_quantile_initial(initial_probability, df));
    double lo = 0.0;
    double hi = std::isfinite(initial_quantile) && initial > 0.0
        ? initial : std::max(1.0, initial);
    const auto survival_value = [df, refined_cdf, prepared](double value) {
        if (prepared != nullptr) return student_cdf_refined_value(-value, *prepared);
        return refined_cdf
            ? student_cdf_refined_value(-value, df)
            : student_survival_positive_value(value, df);
    };
    double hi_survival = survival_value(hi);
    while (hi_survival > tail_probability && hi < 1e12) {
        hi *= 2.0;
        hi_survival = survival_value(hi);
    }

    double x = std::min(std::max(initial, lo), hi);
    for (int iter = 0; iter < maximum_iterations; ++iter) {
        const double survival = iter == 0 && x == hi
            ? hi_survival : survival_value(x);
        const double error = survival - tail_probability;
        if (error > 0.0) {
            lo = x;
        } else {
            hi = x;
        }
        if (std::abs(error) <= tolerance * tail_probability
            || hi - lo <= tolerance * std::max(1.0, std::abs(x))) {
            return negative ? -x : x;
        }

        const double pdf = prepared != nullptr
            ? student_pdf_value(x, *prepared) : student_pdf_value(x, df);
        const double candidate = x + error / pdf;
        if (std::isfinite(candidate) && candidate > lo && candidate < hi) {
            x = candidate;
        } else {
            x = 0.5 * (lo + hi);
        }
    }
    const double result = 0.5 * (lo + hi);
    return negative ? -result : result;
}

void student_quantile_large_df(
    double p,
    double df,
    double& value,
    double* derivative) {

    const double z = scar::math::normal_quantile_refined(
        clip_pseudo_observation(p));
    const double z2 = z * z;
    const double z3 = z * z2;
    const double z5 = z3 * z2;
    const double z7 = z5 * z2;
    const double a1 = 0.25 * (z3 + z);
    const double a2 =
        (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0;
    const double a3 =
        (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z)
        / 384.0;
    const double inv_df = 1.0 / df;
    const double inv_df2 = inv_df * inv_df;
    const double inv_df3 = inv_df2 * inv_df;
    value = z + a1 * inv_df + a2 * inv_df2 + a3 * inv_df3;
    if (derivative != nullptr) {
        *derivative = -a1 * inv_df2
            - 2.0 * a2 * inv_df3
            - 3.0 * a3 * inv_df3 * inv_df;
    }
}

void student_quantile_large_df_refined(
    double p, double df, double& value, double* derivative) {
    // Expand the normalized Student CDF inverse through O(df^-6).
    // Unlike CDF inversion, this remains well-conditioned in the normal limit.
    student_quantile_large_df(p, df, value, derivative);
    const double z = scar::math::normal_quantile_refined(clip_pseudo_observation(p));
    const double s = z * z;
    const double a4 = z * (-21.0 / 2048.0 + s * (-1.0 / 48.0
        + s * (247.0 / 15360.0 + s * (97.0 / 11520.0 + s * 79.0 / 92160.0))));
    const double a5 = z * (399.0 / 8192.0 + s * (-17.0 / 8192.0
        + s * (-99.0 / 20480.0 + s * (31.0 / 12288.0
        + s * (113.0 / 122880.0 + s * 3.0 / 40960.0)))));
    const double a6 = z * (869.0 / 65536.0 + s * (147.0 / 4096.0
        + s * (3263.0 / 983040.0 + s * (-229.0 / 516096.0
        + s * (48821.0 / 185794560.0 + s * (1931.0 / 23224320.0
        + s * 71.0 / 12386304.0))))));
    const double inverse = 1.0 / df;
    const double inverse2 = inverse * inverse;
    const double inverse4 = inverse2 * inverse2;
    value += inverse4 * (a4 + inverse * (a5 + inverse * a6));
    if (derivative != nullptr) {
        *derivative -= inverse4 * inverse
            * (4.0 * a4 + inverse * (5.0 * a5 + inverse * 6.0 * a6));
    }
}

void student_quantile_exact_with_derivative(
    double p,
    double df,
    double& value,
    double& derivative) {
    if (df >= kStudentNormalAsymptoticDf) {
        student_quantile_large_df_refined(p, df, value, &derivative);
        return;
    }
    value = student_quantile(p, df);
    if (value == 0.0) {
        derivative = 0.0;
        return;
    }
    const double magnitude = std::abs(value);
    double survival = 0.0;
    double survival_derivative = 0.0;
    student_survival_positive_df_value_and_derivative(
        magnitude, df, survival, survival_derivative);
    const double slope =
        survival_derivative / student_pdf_value(magnitude, df);
    derivative = value < 0.0 ? -slope : slope;
}

}  // namespace

bool use_large_df_quantile(
    const scar::CopulaSpec& spec,
    double df) {

    const auto& cache =
        scar::copula::multivariate::student::ppf_cache(spec);
    return use_large_df_quantile(cache, df);
}

bool use_large_df_quantile(
    const scar::copula::multivariate::student::PpfCache& cache,
    double df) {

    // A populated dynamic PPF cache defines the point beyond which the
    // third-order Cornish-Fisher expansion is used. Static Student
    // likelihoods carry no nodes and use the sixth-order normal-limit kernel.
    return !cache.nodes.empty()
        && df > std::max(
            kStudentNormalAsymptoticDf, cache.nodes.back());
}

void student_quantile_for_emission(
    const scar::CopulaSpec& spec,
    double p,
    double df,
    double& value,
    double* derivative) {

    const auto& cache =
        scar::copula::multivariate::student::ppf_cache(spec);
    student_quantile_for_emission(cache, p, df, value, derivative);
}

void student_quantile_for_emission(
    const scar::copula::multivariate::student::PpfCache& cache,
    double p,
    double df,
    double& value,
    double* derivative) {

    if (use_large_df_quantile(cache, df)) {
        student_quantile_large_df(p, df, value, derivative);
        return;
    }
    if (derivative == nullptr) {
        value = student_quantile(p, df);
        return;
    }
    student_quantile_exact_with_derivative(
        p, df, value, *derivative);
}

double student_quantile_value(double p, double df) {
    return student_quantile(p, df);
}

double student_quantile_refined_value(double p, double df) {
    return student_quantile_refined_value(p, student_distribution_parameters(df));
}

double student_quantile_refined_value(double p, const StudentDistributionParameters& params) {
    return student_quantile_refined_value(
        p, params, std::numeric_limits<double>::quiet_NaN());
}

double student_quantile_refined_value(
    double p, const StudentDistributionParameters& params, double initial_quantile) {
    const double df = params.df;
    if (df >= kStudentNormalAsymptoticDf) {
        double value = 0.0;
        student_quantile_large_df_refined(p, df, value, nullptr);
        return value;
    }
    return student_quantile(
        p,
        df,
        16.0 * std::numeric_limits<double>::epsilon(),
        80,
        true,
        &params,
        initial_quantile);
}

double student_quantile_for_observation(
    const scar::CopulaSpec& spec,
    double p,
    double df,
    std::int64_t row_index,
    int column) {

    const auto& cache =
        scar::copula::multivariate::student::ppf_cache(spec);
    const bool use_cache =
        column >= 0
        && column < spec.dim
        && student_ppf_cache_available(cache, spec.dim, row_index)
        && df >= cache.nodes.front()
        && df <= cache.nodes.back();
    if (use_cache) {
        const PpfInterpolation interpolation =
            make_ppf_interpolation(cache.nodes, df);
        return interpolate_ppf_value(
            cache,
            spec.dim,
            interpolation,
            row_index,
            column,
            nullptr);
    }

    double value = std::numeric_limits<double>::quiet_NaN();
    student_quantile_for_emission(cache, p, df, value, nullptr);
    return value;
}

void student_quantile_value_and_derivative(
    double p,
    double df,
    double& value,
    double& derivative) {

    student_quantile_exact_with_derivative(p, df, value, derivative);
}

void student_quantile_large_df_value_and_derivative(
    double p,
    double df,
    double& value,
    double& derivative) {

    student_quantile_large_df(p, df, value, &derivative);
}

}  // namespace scar_internal
