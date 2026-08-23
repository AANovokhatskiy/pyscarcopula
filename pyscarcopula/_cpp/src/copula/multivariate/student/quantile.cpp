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

double student_quantile(double p, double df) {
    p = clip_pseudo_observation(p);
    if (p == 0.5) {
        return 0.0;
    }

    const bool negative = p < 0.5;
    const double tail_probability = negative ? p : 1.0 - p;
    const double initial_probability = 1.0 - tail_probability;
    const double initial = std::abs(
        student_quantile_initial(initial_probability, df));
    double lo = 0.0;
    double hi = std::max(1.0, initial);
    while (student_survival_positive_value(hi, df) > tail_probability
           && hi < 1e12) {
        hi *= 2.0;
    }

    double x = std::min(std::max(initial, lo), hi);
    for (int iter = 0; iter < 50; ++iter) {
        const double survival = student_survival_positive_value(x, df);
        const double error = survival - tail_probability;
        if (error > 0.0) {
            lo = x;
        } else {
            hi = x;
        }
        if (std::abs(error) <= 2e-13 * tail_probability
            || hi - lo <= 2e-13 * std::max(1.0, std::abs(x))) {
            return negative ? -x : x;
        }

        const double pdf = student_pdf_value(x, df);
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

void student_quantile_exact_with_derivative(
    double p,
    double df,
    double& value,
    double& derivative) {

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
    // likelihoods carry no nodes and therefore retain exact quantiles.
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
