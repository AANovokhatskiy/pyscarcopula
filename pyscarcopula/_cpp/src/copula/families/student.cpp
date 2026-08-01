#include "scar/detail/copula.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/factor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace scar_internal {
namespace {

bool valid_factor_student_spec(const scar::CopulaSpec& spec) {
    return spec.correlation_kind == scar::CorrelationKind::Factor
        && spec.factor_correlation != nullptr
        && spec.dim >= 2
        && spec.factor_correlation->dimension()
            == static_cast<std::size_t>(spec.dim)
        && std::isfinite(spec.factor_correlation->logdet());
}

bool factor_precision_product(
    const scar::FactorCorrelationOperator& correlation,
    const std::vector<double>& values,
    StudentWorkspace& workspace) {

    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    if (values.size() != dimension) {
        return false;
    }
    workspace.resize_precision_x(dimension);
    workspace.resize_factor_small(rank);
    std::fill(
        workspace.factor_small.begin(),
        workspace.factor_small.end(),
        0.0);
    const std::vector<double>& weighted =
        correlation.weighted_loadings();
    for (std::size_t column = 0; column < dimension; ++column) {
        const double* loading = weighted.data() + column * rank;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            workspace.factor_small[factor] +=
                loading[factor] * values[column];
        }
    }
    correlation.solve_core_inplace(workspace.factor_small.data());
    const std::vector<double>& inverse_uniqueness =
        correlation.inverse_uniqueness();
    for (std::size_t column = 0; column < dimension; ++column) {
        const double* loading = weighted.data() + column * rank;
        double result = inverse_uniqueness[column] * values[column];
        for (std::size_t factor = 0; factor < rank; ++factor) {
            result -= loading[factor] * workspace.factor_small[factor];
        }
        if (!std::isfinite(result)) {
            return false;
        }
        workspace.precision_x[column] = result;
    }
    return true;
}

struct DualValue {
    double value;
    double derivative;
};

constexpr double kStudentNormalAsymptoticDf = 1000.0;

double thread_safe_lgamma(double value) {
#if defined(__GLIBC__)
    int sign = 0;
    return ::lgamma_r(value, &sign);
#else
    return std::lgamma(value);
#endif
}

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

double digamma_positive(double x);

double betacf(double a, double b, double x) {
    constexpr int max_iter = 200;
    constexpr double eps = 3e-14;
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
        if (std::abs(del - 1.0) < eps) {
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

double regularized_beta(double x, double a, double b) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    const double bt = std::exp(
        thread_safe_lgamma(a + b)
        - thread_safe_lgamma(a)
        - thread_safe_lgamma(b)
        + a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x) / a;
    }
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
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
        thread_safe_lgamma(a.value + b.value)
        - thread_safe_lgamma(a.value)
        - thread_safe_lgamma(b.value)
        + a.value * std::log(x.value)
        + b.value * std::log1p(-x.value);
    const double log_bt_derivative =
        digamma_positive(a.value + b.value)
            * (a.derivative + b.derivative)
        - digamma_positive(a.value) * a.derivative
        - digamma_positive(b.value) * b.derivative
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

double student_pdf(double t, double df) {
    const double log_pdf =
        thread_safe_lgamma(0.5 * (df + 1.0))
        - thread_safe_lgamma(0.5 * df)
        - 0.5 * std::log(df * kPi)
        - 0.5 * (df + 1.0) * std::log1p((t * t) / df);
    return std::exp(log_pdf);
}

double student_survival_positive(double t, double df) {
    const double x = df / (df + t * t);
    return 0.5 * regularized_beta(x, 0.5 * df, 0.5);
}

DualValue student_survival_positive_df(double t, double df) {
    const double t2 = t * t;
    const double denominator = df + t2;
    const DualValue x = dual(
        df / denominator,
        t2 / (denominator * denominator));
    return dual(0.5) * regularized_beta_dual(
        x, dual(0.5 * df, 0.5), dual(0.5));
}

double student_quantile_initial(double p, double df) {
    const double z = normal_quantile(p);
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
    while (student_survival_positive(hi, df) > tail_probability
           && hi < 1e12) {
        hi *= 2.0;
    }

    double x = std::min(std::max(initial, lo), hi);
    for (int iter = 0; iter < 50; ++iter) {
        const double survival = student_survival_positive(x, df);
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

        const double pdf = student_pdf(x, df);
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

    const double z = normal_quantile_refined(clip_pseudo_observation(p));
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
    const DualValue survival = student_survival_positive_df(magnitude, df);
    const double slope = survival.derivative / student_pdf(magnitude, df);
    derivative = value < 0.0 ? -slope : slope;
}

bool use_large_df_quantile(
    const scar::CopulaSpec& spec,
    double df) {

    // A populated dynamic PPF cache defines the point beyond which the
    // third-order Cornish-Fisher expansion is used.  Static Student
    // likelihoods carry no nodes and therefore retain exact quantiles.
    return !spec.ppf_nodes.empty()
        && df > std::max(
            kStudentNormalAsymptoticDf, spec.ppf_nodes.back());
}

void student_quantile_for_emission(
    const scar::CopulaSpec& spec,
    double p,
    double df,
    double& value,
    double* derivative) {

    if (use_large_df_quantile(spec, df)) {
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

bool student_ppf_cache_available(
    const scar::CopulaSpec& spec,
    std::int64_t row_index) {

    const std::size_t n_nodes = spec.ppf_nodes.size();
    std::size_t n_obs = 0;
    std::size_t dim = 0;
    std::size_t rows = 0;
    std::size_t expected = 0;
    if (spec.dim <= 0) {
        return false;
    }
    dim = static_cast<std::size_t>(spec.dim);
    if (!checked_nonnegative_size(spec.ppf_n_obs, n_obs)
        || !checked_size_mul(n_nodes, n_obs, rows)
        || !checked_size_mul(rows, dim, expected)) {
        return false;
    }
    return row_index >= 0
        && row_index < spec.ppf_n_obs
        && n_nodes >= 2
        && spec.ppf_table.size() == expected;
}

double digamma_positive(double x) {
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

struct PpfInterpolation {
    std::array<std::size_t, 4> node{};
    std::array<double, 4> value_weight{};
    std::array<double, 4> derivative_weight{};
    int count = 0;
};

void add_interpolation_weight(
    PpfInterpolation& interpolation,
    std::size_t node,
    double value_weight,
    double derivative_weight) {

    for (int i = 0; i < interpolation.count; ++i) {
        if (interpolation.node[static_cast<std::size_t>(i)] == node) {
            interpolation.value_weight[static_cast<std::size_t>(i)]
                += value_weight;
            interpolation.derivative_weight[static_cast<std::size_t>(i)]
                += derivative_weight;
            return;
        }
    }
    const std::size_t index =
        static_cast<std::size_t>(interpolation.count);
    interpolation.node[index] = node;
    interpolation.value_weight[index] = value_weight;
    interpolation.derivative_weight[index] = derivative_weight;
    ++interpolation.count;
}

PpfInterpolation make_ppf_interpolation(
    const std::vector<double>& nodes,
    double df) {

    const auto upper = std::upper_bound(nodes.begin(), nodes.end(), df);
    std::size_t idx = upper == nodes.begin()
        ? 0
        : static_cast<std::size_t>(upper - nodes.begin() - 1);
    idx = std::min(idx, nodes.size() - 2);

    const double lo = nodes[idx];
    const double hi = nodes[idx + 1];
    const double interval = hi - lo;
    const double alpha = (df - lo) / interval;
    PpfInterpolation interpolation;
    if (df <= nodes.front() || df >= nodes.back()) {
        add_interpolation_weight(
            interpolation, idx, 1.0 - alpha, -1.0 / interval);
        add_interpolation_weight(
            interpolation, idx + 1, alpha, 1.0 / interval);
        return interpolation;
    }

    const double alpha2 = alpha * alpha;
    const double alpha3 = alpha2 * alpha;
    const double h00 = 2.0 * alpha3 - 3.0 * alpha2 + 1.0;
    const double h10 = alpha3 - 2.0 * alpha2 + alpha;
    const double h01 = -2.0 * alpha3 + 3.0 * alpha2;
    const double h11 = alpha3 - alpha2;
    const double dh00 = (6.0 * alpha2 - 6.0 * alpha) / interval;
    const double dh10 = 3.0 * alpha2 - 4.0 * alpha + 1.0;
    const double dh01 = (-6.0 * alpha2 + 6.0 * alpha) / interval;
    const double dh11 = 3.0 * alpha2 - 2.0 * alpha;

    const std::size_t lo_slope_node = idx == 0 ? idx : idx - 1;
    const double lo_slope_interval =
        nodes[idx + 1] - nodes[lo_slope_node];
    const std::size_t hi_slope_node =
        idx + 1 == nodes.size() - 1 ? idx + 1 : idx + 2;
    const double hi_slope_interval =
        nodes[hi_slope_node] - nodes[idx];

    add_interpolation_weight(interpolation, idx, h00, dh00);
    add_interpolation_weight(interpolation, idx + 1, h01, dh01);
    add_interpolation_weight(
        interpolation,
        lo_slope_node,
        -h10 * interval / lo_slope_interval,
        -dh10 / lo_slope_interval);
    add_interpolation_weight(
        interpolation,
        idx + 1,
        h10 * interval / lo_slope_interval,
        dh10 / lo_slope_interval);
    add_interpolation_weight(
        interpolation,
        idx,
        -h11 * interval / hi_slope_interval,
        -dh11 / hi_slope_interval);
    add_interpolation_weight(
        interpolation,
        hi_slope_node,
        h11 * interval / hi_slope_interval,
        dh11 / hi_slope_interval);
    return interpolation;
}

double interpolate_ppf_value(
    const scar::CopulaSpec& spec,
    const PpfInterpolation& interpolation,
    std::int64_t row_index,
    int col,
    double* derivative) {

    const std::size_t node_stride =
        static_cast<std::size_t>(spec.ppf_n_obs)
        * static_cast<std::size_t>(spec.dim);
    const std::size_t row_offset =
        static_cast<std::size_t>(row_index)
        * static_cast<std::size_t>(spec.dim)
        + static_cast<std::size_t>(col);
    double value = 0.0;
    double derivative_value = 0.0;
    for (int i = 0; i < interpolation.count; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const double table_value = spec.ppf_table[
            interpolation.node[index] * node_stride + row_offset];
        value += interpolation.value_weight[index] * table_value;
        derivative_value += (
            interpolation.derivative_weight[index] * table_value);
    }
    if (derivative != nullptr) {
        *derivative = derivative_value;
    }
    return value;
}

void interpolate_bivariate_ppf(
    const scar::CopulaSpec& spec,
    const PpfInterpolation& interpolation,
    std::size_t observation,
    double& x1,
    double& x2,
    double& dx1,
    double& dx2) {

    const std::size_t row_offset = observation * 2;
    const std::size_t node_stride =
        static_cast<std::size_t>(spec.ppf_n_obs) * 2;
    x1 = 0.0;
    x2 = 0.0;
    dx1 = 0.0;
    dx2 = 0.0;
    for (int i = 0; i < interpolation.count; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const std::size_t offset =
            interpolation.node[index] * node_stride + row_offset;
        const double value_weight = interpolation.value_weight[index];
        const double derivative_weight =
            interpolation.derivative_weight[index];
        const double table_x1 = spec.ppf_table[offset];
        const double table_x2 = spec.ppf_table[offset + 1];
        x1 += value_weight * table_x1;
        x2 += value_weight * table_x2;
        dx1 += derivative_weight * table_x1;
        dx2 += derivative_weight * table_x2;
    }
}

double student_log_pdf_with_work(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace,
    double* dlog_ddf) {

    const int d = spec.dim;
    std::size_t matrix_elements = 0;
    const bool factor_correlation = valid_factor_student_spec(spec);
    if (d < 2
        || !valid_student_dimension(d, matrix_elements)
        || (!factor_correlation
            && spec.l_inv.size() != matrix_elements)
        || !std::isfinite(
            factor_correlation
                ? spec.factor_correlation->logdet()
                : spec.log_det)
        || !std::isfinite(df)
        || df <= 2.0) {
        return -std::numeric_limits<double>::infinity();
    }

    workspace.resize_x(static_cast<std::size_t>(d));
    const bool use_cache =
        student_ppf_cache_available(spec, row_index)
        && df >= spec.ppf_nodes.front()
        && df <= spec.ppf_nodes.back();
    const bool compute_derivative = dlog_ddf != nullptr;
    if (compute_derivative) {
        workspace.resize_dx_ddf(static_cast<std::size_t>(d));
    } else {
        workspace.dx_ddf.clear();
    }
    PpfInterpolation interpolation;
    if (use_cache) {
        interpolation = make_ppf_interpolation(spec.ppf_nodes, df);
        workspace.diagnostics.ppf_cache_values +=
            static_cast<std::uint64_t>(d);
    } else if (use_large_df_quantile(spec, df)) {
        workspace.diagnostics.ppf_asymptotic_values +=
            static_cast<std::uint64_t>(d);
    } else {
        workspace.diagnostics.ppf_exact_values +=
            static_cast<std::uint64_t>(d);
    }
    for (int i = 0; i < d; ++i) {
        if (use_cache) {
            workspace.x[static_cast<std::size_t>(i)] = interpolate_ppf_value(
                spec,
                interpolation,
                row_index,
                i,
                compute_derivative
                    ? &workspace.dx_ddf[static_cast<std::size_t>(i)]
                    : nullptr);
        } else {
            student_quantile_for_emission(
                spec,
                row[i],
                df,
                workspace.x[static_cast<std::size_t>(i)],
                compute_derivative
                    ? &workspace.dx_ddf[static_cast<std::size_t>(i)]
                    : nullptr);
        }
    }

    double quad = 0.0;
    double dquad_ddf = 0.0;
    if (factor_correlation) {
        if (!factor_precision_product(
                *spec.factor_correlation, workspace.x, workspace)) {
            return -std::numeric_limits<double>::infinity();
        }
        for (int i = 0; i < d; ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            quad += workspace.x[index] * workspace.precision_x[index];
            if (compute_derivative) {
                dquad_ddf +=
                    2.0
                    * workspace.precision_x[index]
                    * workspace.dx_ddf[index];
            }
        }
    } else {
        for (int i = 0; i < d; ++i) {
            double yi = 0.0;
            double dyi_ddf = 0.0;
            const std::size_t row_offset =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
            for (int j = 0; j <= i; ++j) {
                yi += spec.l_inv[
                    row_offset + static_cast<std::size_t>(j)]
                    * workspace.x[static_cast<std::size_t>(j)];
                if (compute_derivative) {
                    dyi_ddf += spec.l_inv[
                        row_offset + static_cast<std::size_t>(j)]
                        * workspace.dx_ddf[static_cast<std::size_t>(j)];
                }
            }
            quad += yi * yi;
            if (compute_derivative) {
                dquad_ddf += 2.0 * yi * dyi_ddf;
            }
        }
    }

    double log_pdf = -std::numeric_limits<double>::infinity();
    if (!student_log_pdf_from_quantiles(
            workspace.x.data(),
            compute_derivative ? workspace.dx_ddf.data() : nullptr,
            static_cast<std::size_t>(d),
            df,
            factor_correlation
                ? spec.factor_correlation->logdet()
                : spec.log_det,
            quad,
            dquad_ddf,
            log_pdf,
            dlog_ddf)) {
        return -std::numeric_limits<double>::infinity();
    }
    return log_pdf;
}

bool student_corr_score_row_impl(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    const std::vector<double>* direction,
    double* scores) {

    const int d = spec.dim;
    std::size_t matrix_elements = 0;
    if (!valid_student_dimension(d, matrix_elements)) {
        return false;
    }
    const std::size_t dim_size = static_cast<std::size_t>(d);
    std::size_t n_corr = 0;
    if (!valid_student_correlation_count(d, n_corr)) {
        return false;
    }
    std::size_t score_elements = 0;
    if (row == nullptr
        || scores == nullptr
        || d < 2
        || spec.l_inv.size() != matrix_elements
        || precision.size() != matrix_elements
        || (direction == nullptr
            && !checked_size_mul(
                df_grid.size(), n_corr, score_elements))
        || (direction != nullptr && direction->size() != n_corr)) {
        return false;
    }
    if (direction != nullptr) {
        for (double value : *direction) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }

    std::vector<double> x(static_cast<std::size_t>(d), 0.0);
    std::vector<double> whitened(static_cast<std::size_t>(d), 0.0);
    std::vector<double> precision_x(static_cast<std::size_t>(d), 0.0);
    for (std::size_t grid_index = 0;
         grid_index < df_grid.size();
         ++grid_index) {
        const double df = df_grid[grid_index];
        if (!std::isfinite(df) || df <= 2.0) {
            return false;
        }
        const bool use_cache =
            student_ppf_cache_available(spec, row_index)
            && df >= spec.ppf_nodes.front()
            && df <= spec.ppf_nodes.back();
        PpfInterpolation interpolation;
        if (use_cache) {
            interpolation = make_ppf_interpolation(spec.ppf_nodes, df);
        }
        for (int i = 0; i < d; ++i) {
            if (use_cache) {
                x[static_cast<std::size_t>(i)] = interpolate_ppf_value(
                    spec, interpolation, row_index, i, nullptr);
            } else {
                student_quantile_for_emission(
                    spec, row[i], df,
                    x[static_cast<std::size_t>(i)], nullptr);
            }
        }

        double quad = 0.0;
        for (int i = 0; i < d; ++i) {
            double value = 0.0;
            for (int j = 0; j <= i; ++j) {
                value += spec.l_inv[
                    static_cast<std::size_t>(i) * dim_size
                    + static_cast<std::size_t>(j)]
                    * x[static_cast<std::size_t>(j)];
            }
            whitened[static_cast<std::size_t>(i)] = value;
            quad += value * value;
        }
        for (int i = 0; i < d; ++i) {
            double value = 0.0;
            for (int j = i; j < d; ++j) {
                value += spec.l_inv[
                    static_cast<std::size_t>(j) * dim_size
                    + static_cast<std::size_t>(i)]
                    * whitened[static_cast<std::size_t>(j)];
            }
            precision_x[static_cast<std::size_t>(i)] = value;
        }

        const double shape_weight =
            (df + static_cast<double>(d)) / (df + quad);
        double directional_score = 0.0;
        std::size_t corr_index = 0;
        for (int i = 1; i < d; ++i) {
            for (int j = 0; j < i; ++j) {
                const double entry_score =
                    -precision[
                        static_cast<std::size_t>(i) * dim_size
                        + static_cast<std::size_t>(j)]
                    + shape_weight
                        * precision_x[static_cast<std::size_t>(i)]
                        * precision_x[static_cast<std::size_t>(j)];
                if (direction == nullptr) {
                    scores[grid_index * n_corr + corr_index] = entry_score;
                } else {
                    directional_score += (*direction)[corr_index]
                        * entry_score;
                }
                ++corr_index;
            }
        }
        if (direction != nullptr) {
            scores[grid_index] = directional_score;
        }
    }
    return true;
}

}  // namespace

bool student_log_pdf_from_quantiles(
    const double* quantiles,
    const double* quantile_derivatives,
    std::size_t dimension,
    double df,
    double logdet,
    double quadratic_form,
    double quadratic_form_derivative,
    double& log_pdf,
    double* dlog_ddf) {

    const bool compute_derivative = dlog_ddf != nullptr;
    if (quantiles == nullptr
        || dimension < 2
        || !std::isfinite(df)
        || df <= 2.0
        || !std::isfinite(logdet)
        || !std::isfinite(quadratic_form)
        || quadratic_form < 0.0
        || (compute_derivative
            && (quantile_derivatives == nullptr
                || !std::isfinite(quadratic_form_derivative)))) {
        return false;
    }

    double marginal_log = 0.0;
    double marginal_dlog_ddf = 0.0;
    double marginal_constant = 0.0;
    double marginal_constant_derivative = 0.0;
    if (!student_marginal_log_pdf_constants(
            df,
            marginal_constant,
            marginal_constant_derivative)) {
        return false;
    }
    for (std::size_t index = 0; index < dimension; ++index) {
        const double quantile = quantiles[index];
        if (!std::isfinite(quantile)
            || (compute_derivative
                && !std::isfinite(quantile_derivatives[index]))) {
            return false;
        }
        double marginal_value = 0.0;
        double marginal_derivative = 0.0;
        if (!student_marginal_log_pdf_from_quantile(
                quantile,
                compute_derivative ? quantile_derivatives[index] : 0.0,
                df,
                marginal_constant,
                marginal_constant_derivative,
                marginal_value,
                marginal_derivative)) {
            return false;
        }
        marginal_log += marginal_value;
        marginal_dlog_ddf += marginal_derivative;
    }

    return student_log_pdf_from_summaries(
        dimension,
        df,
        logdet,
        quadratic_form,
        quadratic_form_derivative,
        marginal_log,
        marginal_dlog_ddf,
        log_pdf,
        dlog_ddf);
}

bool student_marginal_log_pdf_from_quantile(
    double quantile,
    double quantile_derivative,
    double df,
    double marginal_constant,
    double marginal_constant_derivative,
    double& log_pdf,
    double& dlog_ddf) {

    if (!std::isfinite(quantile)
        || !std::isfinite(quantile_derivative)
        || !std::isfinite(df)
        || df <= 2.0
        || !std::isfinite(marginal_constant)
        || !std::isfinite(marginal_constant_derivative)) {
        return false;
    }
    const double quantile_squared = quantile * quantile;
    const double marginal_shape =
        std::log1p(quantile_squared / df);
    log_pdf = marginal_constant
        - 0.5 * (df + 1.0) * marginal_shape;
    const double quantile_squared_derivative =
        2.0 * quantile * quantile_derivative;
    const double marginal_shape_derivative =
        (df * quantile_squared_derivative - quantile_squared)
        / (df * (df + quantile_squared));
    dlog_ddf =
        marginal_constant_derivative
        - 0.5 * marginal_shape
        - 0.5 * (df + 1.0) * marginal_shape_derivative;
    return std::isfinite(log_pdf) && std::isfinite(dlog_ddf);
}

bool student_marginal_log_pdf_constants(
    double df,
    double& marginal_constant,
    double& marginal_constant_derivative) {

    if (!std::isfinite(df) || df <= 2.0) {
        return false;
    }
    marginal_constant =
        thread_safe_lgamma(0.5 * (df + 1.0))
        - thread_safe_lgamma(0.5 * df)
        - 0.5 * std::log(df * kPi);
    marginal_constant_derivative =
        0.5 * digamma_positive(0.5 * (df + 1.0))
        - 0.5 * digamma_positive(0.5 * df)
        - 0.5 / df;
    return std::isfinite(marginal_constant)
        && std::isfinite(marginal_constant_derivative);
}

bool student_log_pdf_from_summaries(
    std::size_t dimension,
    double df,
    double logdet,
    double quadratic_form,
    double quadratic_form_derivative,
    double marginal_log_pdf,
    double marginal_dlog_ddf,
    double& log_pdf,
    double* dlog_ddf) {

    const bool compute_derivative = dlog_ddf != nullptr;
    if (dimension < 2
        || !std::isfinite(df)
        || df <= 2.0
        || !std::isfinite(logdet)
        || !std::isfinite(quadratic_form)
        || quadratic_form < 0.0
        || !std::isfinite(marginal_log_pdf)
        || (compute_derivative
            && (!std::isfinite(quadratic_form_derivative)
                || !std::isfinite(marginal_dlog_ddf)))) {
        return false;
    }
    const double dimension_value = static_cast<double>(dimension);
    const double joint_shape = std::log1p(quadratic_form / df);
    const double joint_log =
        thread_safe_lgamma(0.5 * (df + dimension_value))
        - thread_safe_lgamma(0.5 * df)
        - 0.5 * dimension_value * std::log(df * kPi)
        - 0.5 * logdet
        - 0.5 * (df + dimension_value) * joint_shape;
    log_pdf = joint_log - marginal_log_pdf;
    if (compute_derivative) {
        const double joint_const_derivative =
            0.5 * digamma_positive(0.5 * (df + dimension_value))
            - 0.5 * digamma_positive(0.5 * df)
            - 0.5 * dimension_value / df;
        const double joint_shape_derivative =
            (df * quadratic_form_derivative - quadratic_form)
            / (df * (df + quadratic_form));
        const double joint_dlog_ddf =
            joint_const_derivative
            - 0.5 * joint_shape
            - 0.5 * (df + dimension_value) * joint_shape_derivative;
        *dlog_ddf = joint_dlog_ddf - marginal_dlog_ddf;
    }
    return std::isfinite(log_pdf)
        && (!compute_derivative || std::isfinite(*dlog_ddf));
}

double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index) {

    StudentWorkspace workspace;
    return student_log_pdf(
        spec, row, df, row_index, workspace);
}

double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace) {

    return student_log_pdf_with_work(
        spec, row, df, row_index, workspace, nullptr);
}

bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf) {

    StudentWorkspace workspace;
    return student_log_pdf_and_dlog_ddf(
        spec, row, df, row_index, log_pdf, dlog_ddf, workspace);
}

bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf,
    StudentWorkspace& workspace) {

    log_pdf = student_log_pdf_with_work(
        spec, row, df, row_index, workspace, &dlog_ddf);
    if (!std::isfinite(log_pdf)) {
        return false;
    }
    return std::isfinite(dlog_ddf);
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

    const bool use_cache =
        column >= 0
        && column < spec.dim
        && student_ppf_cache_available(spec, row_index)
        && df >= spec.ppf_nodes.front()
        && df <= spec.ppf_nodes.back();
    if (use_cache) {
        const PpfInterpolation interpolation =
            make_ppf_interpolation(spec.ppf_nodes, df);
        return interpolate_ppf_value(
            spec, interpolation, row_index, column, nullptr);
    }

    double value = std::numeric_limits<double>::quiet_NaN();
    student_quantile_for_emission(spec, p, df, value, nullptr);
    return value;
}

double student_cdf_value(double value, double df) {
    if (!std::isfinite(value) || !std::isfinite(df) || df <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double cdf = value >= 0.0
        ? 1.0 - student_survival_positive(value, df)
        : student_survival_positive(-value, df);
    return std::clamp(cdf, 0.0, 1.0);
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

bool student_precision_matrix(
    const scar::CopulaSpec& spec,
    std::vector<double>& precision) {

    const int d = spec.dim;
    std::size_t matrix_elements = 0;
    if (d < 2
        || !valid_student_dimension(d, matrix_elements)
        || spec.l_inv.size() != matrix_elements) {
        return false;
    }
    precision.assign(matrix_elements, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = 0.0;
            for (int k = i; k < d; ++k) {
                value +=
                    spec.l_inv[
                        static_cast<std::size_t>(k)
                            * static_cast<std::size_t>(d)
                        + static_cast<std::size_t>(i)]
                    * spec.l_inv[
                        static_cast<std::size_t>(k)
                            * static_cast<std::size_t>(d)
                        + static_cast<std::size_t>(j)];
            }
            precision[
                static_cast<std::size_t>(i) * static_cast<std::size_t>(d)
                + static_cast<std::size_t>(j)] = value;
            precision[
                static_cast<std::size_t>(j) * static_cast<std::size_t>(d)
                + static_cast<std::size_t>(i)] = value;
        }
    }
    return true;
}

bool student_corr_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    double* scores) {

    return student_corr_score_row_impl(
        spec,
        row,
        row_index,
        df_grid,
        precision,
        nullptr,
        scores);
}

bool student_corr_directional_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    const std::vector<double>& direction,
    double* scores) {

    return student_corr_score_row_impl(
        spec,
        row,
        row_index,
        df_grid,
        precision,
        &direction,
        scores);
}

void student_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace::Diagnostics* diagnostics) {

    StudentWorkspace workspace;
    workspace.reserve_x(static_cast<std::size_t>(spec.dim));
    workspace.reserve_dx_ddf(static_cast<std::size_t>(spec.dim));
    student_fill_row_with_workspace(
        spec,
        row,
        row_index,
        df_grid,
        dpsi_grid,
        fi_row,
        dfi_dx_row,
        workspace);
    if (diagnostics != nullptr) {
        *diagnostics = workspace.diagnostics;
    }
}

void student_fill_row_with_workspace(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace& workspace) {

    for (std::size_t j = 0; j < df_grid.size(); ++j) {
        const double df = df_grid[j];
        double dlog = std::numeric_limits<double>::quiet_NaN();
        const double log_pdf = student_log_pdf_with_work(
            spec,
            row,
            df,
            row_index,
            workspace,
            dfi_dx_row == nullptr ? nullptr : &dlog);
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            dfi_dx_row[j] = pdf * dlog * dpsi_grid[j];
        }
    }
}

void student_fill_row_from_x_grid(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& x_grid,
    double* fi_row) {

    StudentWorkspace workspace;
    workspace.reserve_x(static_cast<std::size_t>(spec.dim));
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        const double df = copula_transform(spec, x_grid[j]);
        fi_row[j] = std::exp(student_log_pdf_with_work(
            spec, row, df, row_index, workspace, nullptr));
    }
}

bool student_fill_grid_bivariate(
    const scar::CopulaSpec& spec,
    std::int64_t n_obs,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi,
    double* dfi_dx,
    int n_threads) {

    if (spec.dim != 2
        || n_obs <= 0
        || spec.ppf_n_obs != n_obs
        || !student_ppf_cache_available(spec, 0)
        || df_grid.size() != dpsi_grid.size()
        || spec.l_inv.size() != 4) {
        return false;
    }
    if (!std::all_of(
            df_grid.begin(),
            df_grid.end(),
            [&spec](double df) {
                return std::isfinite(df)
                    && df > 2.0
                    && df >= spec.ppf_nodes.front()
                    && df <= spec.ppf_nodes.back();
            })) {
        // The optimized bivariate kernel assumes every quantile comes from
        // the interpolation table. Fall back to the general row evaluator,
        // which uses exact Student quantiles outside the cache range.
        return false;
    }

    const double l11 = spec.l_inv[3];
    if (!std::isfinite(l11) || std::abs(l11) < 1e-15) {
        return false;
    }
    const double rho = -spec.l_inv[2] / l11;
    const double one_minus_rho2 = 1.0 - rho * rho;
    if (!std::isfinite(rho) || one_minus_rho2 <= 0.0) {
        return false;
    }

    const std::size_t K = df_grid.size();
    constexpr std::int64_t min_rows_per_block = 8;
    parallel_for_blocks(
        0,
        n_obs,
        min_rows_per_block,
        n_threads,
        [&](std::int64_t begin, std::int64_t end, std::size_t) {
            for (std::size_t j = 0; j < K; ++j) {
                const double df = df_grid[j];
                const PpfInterpolation interpolation =
                    make_ppf_interpolation(spec.ppf_nodes, df);

        const double half_df = 0.5 * df;
        const double log_df_pi = std::log(df * kPi);
        const double joint_const =
            thread_safe_lgamma(half_df + 1.0)
            - thread_safe_lgamma(half_df)
            - log_df_pi
            - 0.5 * spec.log_det;
        const double marginal_const =
            thread_safe_lgamma(half_df + 0.5)
            - thread_safe_lgamma(half_df)
            - 0.5 * log_df_pi;
        const double copula_const = joint_const - 2.0 * marginal_const;

        const double digamma_half_df = digamma_positive(half_df);
        const double joint_const_derivative =
            0.5 * digamma_positive(half_df + 1.0)
            - 0.5 * digamma_half_df
            - 1.0 / df;
        const double marginal_const_derivative =
            0.5 * digamma_positive(half_df + 0.5)
            - 0.5 * digamma_half_df
            - 0.5 / df;
        const double copula_const_derivative =
            joint_const_derivative - 2.0 * marginal_const_derivative;

        for (std::int64_t t = begin; t < end; ++t) {
            double x1 = 0.0;
            double x2 = 0.0;
            double dx1 = 0.0;
            double dx2 = 0.0;
            interpolate_bivariate_ppf(
                spec,
                interpolation,
                static_cast<std::size_t>(t),
                x1,
                x2,
                dx1,
                dx2);

            const double x1_sq = x1 * x1;
            const double x2_sq = x2 * x2;
            const double cross = x1 * x2;
            const double quad =
                (x1_sq - 2.0 * rho * cross + x2_sq)
                / one_minus_rho2;
            const double dquad =
                2.0 * (
                    x1 * dx1
                    - rho * (dx1 * x2 + x1 * dx2)
                    + x2 * dx2
                ) / one_minus_rho2;
            const double joint_shape = std::log1p(quad / df);
            const double marginal_shape1 = std::log1p(x1_sq / df);
            const double marginal_shape2 = std::log1p(x2_sq / df);
            const double log_pdf =
                copula_const
                - 0.5 * (df + 2.0) * joint_shape
                + 0.5 * (df + 1.0)
                    * (marginal_shape1 + marginal_shape2);
            const double pdf = std::exp(log_pdf);

            const double joint_shape_derivative =
                (df * dquad - quad) / (df * (df + quad));
            const double marginal_shape1_derivative =
                (df * 2.0 * x1 * dx1 - x1_sq)
                / (df * (df + x1_sq));
            const double marginal_shape2_derivative =
                (df * 2.0 * x2 * dx2 - x2_sq)
                / (df * (df + x2_sq));
            const double dlog_ddf =
                copula_const_derivative
                - 0.5 * joint_shape
                - 0.5 * (df + 2.0) * joint_shape_derivative
                + 0.5 * (marginal_shape1 + marginal_shape2)
                + 0.5 * (df + 1.0)
                    * (
                        marginal_shape1_derivative
                        + marginal_shape2_derivative);

            const std::size_t output =
                static_cast<std::size_t>(t) * K + j;
            fi[output] = pdf;
            dfi_dx[output] = pdf * dlog_ddf * dpsi_grid[j];
            }
            }
        });
    return true;
}

}  // namespace scar_internal
