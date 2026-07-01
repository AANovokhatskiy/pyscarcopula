#include "scar/detail/copula.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace scar_internal {
namespace {

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

double regularized_beta(double x, double a, double b) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    const double bt = std::exp(
        std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
        + a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x) / a;
    }
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double student_pdf(double t, double df) {
    const double log_pdf =
        std::lgamma(0.5 * (df + 1.0))
        - std::lgamma(0.5 * df)
        - 0.5 * std::log(df * kPi)
        - 0.5 * (df + 1.0) * std::log1p((t * t) / df);
    return std::exp(log_pdf);
}

double student_survival_positive(double t, double df) {
    const double x = df / (df + t * t);
    return 0.5 * regularized_beta(x, 0.5 * df, 0.5);
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

struct StudentWorkspace {
    std::vector<double> x;
    std::vector<double> dx_ddf;
};

double student_log_pdf_with_work(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace,
    double* dlog_ddf) {

    const int d = spec.dim;
    std::size_t matrix_elements = 0;
    if (d < 2
        || !valid_student_dimension(d, matrix_elements)
        || spec.l_inv.size() != matrix_elements
        || !std::isfinite(spec.log_det)
        || !std::isfinite(df)
        || df <= 2.0) {
        return -std::numeric_limits<double>::infinity();
    }

    workspace.x.resize(static_cast<std::size_t>(d));
    const bool use_cache =
        student_ppf_cache_available(spec, row_index)
        && df >= spec.ppf_nodes.front()
        && df <= spec.ppf_nodes.back();
    const bool compute_derivative = dlog_ddf != nullptr && use_cache;
    if (compute_derivative) {
        workspace.dx_ddf.resize(static_cast<std::size_t>(d));
    } else {
        workspace.dx_ddf.clear();
    }
    PpfInterpolation interpolation;
    if (use_cache) {
        interpolation = make_ppf_interpolation(spec.ppf_nodes, df);
    }
    for (int i = 0; i < d; ++i) {
        workspace.x[static_cast<std::size_t>(i)] = use_cache
            ? interpolate_ppf_value(
                spec,
                interpolation,
                row_index,
                i,
                compute_derivative
                    ? &workspace.dx_ddf[static_cast<std::size_t>(i)]
                    : nullptr)
            : student_quantile(row[i], df);
    }

    double quad = 0.0;
    double dquad_ddf = 0.0;
    double marginal_log = 0.0;
    double marginal_dlog_ddf = 0.0;
    const double marginal_const =
        std::lgamma(0.5 * (df + 1.0))
        - std::lgamma(0.5 * df)
        - 0.5 * std::log(df * kPi);
    const double marginal_const_derivative =
        0.5 * digamma_positive(0.5 * (df + 1.0))
        - 0.5 * digamma_positive(0.5 * df)
        - 0.5 / df;
    for (int i = 0; i < d; ++i) {
        const double xi = workspace.x[static_cast<std::size_t>(i)];
        double yi = 0.0;
        double dyi_ddf = 0.0;
        const std::size_t row_offset =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
        for (int j = 0; j <= i; ++j) {
            yi += spec.l_inv[row_offset + static_cast<std::size_t>(j)]
                * workspace.x[static_cast<std::size_t>(j)];
            if (compute_derivative) {
                dyi_ddf +=
                    spec.l_inv[row_offset + static_cast<std::size_t>(j)]
                    * workspace.dx_ddf[static_cast<std::size_t>(j)];
            }
        }
        quad += yi * yi;
        if (compute_derivative) {
            dquad_ddf += 2.0 * yi * dyi_ddf;
        }
        const double xi2 = xi * xi;
        const double marginal_shape = std::log1p(xi2 / df);
        marginal_log += marginal_const
            - 0.5 * (df + 1.0) * marginal_shape;
        if (compute_derivative) {
            const double dxi2_ddf =
                2.0 * xi * workspace.dx_ddf[static_cast<std::size_t>(i)];
            const double dshape_ddf =
                (df * dxi2_ddf - xi2) / (df * (df + xi2));
            marginal_dlog_ddf +=
                marginal_const_derivative
                - 0.5 * marginal_shape
                - 0.5 * (df + 1.0) * dshape_ddf;
        }
    }

    const double joint_shape = std::log1p(quad / df);
    const double joint_log =
        std::lgamma(0.5 * (df + static_cast<double>(d)))
        - std::lgamma(0.5 * df)
        - 0.5 * static_cast<double>(d) * std::log(df * kPi)
        - 0.5 * spec.log_det
        - 0.5 * (df + static_cast<double>(d)) * joint_shape;
    if (dlog_ddf != nullptr) {
        if (!compute_derivative) {
            *dlog_ddf = std::numeric_limits<double>::quiet_NaN();
        } else {
            const double joint_const_derivative =
                0.5 * digamma_positive(
                    0.5 * (df + static_cast<double>(d)))
                - 0.5 * digamma_positive(0.5 * df)
                - 0.5 * static_cast<double>(d) / df;
            const double dshape_ddf =
                (df * dquad_ddf - quad) / (df * (df + quad));
            const double joint_dlog_ddf =
                joint_const_derivative
                - 0.5 * joint_shape
                - 0.5 * (df + static_cast<double>(d)) * dshape_ddf;
            *dlog_ddf = joint_dlog_ddf - marginal_dlog_ddf;
        }
    }
    return joint_log - marginal_log;
}

double student_log_pdf_impl(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index) {

    StudentWorkspace workspace;
    return student_log_pdf_with_work(
        spec, row, df, row_index, workspace, nullptr);
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
            x[static_cast<std::size_t>(i)] = use_cache
                ? interpolate_ppf_value(
                    spec, interpolation, row_index, i, nullptr)
                : student_quantile(row[i], df);
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

double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index) {

    return student_log_pdf_impl(spec, row, df, row_index);
}

bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf) {

    StudentWorkspace workspace;
    log_pdf = student_log_pdf_with_work(
        spec, row, df, row_index, workspace, &dlog_ddf);
    if (!std::isfinite(log_pdf)) {
        return false;
    }
    if (!std::isfinite(dlog_ddf)) {
        const double step = std::max(1e-4, 1e-5 * std::abs(df));
        const double lo = std::max(2.0 + 1e-6, df - step);
        const double hi = df + step;
        const double log_hi = student_log_pdf_with_work(
            spec, row, hi, row_index, workspace, nullptr);
        const double log_lo = student_log_pdf_with_work(
            spec, row, lo, row_index, workspace, nullptr);
        dlog_ddf = (log_hi - log_lo) / (hi - lo);
    }
    return std::isfinite(dlog_ddf);
}

double student_quantile_value(double p, double df) {
    return student_quantile(p, df);
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
    double* dfi_dx_row) {

    StudentWorkspace workspace;
    workspace.x.reserve(static_cast<std::size_t>(spec.dim));
    workspace.dx_ddf.reserve(static_cast<std::size_t>(spec.dim));
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
            if (!std::isfinite(dlog)) {
                const double step = std::max(1e-4, 1e-5 * std::abs(df));
                const double lo = std::max(2.0 + 1e-6, df - step);
                const double hi = df + step;
                dlog =
                    (student_log_pdf_with_work(
                        spec, row, hi, row_index, workspace, nullptr)
                     - student_log_pdf_with_work(
                        spec, row, lo, row_index, workspace, nullptr))
                    / (hi - lo);
            }
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
    workspace.x.reserve(static_cast<std::size_t>(spec.dim));
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
    double* dfi_dx) {

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
    for (std::size_t j = 0; j < K; ++j) {
        const double df = df_grid[j];
        if (!std::isfinite(df) || df <= 2.0) {
            return false;
        }
        const PpfInterpolation interpolation =
            make_ppf_interpolation(spec.ppf_nodes, df);

        const double half_df = 0.5 * df;
        const double log_df_pi = std::log(df * kPi);
        const double joint_const =
            std::lgamma(half_df + 1.0)
            - std::lgamma(half_df)
            - log_df_pi
            - 0.5 * spec.log_det;
        const double marginal_const =
            std::lgamma(half_df + 0.5)
            - std::lgamma(half_df)
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

        for (std::int64_t t = 0; t < n_obs; ++t) {
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
    return true;
}

}  // namespace scar_internal
