#include "scar/copula.hpp"

#include "scar_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

bool is_valid_rotation(Rotation rotation) {
    return rotation == Rotation::R0
        || rotation == Rotation::R90
        || rotation == Rotation::R180
        || rotation == Rotation::R270;
}

std::int64_t checked_size(const Observations& u) {
    return static_cast<std::int64_t>(u.size());
}

double row_value(const Observations& u, std::int64_t row, int col) {
    return u[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
}

double r_value(const std::vector<double>& r, std::int64_t row) {
    if (r.size() == 1) {
        return r[0];
    }
    return r[static_cast<std::size_t>(row)];
}

void apply_rotation(
    const CopulaSpec& spec,
    double u1,
    double u2,
    double& v1,
    double& v2) {

    scar_internal::apply_rotation(
        u1, u2, static_cast<int>(spec.rotation), v1, v2);
}

}  // namespace

bool is_supported(const CopulaSpec& spec) {
    return is_valid_rotation(spec.rotation)
        && scar_internal::copula_is_supported(spec);
}

std::vector<double> copula_log_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(u);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    if (!is_supported(spec) || (r.size() != 1 && r.size() != u.size())) {
        std::fill(out.begin(), out.end(), -std::numeric_limits<double>::infinity());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, i, 0), row_value(u, i, 1), v1, v2);
        out[static_cast<std::size_t>(i)] =
            scar_internal::copula_log_pdf_unrotated(spec, v1, v2, r_value(r, i));
    }
    return out;
}

std::vector<double> copula_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    std::vector<double> out = copula_log_pdf(spec, u, r);
    for (double& value : out) {
        value = std::exp(value);
    }
    return out;
}

std::vector<double> copula_h(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(u);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    if (!is_supported(spec) || (r.size() != 1 && r.size() != u.size())) {
        std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = scar_internal::copula_h_rotated(
            spec, row_value(u, i, 0), row_value(u, i, 1), r_value(r, i));
    }
    return out;
}

std::pair<std::vector<double>, std::vector<double>> copula_h_pair(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    return {
        copula_h(spec, u, r),
        [&]() {
            Observations reversed = u;
            for (std::vector<double>& row : reversed) {
                std::swap(row[0], row[1]);
            }
            return copula_h(spec, reversed, r);
        }(),
    };
}

std::vector<double> copula_h_inverse(
    const CopulaSpec& spec,
    const Observations& q_given,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(q_given);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    if (!is_supported(spec) || (r.size() != 1 && r.size() != q_given.size())) {
        std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = scar_internal::copula_h_inverse_rotated(
            spec, row_value(q_given, i, 0), row_value(q_given, i, 1), r_value(r, i));
    }
    return out;
}

GridValues copula_pdf_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid) {

    GridValues out;
    out.n_obs = checked_size(u);
    out.n_grid = static_cast<std::int64_t>(x_grid.size());
    out.values.assign(
        static_cast<std::size_t>(out.n_obs * out.n_grid), 0.0);

    if (!is_supported(spec)) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        for (std::int64_t j = 0; j < out.n_grid; ++j) {
            const double r = scar_internal::copula_transform(
                spec, x_grid[static_cast<std::size_t>(j)]);
            const double log_c = scar_internal::copula_log_pdf_unrotated(
                spec, v1, v2, r);
            out.values[static_cast<std::size_t>(t * out.n_grid + j)] =
                std::exp(log_c);
        }
    }
    return out;
}

GridValuesWithGrad copula_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid) {

    GridValuesWithGrad out;
    out.pdf = copula_pdf_grid(spec, u, x_grid);
    out.d_pdf_dx.n_obs = out.pdf.n_obs;
    out.d_pdf_dx.n_grid = out.pdf.n_grid;
    out.d_pdf_dx.values.assign(out.pdf.values.size(), 0.0);

    if (!is_supported(spec)) {
        std::fill(out.d_pdf_dx.values.begin(), out.d_pdf_dx.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.pdf.n_obs; ++t) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        for (std::int64_t j = 0; j < out.pdf.n_grid; ++j) {
            const double x = x_grid[static_cast<std::size_t>(j)];
            const double r = scar_internal::copula_transform(spec, x);
            const double dpsi = scar_internal::copula_dtransform(spec, x);
            const std::size_t idx = static_cast<std::size_t>(t * out.pdf.n_grid + j);
            out.d_pdf_dx.values[idx] =
                out.pdf.values[idx]
                * scar_internal::copula_dlog_pdf_dr_unrotated(spec, v1, v2, r)
                * dpsi;
        }
    }
    return out;
}

}  // namespace scar

