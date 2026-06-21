#include "scar/copula.hpp"

#include "scar/detail/internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

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
    return scar_internal::is_valid_rotation(static_cast<int>(spec.rotation))
        && scar_internal::copula_is_supported(spec);
}

bool supports_transform(const CopulaSpec& spec) {
    if (!scar_internal::is_valid_rotation(
            static_cast<int>(spec.rotation))) {
        return false;
    }
    if (spec.family == CopulaFamily::EquicorrGaussian) {
        return spec.dim >= 2
            && spec.rotation == Rotation::R0
            && spec.transform == Transform::GaussianTanh;
    }
    return spec.family == CopulaFamily::Student
        || scar_internal::copula_is_supported(spec);
}

std::vector<double> copula_transform(
    const CopulaSpec& spec,
    const std::vector<double>& x) {

    std::vector<double> out(x.size(), 0.0);
    if (!supports_transform(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(x.begin(), x.end(), out.begin(), [&](double value) {
        return scar_internal::copula_transform(spec, value);
    });
    return out;
}

std::vector<double> copula_inverse_transform(
    const CopulaSpec& spec,
    const std::vector<double>& r) {

    std::vector<double> out(r.size(), 0.0);
    if (!supports_transform(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(r.begin(), r.end(), out.begin(), [&](double value) {
        return scar_internal::copula_inverse_transform(spec, value);
    });
    return out;
}

std::vector<double> copula_dtransform(
    const CopulaSpec& spec,
    const std::vector<double>& x) {

    std::vector<double> out(x.size(), 0.0);
    if (!supports_transform(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(x.begin(), x.end(), out.begin(), [&](double value) {
        return scar_internal::copula_dtransform(spec, value);
    });
    return out;
}

std::vector<double> copula_tau_to_param(
    const CopulaSpec& spec,
    const std::vector<double>& tau) {

    std::vector<double> out(tau.size(), 0.0);
    if (!is_supported(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(tau.begin(), tau.end(), out.begin(), [&](double value) {
        return scar_internal::copula_tau_to_param(spec, value);
    });
    return out;
}

std::vector<double> copula_param_to_tau(
    const CopulaSpec& spec,
    const std::vector<double>& r) {

    std::vector<double> out(r.size(), 0.0);
    if (!is_supported(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(r.begin(), r.end(), out.begin(), [&](double value) {
        return scar_internal::copula_param_to_tau(spec, value);
    });
    return out;
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

std::vector<double> copula_dlog_pdf_dr(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(u);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    if (!is_supported(spec) || (r.size() != 1 && r.size() != u.size())) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, i, 0), row_value(u, i, 1), v1, v2);
        out[static_cast<std::size_t>(i)] =
            scar_internal::copula_dlog_pdf_dr_unrotated(
                spec, v1, v2, r_value(r, i));
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
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), x_grid.size(), value_count)) {
        return out;
    }
    out.values.assign(value_count, 0.0);

    if (!is_supported(spec)) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        const std::size_t row =
            static_cast<std::size_t>(t) * x_grid.size();
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        for (std::int64_t j = 0; j < out.n_grid; ++j) {
            const double r = scar_internal::copula_transform(
                spec, x_grid[static_cast<std::size_t>(j)]);
            const double log_c = scar_internal::copula_log_pdf_unrotated(
                spec, v1, v2, r);
            out.values[row + static_cast<std::size_t>(j)] =
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
        const std::size_t row =
            static_cast<std::size_t>(t)
            * static_cast<std::size_t>(out.pdf.n_grid);
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        for (std::int64_t j = 0; j < out.pdf.n_grid; ++j) {
            const double x = x_grid[static_cast<std::size_t>(j)];
            const double r = scar_internal::copula_transform(spec, x);
            const double dpsi = scar_internal::copula_dtransform(spec, x);
            const std::size_t idx = row + static_cast<std::size_t>(j);
            out.d_pdf_dx.values[idx] =
                out.pdf.values[idx]
                * scar_internal::copula_dlog_pdf_dr_unrotated(spec, v1, v2, r)
                * dpsi;
        }
    }
    return out;
}

GridValues copula_pdf_parameter_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r_grid) {

    GridValues out;
    out.n_obs = checked_size(u);
    out.n_grid = static_cast<std::int64_t>(r_grid.size());
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), r_grid.size(), value_count)) {
        return out;
    }
    out.values.assign(value_count, 0.0);
    if (!is_supported(spec)) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(
            spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        const std::size_t base =
            static_cast<std::size_t>(t) * r_grid.size();
        for (std::size_t j = 0; j < r_grid.size(); ++j) {
            out.values[base + j] = std::exp(
                scar_internal::copula_log_pdf_unrotated(
                    spec, v1, v2, r_grid[j]));
        }
    }
    return out;
}

GridValues copula_h_parameter_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r_grid) {

    GridValues out;
    out.n_obs = checked_size(u);
    out.n_grid = static_cast<std::int64_t>(r_grid.size());
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), r_grid.size(), value_count)) {
        return out;
    }
    out.values.assign(value_count, 0.0);
    if (!is_supported(spec)) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        const std::size_t base =
            static_cast<std::size_t>(t) * r_grid.size();
        for (std::size_t j = 0; j < r_grid.size(); ++j) {
            out.values[base + j] = scar_internal::copula_h_rotated(
                spec,
                row_value(u, t, 1),
                row_value(u, t, 0),
                r_grid[j]);
        }
    }
    return out;
}

}  // namespace scar
