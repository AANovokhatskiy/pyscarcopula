#include "scar/detail/internal.hpp"

namespace scar_internal {

namespace {

void clayton_fill_row(
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_abs_logv1 = std::log(-log_v1);
    const double log_abs_logv2 = std::log(-log_v2);

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double a = -r * log_v1;
        const double b = -r * log_v2;
        const double log_max = std::max(a, b);
        const double log_min = std::min(a, b);
        const double correction = std::exp(log_min - log_max) - std::exp(-log_max);
        const double log_s = log_max + std::log1p(correction);
        const double log_pdf =
            std::log1p(r)
            + (-r - 1.0) * log_v1
            + (-r - 1.0) * log_v2
            + (-2.0 - 1.0 / r) * log_s;
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            const double p = log_abs_logv1 + a;
            const double q = log_abs_logv2 + b;
            const double pq_max = std::max(p, q);
            const double pq_min = std::min(p, q);
            const double log_ds = pq_max + std::log1p(std::exp(pq_min - pq_max));
            const double ds_over_s = std::exp(log_ds - log_s);
            const double dlog_dr =
                1.0 / (1.0 + r)
                - log_v1
                - log_v2
                + log_s / (r * r)
                + (-2.0 - 1.0 / r) * ds_over_s;
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

void gumbel_fill_row(
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_p1 = std::log(std::max(-log_v1, kPdfEps));
    const double log_p2 = std::log(std::max(-log_v2, kPdfEps));
    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double delta = r * (log_min - log_max);
        const double exp_delta = std::exp(delta);
        const double S = r * log_max + std::log1p(exp_delta);
        const double A = std::exp(S / r);
        const double log_pdf =
            (r - 1.0) * (log_p1 + log_p2)
            + (1.0 / r - 2.0) * S
            + std::log(r - 1.0 + A)
            - A
            - log_v1
            - log_v2;
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            const double sig = exp_delta / (1.0 + exp_delta);
            const double dS_dr = log_max + (log_min - log_max) * sig;
            const double dlogA_dr = (dS_dr * r - S) / (r * r);
            const double dA_dr = A * dlogA_dr;
            const double dlog_dr =
                (log_p1 + log_p2)
                - S / (r * r)
                + (1.0 / r - 2.0) * dS_dr
                + (1.0 + dA_dr) / (r - 1.0 + A)
                - dA_dr;
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

void frank_fill_row(
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double a = r * v1;
        const double b = r * v2;
        const double log_num = std::log(r) + log1mexp(r) - a - b;
        const double log_t1 = -a + log1mexp(b);
        const double log_t2 = -b + log1mexp(r - b);
        const double pdf = std::exp(log_num - 2.0 * logsumexp(log_t1, log_t2));
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            const double emr = std::exp(-r);
            const double emrv1 = std::exp(-r * v1);
            const double emrv2 = std::exp(-r * v2);
            const double emr1v2 = std::exp(-r * (1.0 - v2));
            const double A = emrv1 * (1.0 - emrv2);
            const double B = emrv2 * (1.0 - emr1v2);
            const double D = std::max(A + B, kPdfEps);
            const double dA = emrv1 * (-v1 * (1.0 - emrv2) + v2 * emrv2);
            const double dB = emrv2 * (-v2 * (1.0 - emr1v2) + (1.0 - v2) * emr1v2);
            const double dlog_dr =
                1.0 / r
                + emr / (1.0 - emr)
                - (v1 + v2)
                - 2.0 * (dA + dB) / D;
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

void joe_fill_row(
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::min(std::max(1.0 - v1, kPdfEps), 1.0 - kPdfEps);
    const double q2 = std::min(std::max(1.0 - v2, kPdfEps), 1.0 - kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double log_t1 = r * log_q1 + log1mexp(-r * log_q2);
        const double log_t2 = r * log_q2;
        const double log_B = logsumexp(log_t1, log_t2);
        const double B_for_log = std::exp(log_B);
        double log_rp = 0.0;
        if (B_for_log > r - 1.0) {
            log_rp = log_B + std::log1p((r - 1.0) / B_for_log);
        } else {
            log_rp = std::log(r - 1.0) + std::log1p(B_for_log / (r - 1.0));
        }
        const double log_pdf =
            (r - 1.0) * (log_q1 + log_q2)
            + log_rp
            - (2.0 - 1.0 / r) * log_B;
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            const double q1r = std::exp(r * log_q1);
            const double q2r = std::exp(r * log_q2);
            const double B = std::max(q1r + q2r - q1r * q2r, kPdfEps);
            const double dB =
                q1r * log_q1 * (1.0 - q2r)
                + q2r * log_q2 * (1.0 - q1r);
            const double dlog_dr =
                log_q1 + log_q2
                + (1.0 + dB) / (r - 1.0 + B)
                - std::log(B) / (r * r)
                - (2.0 - 1.0 / r) * dB / B;
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

void gaussian_fill_row(
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const double v1 = clip_pseudo_observation(u1);
    const double v2 = clip_pseudo_observation(u2);
    const double x1 = normal_quantile(v1);
    const double x2 = normal_quantile(v2);
    const double s1 = x1 * x1 + x2 * x2;
    const double s12 = x1 * x2;

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double r2 = r * r;
        const double omr2 = 1.0 - r2;
        const double log_pdf =
            -0.5 * std::log(omr2)
            - 0.5 * (r2 * s1 - 2.0 * r * s12) / omr2;
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            const double dlog_det = r / omr2;
            const double num =
                (2.0 * r * s1 - 2.0 * s12) * omr2
                + 2.0 * r * (r2 * s1 - 2.0 * r * s12);
            const double dquad = num / (omr2 * omr2);
            const double dlog_dr = dlog_det - 0.5 * dquad;
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

void equicorr_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        double dlog_dr = 0.0;
        const double log_pdf = equicorr_log_pdf(
            spec,
            row,
            r_grid[j],
            dfi_dx_row == nullptr ? nullptr : &dlog_dr);
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            dfi_dx_row[j] = pdf * dlog_dr * dpsi_grid[j];
        }
    }
}

}  // namespace

bool copula_is_supported(const scar::CopulaSpec& spec) {
    if (!is_valid_rotation(static_cast<int>(spec.rotation))) {
        return false;
    }
    if (!std::isfinite(spec.offset) || spec.offset < 0.0) {
        return false;
    }
    if (spec.family == scar::CopulaFamily::Independent) {
        return true;
    }
    if (spec.family == scar::CopulaFamily::Clayton
        || spec.family == scar::CopulaFamily::Gumbel
        || spec.family == scar::CopulaFamily::Joe) {
        return spec.transform == scar::Transform::Softplus
            || spec.transform == scar::Transform::XTanh;
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return spec.rotation == scar::Rotation::R0
            && (spec.transform == scar::Transform::Softplus
                || spec.transform == scar::Transform::XTanh);
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        return spec.rotation == scar::Rotation::R0
            && spec.transform == scar::Transform::GaussianTanh;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return spec.rotation == scar::Rotation::R0
            && spec.transform == scar::Transform::GaussianTanh
            && spec.dim >= 2;
    }
    if (spec.family == scar::CopulaFamily::Student) {
        std::size_t expected = 0;
        if (!valid_student_dimension(spec.dim, expected)) {
            return false;
        }
        const bool valid_values = std::all_of(
            spec.l_inv.begin(), spec.l_inv.end(), [](double value) {
                return std::isfinite(value);
            });
        bool lower_triangular =
            spec.dim >= 2 && spec.l_inv.size() == expected;
        if (lower_triangular) {
            for (int i = 0; i < spec.dim && lower_triangular; ++i) {
                for (int j = i + 1; j < spec.dim; ++j) {
                    const std::size_t index =
                        static_cast<std::size_t>(i)
                            * static_cast<std::size_t>(spec.dim)
                        + static_cast<std::size_t>(j);
                    if (std::abs(spec.l_inv[index]) > 1e-14) {
                        lower_triangular = false;
                        break;
                    }
                }
            }
        }
        return spec.rotation == scar::Rotation::R0
            && spec.transform == scar::Transform::Softplus
            && spec.offset >= 2.0
            && spec.dim >= 2
            && spec.l_inv.size() == expected
            && std::isfinite(spec.log_det)
            && valid_values
            && lower_triangular;
    }
    return false;
}

bool copula_is_supported_for_ou(const scar::CopulaSpec& spec) {
    return copula_is_supported(spec)
        && (spec.transform == scar::Transform::Softplus
            || spec.transform == scar::Transform::XTanh
            || spec.transform == scar::Transform::GaussianTanh);
}

double copula_log_pdf_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r) {

    if (spec.family == scar::CopulaFamily::Clayton) {
        return clayton_log_pdf_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        return gumbel_log_pdf_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return frank_log_pdf_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        return joe_log_pdf_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        return gaussian_log_pdf_unrotated(u1, u2, r);
    }
    return -std::numeric_limits<double>::infinity();
}

double copula_dlog_pdf_dr_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r) {

    if (spec.family == scar::CopulaFamily::Clayton) {
        return clayton_dlog_pdf_dr_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        return gumbel_dlog_pdf_dr_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return frank_dlog_pdf_dr_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        return joe_dlog_pdf_dr_unrotated(u1, u2, r);
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        return gaussian_dlog_pdf_dr_unrotated(u1, u2, r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_pdf_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x) {

    const double r = copula_transform(spec, x);
    return std::exp(copula_log_pdf_unrotated(spec, u1, u2, r));
}

void copula_pdf_and_grad_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx) {

    if (spec.family == scar::CopulaFamily::Clayton
        && spec.transform == scar::Transform::Softplus
        && spec.offset == kOffset) {
        clayton_pdf_and_grad_x_unrotated(u1, u2, x, pdf, d_pdf_dx);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gumbel
        && spec.transform == scar::Transform::Softplus) {
        gumbel_pdf_and_grad_x_unrotated(u1, u2, x, spec.offset, pdf, d_pdf_dx);
        return;
    }
    if (spec.family == scar::CopulaFamily::Frank
        && spec.transform == scar::Transform::Softplus) {
        frank_pdf_and_grad_x_unrotated(u1, u2, x, spec.offset, pdf, d_pdf_dx);
        return;
    }
    if (spec.family == scar::CopulaFamily::Joe
        && spec.transform == scar::Transform::Softplus) {
        joe_pdf_and_grad_x_unrotated(u1, u2, x, spec.offset, pdf, d_pdf_dx);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian
        && spec.transform == scar::Transform::GaussianTanh) {
        gaussian_pdf_and_grad_x_unrotated(u1, u2, x, pdf, d_pdf_dx);
        return;
    }

    const double r = copula_transform(spec, x);
    const double log_pdf = copula_log_pdf_unrotated(spec, u1, u2, r);
    pdf = std::exp(log_pdf);
    d_pdf_dx = pdf
        * copula_dlog_pdf_dr_unrotated(spec, u1, u2, r)
        * copula_dtransform(spec, x);
}

void copula_prepare_grid_transform(
    const scar::CopulaSpec& spec,
    const std::vector<double>& x_grid,
    std::vector<double>& r_grid,
    std::vector<double>& dpsi_grid) {

    r_grid.resize(x_grid.size());
    dpsi_grid.resize(x_grid.size());
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        r_grid[j] = copula_transform(spec, x_grid[j]);
        dpsi_grid[j] = copula_dtransform(spec, x_grid[j]);
    }
}

void copula_pdf_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    double* fi_row) {

    double v1 = 0.0;
    double v2 = 0.0;
    apply_rotation(u1, u2, static_cast<int>(spec.rotation), v1, v2);

    if (spec.family == scar::CopulaFamily::Independent) {
        std::fill(fi_row, fi_row + r_grid.size(), 1.0);
        return;
    }
    static const std::vector<double> no_dpsi;
    if (spec.family == scar::CopulaFamily::Clayton) {
        clayton_fill_row(v1, v2, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        gumbel_fill_row(v1, v2, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        frank_fill_row(v1, v2, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        joe_fill_row(v1, v2, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        gaussian_fill_row(v1, v2, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        fi_row[j] = std::exp(copula_log_pdf_unrotated(spec, v1, v2, r_grid[j]));
    }
}

void copula_pdf_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    double* fi_row) {

    const int stride =
        (spec.family == scar::CopulaFamily::Student
         || spec.family == scar::CopulaFamily::EquicorrGaussian)
        ? spec.dim
        : 2;
    const double* row =
        u + static_cast<std::size_t>(t) * static_cast<std::size_t>(stride);
    if (spec.family == scar::CopulaFamily::Student) {
        static const std::vector<double> no_dpsi;
        student_fill_row(spec, row, t, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        static const std::vector<double> no_dpsi;
        equicorr_fill_row(
            spec, row, r_grid, no_dpsi, fi_row, nullptr);
        return;
    }
    copula_pdf_row_precomputed(spec, row[0], row[1], r_grid, fi_row);
}

void copula_pdf_and_grad_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    double v1 = 0.0;
    double v2 = 0.0;
    apply_rotation(u1, u2, static_cast<int>(spec.rotation), v1, v2);

    if (spec.family == scar::CopulaFamily::Independent) {
        std::fill(fi_row, fi_row + r_grid.size(), 1.0);
        std::fill(dfi_dx_row, dfi_dx_row + r_grid.size(), 0.0);
        return;
    }
    if (spec.family == scar::CopulaFamily::Clayton) {
        clayton_fill_row(v1, v2, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        gumbel_fill_row(v1, v2, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        frank_fill_row(v1, v2, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        joe_fill_row(v1, v2, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        gaussian_fill_row(v1, v2, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }

    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double log_pdf = copula_log_pdf_unrotated(spec, v1, v2, r_grid[j]);
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        dfi_dx_row[j] = pdf
            * copula_dlog_pdf_dr_unrotated(spec, v1, v2, r_grid[j])
            * dpsi_grid[j];
    }
}

void copula_pdf_and_grad_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row) {

    const int stride =
        (spec.family == scar::CopulaFamily::Student
         || spec.family == scar::CopulaFamily::EquicorrGaussian)
        ? spec.dim
        : 2;
    const double* row =
        u + static_cast<std::size_t>(t) * static_cast<std::size_t>(stride);
    if (spec.family == scar::CopulaFamily::Student) {
        student_fill_row(
            spec, row, t, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        equicorr_fill_row(
            spec, row, r_grid, dpsi_grid, fi_row, dfi_dx_row);
        return;
    }
    copula_pdf_and_grad_row_precomputed(
        spec, row[0], row[1], r_grid, dpsi_grid, fi_row, dfi_dx_row);
}

void copula_pdf_and_grad_grid_precomputed(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t n_obs,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    std::vector<double>& fi,
    std::vector<double>& dfi_dx) {

    const std::size_t K = r_grid.size();
    std::size_t n_obs_size = 0;
    std::size_t elements = 0;
    if (!checked_nonnegative_size(n_obs, n_obs_size)
        || !checked_size_mul(n_obs_size, K, elements)) {
        fi.clear();
        dfi_dx.clear();
        return;
    }
    fi.assign(elements, 0.0);
    dfi_dx.assign(elements, 0.0);
    if (spec.family == scar::CopulaFamily::Student
        && spec.dim == 2
        && student_fill_grid_bivariate(
            spec,
            n_obs,
            r_grid,
            dpsi_grid,
            fi.data(),
            dfi_dx.data())) {
        return;
    }
    for (std::int64_t t = 0; t < n_obs; ++t) {
        const std::size_t row = static_cast<std::size_t>(t) * K;
        copula_pdf_and_grad_row_precomputed_flat(
            spec,
            u,
            t,
            r_grid,
            dpsi_grid,
            fi.data() + row,
            dfi_dx.data() + row);
    }
}

double copula_h_rotated(
    const scar::CopulaSpec& spec,
    double u,
    double v,
    double r) {

    if (spec.family == scar::CopulaFamily::Clayton) {
        return clayton_h_rotated(u, v, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Independent) {
        return u;
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        return gumbel_h_rotated(u, v, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return frank_h_rotated(u, v, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        return joe_h_rotated(u, v, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        return gaussian_h_rotated(u, v, r, static_cast<int>(spec.rotation));
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_h_inverse_rotated(
    const scar::CopulaSpec& spec,
    double q,
    double given,
    double r) {

    if (spec.family == scar::CopulaFamily::Clayton) {
        return clayton_h_inverse_rotated(q, given, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Independent) {
        return q;
    }
    if (spec.family == scar::CopulaFamily::Gumbel) {
        return gumbel_h_inverse_rotated(q, given, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Frank) {
        return frank_h_inverse_rotated(q, given, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Joe) {
        return joe_h_inverse_rotated(q, given, r, static_cast<int>(spec.rotation));
    }
    if (spec.family == scar::CopulaFamily::Gaussian) {
        return gaussian_h_inverse_rotated(q, given, r, static_cast<int>(spec.rotation));
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void copula_fi_row_on_grid(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& x_grid,
    std::vector<double>& fi_row) {

    const int stride =
        (spec.family == scar::CopulaFamily::Student
         || spec.family == scar::CopulaFamily::EquicorrGaussian)
        ? spec.dim
        : 2;
    const double* row =
        u + static_cast<std::size_t>(t) * static_cast<std::size_t>(stride);
    if (spec.family == scar::CopulaFamily::Student) {
        student_fill_row_from_x_grid(
            spec, row, t, x_grid, fi_row.data());
        return;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        std::vector<double> r_grid;
        std::vector<double> dpsi_grid;
        copula_prepare_grid_transform(
            spec, x_grid, r_grid, dpsi_grid);
        equicorr_fill_row(
            spec, row, r_grid, dpsi_grid, fi_row.data(), nullptr);
        return;
    }

    double v1 = 0.0;
    double v2 = 0.0;
    apply_rotation(row[0], row[1], static_cast<int>(spec.rotation), v1, v2);
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        fi_row[j] = copula_pdf_x(spec, v1, v2, x_grid[j]);
    }
}

}  // namespace scar_internal
