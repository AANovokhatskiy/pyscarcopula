#include "scar_internal.hpp"

namespace scar_internal {

double gumbel_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_p1 = std::log(-log_v1);
    const double log_p2 = std::log(-log_v2);

    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);
    const double delta = r * (log_min - log_max);
    const double S = r * log_max + std::log1p(std::exp(delta));
    const double A = std::exp(S / r);

    return (r - 1.0) * (log_p1 + log_p2)
        + (1.0 / r - 2.0) * S
        + std::log(r - 1.0 + A)
        - A
        - log_v1
        - log_v2;
}

double gumbel_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_p1 = std::log(std::max(-log_v1, kPdfEps));
    const double log_p2 = std::log(std::max(-log_v2, kPdfEps));

    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);
    const double delta = r * (log_min - log_max);
    const double exp_delta = std::exp(delta);
    const double S = r * log_max + std::log1p(exp_delta);
    const double sig = exp_delta / (1.0 + exp_delta);
    const double dS_dr = log_max + (log_min - log_max) * sig;
    const double A = std::exp(S / r);
    const double dlogA_dr = (dS_dr * r - S) / (r * r);
    const double dA_dr = A * dlogA_dr;

    return (log_p1 + log_p2)
        - S / (r * r)
        + (1.0 / r - 2.0) * dS_dr
        + (1.0 + dA_dr) / (r - 1.0 + A)
        - dA_dr;
}

void gumbel_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double r = softplus(x) + offset;
    const double log_v1 = std::log(v1);
    const double log_v2 = std::log(v2);
    const double log_p1 = std::log(std::max(-log_v1, kPdfEps));
    const double log_p2 = std::log(std::max(-log_v2, kPdfEps));

    const double log_max = std::max(log_p1, log_p2);
    const double log_min = std::min(log_p1, log_p2);
    const double delta = r * (log_min - log_max);
    const double exp_delta = std::exp(delta);
    const double S = r * log_max + std::log1p(exp_delta);
    const double sig = exp_delta / (1.0 + exp_delta);
    const double dS_dr = log_max + (log_min - log_max) * sig;
    const double A = std::exp(S / r);

    const double log_pdf =
        (r - 1.0) * (log_p1 + log_p2)
        + (1.0 / r - 2.0) * S
        + std::log(r - 1.0 + A)
        - A
        - log_v1
        - log_v2;
    pdf = std::exp(log_pdf);

    const double dlogA_dr = (dS_dr * r - S) / (r * r);
    const double dA_dr = A * dlogA_dr;
    const double dlog_dr =
        (log_p1 + log_p2)
        - S / (r * r)
        + (1.0 / r - 2.0) * dS_dr
        + (1.0 + dA_dr) / (r - 1.0 + A)
        - dA_dr;
    d_pdf_dx = pdf * dlog_dr * d_softplus(x);
}

double gumbel_h_unrotated(double u, double v, double r) {
    const double u_clipped = std::min(std::max(u, kHEps), 1.0 - kHEps);
    const double v_clipped = std::min(std::max(v, kHEps), 1.0 - kHEps);

    if (r < 1.0 + 1e-8) {
        return u_clipped;
    }

    const double log_u = std::log(u_clipped);
    const double log_v = std::log(v_clipped);
    const double y_u = -log_u;
    const double y_v = -log_v;
    if (y_u <= 0.0 || y_v <= 0.0) {
        return u_clipped;
    }

    const double log_y_u = std::log(y_u);
    const double log_y_v = std::log(y_v);
    const double a = r * log_y_u;
    const double b = r * log_y_v;
    const double log_max = std::max(a, b);
    const double log_min = std::min(a, b);
    const double log_S = log_max + std::log1p(std::exp(log_min - log_max));
    const double A = std::exp(log_S / r);
    const double log_h =
        (r - 1.0) * log_y_v
        + (1.0 / r - 1.0) * log_S
        - A
        - log_v;

    const double log_eps = std::log(kHEps);
    const double log_one_minus_eps = std::log(1.0 - kHEps);
    if (log_h <= log_eps) {
        return kHEps;
    }
    if (log_h >= log_one_minus_eps) {
        return 1.0 - kHEps;
    }
    return std::exp(log_h);
}

double gumbel_h_rotated(double u, double v, double r, int rotation) {
    return evaluate_rotated_conditional(
        u, v, r, rotation, gumbel_h_unrotated);
}

double gumbel_h_inverse_unrotated(double q, double given, double r) {
    const double q_clipped = std::min(std::max(q, kHEps), 1.0 - kHEps);
    const double given_clipped = std::min(std::max(given, kHEps), 1.0 - kHEps);
    if (r < 1.0 + 1e-8) {
        return q_clipped;
    }

    const double y = -std::log(given_clipped);
    if (y <= 0.0 || !std::isfinite(y)) {
        return q_clipped;
    }
    const double target =
        std::log(q_clipped) - ((r - 1.0) * std::log(y) - std::log(given_clipped));

    double lo = y;
    double hi = std::max(y - std::log(q_clipped) + r, y + 1.0);
    for (int j = 0; j < 24; ++j) {
        const double f_hi = (1.0 - r) * std::log(hi) - hi - target;
        if (!(f_hi > 0.0)) {
            break;
        }
        hi *= 2.0;
    }

    double A = std::min(std::max(y - std::log(q_clipped), lo), hi);
    for (int j = 0; j < 32; ++j) {
        A = std::min(std::max(A, lo), hi);
        const double f = (1.0 - r) * std::log(A) - A - target;
        if (std::abs(f) < 1e-12) {
            break;
        }
        if (f > 0.0) {
            lo = A;
        } else {
            hi = A;
        }
        const double fp = (1.0 - r) / A - 1.0;
        const double newton = A - f / fp;
        if (newton > lo && newton < hi && std::isfinite(newton)) {
            A = newton;
        } else {
            A = 0.5 * (lo + hi);
        }
    }

    const double A_pow = std::exp(r * std::log(A));
    const double y_pow = std::exp(r * std::log(y));
    const double z_pow = A_pow - y_pow;
    if (!(z_pow > 0.0) || !std::isfinite(z_pow)) {
        return 1.0 - kHEps;
    }
    const double value = std::exp(-std::exp(std::log(z_pow) / r));
    return std::min(std::max(value, kHEps), 1.0 - kHEps);
}

double gumbel_h_inverse_rotated(double q, double given, double r, int rotation) {
    return evaluate_rotated_conditional(
        q, given, r, rotation, gumbel_h_inverse_unrotated);
}

}  // namespace scar_internal
