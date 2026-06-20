#include "scar_internal.hpp"

namespace scar_internal {

namespace {

double joe_log_B(double log_q0, double log_q1) {
    double q0 = 0.0;
    if (log_q0 > -745.0) {
        q0 = std::exp(log_q0);
    }
    const double q0_clipped = std::min(std::max(q0, 0.0), 1.0);
    const double log_one_minus_q0 = std::log1p(-q0_clipped);
    return logsumexp(log_q0, log_one_minus_q0 + log_q1);
}

}  // namespace

double joe_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::min(std::max(1.0 - v1, kPdfEps), 1.0 - kPdfEps);
    const double q2 = std::min(std::max(1.0 - v2, kPdfEps), 1.0 - kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);
    const double log_t1 = r * log_q1 + log1mexp(-r * log_q2);
    const double log_t2 = r * log_q2;
    const double log_B = logsumexp(log_t1, log_t2);
    const double B = std::exp(log_B);

    double log_rp = 0.0;
    if (B > r - 1.0) {
        log_rp = log_B + std::log1p((r - 1.0) / B);
    } else {
        log_rp = std::log(r - 1.0) + std::log1p(B / (r - 1.0));
    }

    return (r - 1.0) * (log_q1 + log_q2)
        + log_rp
        - (2.0 - 1.0 / r) * log_B;
}

double joe_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::max(1.0 - v1, kPdfEps);
    const double q2 = std::max(1.0 - v2, kPdfEps);
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);
    const double q1r = std::pow(q1, r);
    const double q2r = std::pow(q2, r);
    const double B = std::max(q1r + q2r - q1r * q2r, kPdfEps);
    const double dB =
        q1r * log_q1 * (1.0 - q2r)
        + q2r * log_q2 * (1.0 - q1r);

    return log_q1 + log_q2
        + (1.0 + dB) / (r - 1.0 + B)
        - std::log(B) / (r * r)
        - (2.0 - 1.0 / r) * dB / B;
}

void joe_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double q1 = std::min(std::max(1.0 - v1, kPdfEps), 1.0 - kPdfEps);
    const double q2 = std::min(std::max(1.0 - v2, kPdfEps), 1.0 - kPdfEps);
    const double r = softplus(x) + offset;
    const double log_q1 = std::log(q1);
    const double log_q2 = std::log(q2);
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
    pdf = std::exp(log_pdf);

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
    d_pdf_dx = pdf * dlog_dr * d_softplus(x);
}

double joe_h_unrotated(double u, double v, double r) {
    const double u_clipped = std::min(std::max(u, kHEps), 1.0 - kHEps);
    const double v_clipped = std::min(std::max(v, kHEps), 1.0 - kHEps);
    if (r < 1.0 + 1e-8) {
        return u_clipped;
    }

    const double log_1mu = std::log(1.0 - u_clipped);
    const double log_1mv = std::log(1.0 - v_clipped);
    const double log_qu = r * log_1mu;
    const double log_qv = r * log_1mv;
    const double log_B = joe_log_B(log_qu, log_qv);
    if (!std::isfinite(log_B)) {
        return u_clipped;
    }

    const double log_h =
        log1mexp(-log_qu)
        + (1.0 / r - 1.0) * log_B
        + log_qv
        - log_1mv;
    const double log_eps = std::log(kHEps);
    const double log_one_minus_eps = std::log(1.0 - kHEps);
    if (log_h <= log_eps) {
        return kHEps;
    }
    if (log_h >= log_one_minus_eps) {
        return 1.0 - kHEps;
    }
    if (!std::isfinite(log_h)) {
        return u_clipped;
    }
    return std::exp(log_h);
}

double joe_h_rotated(double u, double v, double r, int rotation) {
    return evaluate_rotated_conditional(
        u, v, r, rotation, joe_h_unrotated);
}

double joe_h_inverse_unrotated(double q, double given, double r) {
    const double q_clipped = std::min(std::max(q, kHEps), 1.0 - kHEps);
    const double given_clipped = std::min(std::max(given, kHEps), 1.0 - kHEps);
    if (r < 1.0 + 1e-8) {
        return q_clipped;
    }

    double lo = kHEps;
    double hi = 1.0 - kHEps;
    double t = q_clipped;
    for (int j = 0; j < 50; ++j) {
        t = std::min(std::max(t, lo), hi);
        const double h_val = joe_h_unrotated(t, given_clipped, r);
        const double err = h_val - q_clipped;
        if (std::abs(err) < 1e-10) {
            break;
        }
        if (err > 0.0) {
            hi = t;
        } else {
            lo = t;
        }

        const double dt = std::max(t * 1e-7, 1e-12);
        const double t_p = std::min(t + dt, 1.0 - kHEps);
        const double dh_dt = (joe_h_unrotated(t_p, given_clipped, r) - h_val)
            / std::max(t_p - t, 1e-300);
        const double newton = t - err / dh_dt;
        if (std::isfinite(newton) && std::abs(dh_dt) > 1e-300
            && newton > lo && newton < hi) {
            t = newton;
        } else {
            t = 0.5 * (lo + hi);
        }
    }
    return std::min(std::max(t, kHEps), 1.0 - kHEps);
}

double joe_h_inverse_rotated(double q, double given, double r, int rotation) {
    return evaluate_rotated_conditional(
        q, given, r, rotation, joe_h_inverse_unrotated);
}

}  // namespace scar_internal
