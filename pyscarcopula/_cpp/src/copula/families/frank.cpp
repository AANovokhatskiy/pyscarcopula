#include "scar/detail/internal.hpp"

namespace scar_internal {

double frank_log_pdf_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);

    const double a = r * v1;
    const double b = r * v2;
    const double log_num = std::log(r) + log1mexp(r) - a - b;
    const double log_t1 = -a + log1mexp(b);
    const double log_t2 = -b + log1mexp(r - b);
    return log_num - 2.0 * logsumexp(log_t1, log_t2);
}

double frank_dlog_pdf_dr_unrotated(double u1, double u2, double r) {
    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    if (std::abs(r) < 1e-10) {
        return 0.0;
    }

    const double emr = std::exp(-r);
    const double emrv1 = std::exp(-r * v1);
    const double emrv2 = std::exp(-r * v2);
    const double emr1v2 = std::exp(-r * (1.0 - v2));
    const double A = emrv1 * (1.0 - emrv2);
    const double B = emrv2 * (1.0 - emr1v2);
    const double D = std::max(A + B, kPdfEps);
    const double dA = emrv1 * (-v1 * (1.0 - emrv2) + v2 * emrv2);
    const double dB = emrv2 * (-v2 * (1.0 - emr1v2) + (1.0 - v2) * emr1v2);

    return 1.0 / r
        + emr / (1.0 - emr)
        - (v1 + v2)
        - 2.0 * (dA + dB) / D;
}

void frank_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx) {

    const double v1 = std::min(std::max(u1, kPdfEps), 1.0 - kPdfEps);
    const double v2 = std::min(std::max(u2, kPdfEps), 1.0 - kPdfEps);
    const double r = softplus(x) + offset;

    const double a = r * v1;
    const double b = r * v2;
    const double log_num = std::log(r) + log1mexp(r) - a - b;
    const double log_t1 = -a + log1mexp(b);
    const double log_t2 = -b + log1mexp(r - b);
    pdf = std::exp(log_num - 2.0 * logsumexp(log_t1, log_t2));

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
    d_pdf_dx = pdf * dlog_dr * d_softplus(x);
}

double frank_h_unrotated(double u, double v, double r) {
    const double u_clipped =
        std::min(std::max(u, kPseudoObsEps), 1.0 - kPseudoObsEps);
    const double v_clipped =
        std::min(std::max(v, kPseudoObsEps), 1.0 - kPseudoObsEps);
    if (std::abs(r) < 1e-8) {
        return u_clipped;
    }

    const double ru = r * u_clipped;
    const double rv = r * v_clipped;
    const double log_numer = -rv + log1mexp(ru);
    const double log_A = -ru + log1mexp(rv);
    const double log_B = -rv + log1mexp(r * (1.0 - v_clipped));
    const double log_h = log_numer - logsumexp(log_A, log_B);

    if (log_h < -700.0) {
        return kPseudoObsEps;
    }
    if (log_h > -kPseudoObsEps) {
        return 1.0 - kPseudoObsEps;
    }
    return std::exp(log_h);
}

double frank_h_rotated(double u, double v, double r, int rotation) {
    if (rotation == 0) {
        return frank_h_unrotated(u, v, r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double frank_h_inverse_unrotated(double q, double given, double r) {
    const double q_clipped =
        std::min(std::max(q, kPseudoObsEps), 1.0 - kPseudoObsEps);
    const double given_clipped =
        std::min(std::max(given, kPseudoObsEps), 1.0 - kPseudoObsEps);
    if (std::abs(r) < 1e-8) {
        return q_clipped;
    }

    const double x3 = std::exp(-r);
    const double log_Q =
        std::log1p(-q_clipped) - std::log(q_clipped) - r * given_clipped;

    double t = q_clipped;
    if (log_Q > 50.0) {
        const double one_minus_arg =
            (1.0 - x3) / (std::exp(log_Q) + 1.0);
        if (one_minus_arg <= 0.0) {
            t = kPseudoObsEps;
        } else {
            t = -std::log1p(-one_minus_arg) / r;
        }
    } else if (log_Q < -745.0) {
        t = 1.0;
    } else {
        const double Q = std::exp(log_Q);
        const double denom = Q + 1.0;
        if (denom <= 0.0 || !std::isfinite(denom)) {
            return q_clipped;
        }
        const double arg = (Q + x3) / denom;
        if (arg <= 0.0) {
            t = 1.0;
        } else if (arg >= 1.0 - kPseudoObsEps) {
            const double one_minus_arg = (1.0 - x3) / denom;
            if (one_minus_arg <= 0.0) {
                t = kPseudoObsEps;
            } else {
                t = -std::log1p(-one_minus_arg) / r;
            }
        } else {
            t = -std::log(arg) / r;
        }
    }
    return std::min(
        std::max(t, kPseudoObsEps), 1.0 - kPseudoObsEps);
}

double frank_h_inverse_rotated(double q, double given, double r, int rotation) {
    if (rotation == 0) {
        return frank_h_inverse_unrotated(q, given, r);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace scar_internal
