#include "scar_internal.hpp"

namespace scar_internal {

bool is_valid_rotation(int rotation) {
    return rotation == 0 || rotation == 90 || rotation == 180 || rotation == 270;
}

double softplus(double x) {
    if (x > 20.0) {
        return x;
    }
    if (x < -20.0) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

double d_softplus(double x) {
    if (x > 20.0) {
        return 1.0;
    }
    if (x < -20.0) {
        return std::exp(x);
    }
    return 1.0 / (1.0 + std::exp(-x));
}

double log1mexp(double x) {
    if (x > 0.693) {
        return std::log1p(-std::exp(-x));
    }
    if (x > 0.0) {
        return std::log(-std::expm1(-x));
    }
    return -std::numeric_limits<double>::infinity();
}

double logsumexp(double a, double b) {
    const double m = std::max(a, b);
    if (!std::isfinite(m)) {
        return m;
    }
    return m + std::log1p(std::exp(std::min(a, b) - m));
}

double normal_quantile(double p) {
    const double a[] = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    };
    const double b[] = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    };
    const double c[] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    };
    const double d[] = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    };

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;
    p = clip_pseudo_observation(p);

    double x = 0.0;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p > phigh) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    return x;
}

double normal_quantile_refined(double p) {
    p = clip_pseudo_observation(p);
    const double x = normal_quantile(p);
    const double cdf = 0.5 * std::erfc(-x / std::sqrt(2.0));
    const double pdf =
        std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
    return x - (cdf - p) / pdf;
}

double equicorr_transform(const scar::CopulaSpec& spec, double x) {
    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    return rho_min
        + 0.5 * (1.0 - rho_min) * (1.0 + std::tanh(x));
}

double equicorr_inverse_transform(
    const scar::CopulaSpec& spec,
    double rho) {

    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    double scaled =
        2.0 * (rho - rho_min) / (1.0 - rho_min) - 1.0;
    scaled = std::clamp(scaled, -0.999999, 0.999999);
    return std::atanh(scaled);
}

double equicorr_dtransform(const scar::CopulaSpec& spec, double x) {
    const double rho_min = -1.0 / static_cast<double>(spec.dim - 1);
    const double th = std::tanh(x);
    return 0.5 * (1.0 - rho_min) * (1.0 - th * th);
}

double equicorr_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double rho,
    double* dlog_drho) {

    const double one_minus_rho = 1.0 - rho;
    const double common_eigenvalue =
        1.0 + static_cast<double>(spec.dim - 1) * rho;
    if (row == nullptr
        || one_minus_rho <= 0.0
        || common_eigenvalue <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    double sum_squares = 0.0;
    double sum = 0.0;
    for (int j = 0; j < spec.dim; ++j) {
        const double z = normal_quantile_refined(row[j]);
        sum_squares += z * z;
        sum += z;
    }
    const double square_sum = sum * sum;
    const double log_det =
        static_cast<double>(spec.dim - 1) * std::log(one_minus_rho)
        + std::log(common_eigenvalue);
    const double diagonal_term = rho / one_minus_rho;
    const double common_term =
        -rho / (one_minus_rho * common_eigenvalue);

    if (dlog_drho != nullptr) {
        const double dlog_det =
            -static_cast<double>(spec.dim - 1) / one_minus_rho
            + static_cast<double>(spec.dim - 1) / common_eigenvalue;
        const double ddiagonal =
            1.0 / (one_minus_rho * one_minus_rho);
        const double dcommon =
            -(common_eigenvalue
              - rho * one_minus_rho
                  * static_cast<double>(spec.dim - 1))
            / std::pow(one_minus_rho * common_eigenvalue, 2.0);
        *dlog_drho =
            -0.5 * dlog_det
            -0.5 * (ddiagonal * sum_squares + dcommon * square_sum);
    }
    return -0.5 * log_det
        -0.5 * (
            diagonal_term * sum_squares + common_term * square_sum);
}

double copula_transform(const scar::CopulaSpec& spec, double x) {
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return equicorr_transform(spec, x);
    }
    if (spec.transform == scar::Transform::Softplus) {
        return softplus(x) + spec.offset;
    }
    if (spec.transform == scar::Transform::XTanh) {
        return x * std::tanh(x) + spec.offset;
    }
    if (spec.transform == scar::Transform::GaussianTanh) {
        return 0.9999 * std::tanh(x / 4.0);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_inverse_transform(const scar::CopulaSpec& spec, double r) {
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return equicorr_inverse_transform(spec, r);
    }
    if (spec.transform == scar::Transform::Softplus) {
        const double y = spec.family == scar::CopulaFamily::Student
            ? std::max(r - spec.offset, 1e-15)
            : r - spec.offset;
        if (y > 20.0) {
            return y;
        }
        if (y <= 0.0) {
            return std::log(1e-300);
        }
        if (y < 1e-8) {
            return std::log(y);
        }
        return std::log(std::expm1(y));
    }
    if (spec.transform == scar::Transform::XTanh) {
        // x*tanh(x) is even and therefore has no globally unique inverse.
        // Preserve the historical modulus-based positive-branch
        // approximation used for initialization; this is not a round-trip
        // inverse of copula_transform().
        return std::abs(r) + spec.offset;
    }
    if (spec.transform == scar::Transform::GaussianTanh) {
        const double scaled = std::clamp(r / 0.9999, -0.9999, 0.9999);
        return 4.0 * std::atanh(scaled);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double copula_dtransform(const scar::CopulaSpec& spec, double x) {
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return equicorr_dtransform(spec, x);
    }
    if (spec.transform == scar::Transform::Softplus) {
        return d_softplus(x);
    }
    if (spec.transform == scar::Transform::XTanh) {
        const double th = std::tanh(x);
        return th + x * (1.0 - th * th);
    }
    if (spec.transform == scar::Transform::GaussianTanh) {
        const double th = std::tanh(x / 4.0);
        return 0.9999 * 0.25 * (1.0 - th * th);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void apply_rotation(double u1, double u2, int rotation, double& v1, double& v2) {
    v1 = u1;
    v2 = u2;
    if (rotation == 90) {
        v1 = 1.0 - u1;
    } else if (rotation == 180) {
        v1 = 1.0 - u1;
        v2 = 1.0 - u2;
    } else if (rotation == 270) {
        v2 = 1.0 - u2;
    }
}

double evaluate_rotated_conditional(
    double first,
    double second,
    double parameter,
    int rotation,
    ConditionalKernel kernel) {

    if (!is_valid_rotation(rotation) || kernel == nullptr) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double rotated_first = 0.0;
    double rotated_second = 0.0;
    apply_rotation(
        first, second, rotation, rotated_first, rotated_second);
    const double value = kernel(rotated_first, rotated_second, parameter);
    if (rotation == 90 || rotation == 180) {
        return 1.0 - value;
    }
    return value;
}

}  // namespace scar_internal
