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
    p = std::min(std::max(p, 1e-10), 1.0 - 1e-10);

    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > phigh) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

double copula_transform(const scar::CopulaSpec& spec, double x) {
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

double copula_dtransform(const scar::CopulaSpec& spec, double x) {
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

}  // namespace scar_internal
