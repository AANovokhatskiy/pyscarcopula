#include "scar/detail/copula/common.hpp"

#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/copula/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace scar_internal {

scar::CopulaSpec transposed_copula_spec(const scar::CopulaSpec& spec) {
    scar::CopulaSpec transposed = spec;
    if (spec.rotation == scar::Rotation::R90) {
        transposed.rotation = scar::Rotation::R270;
    } else if (spec.rotation == scar::Rotation::R270) {
        transposed.rotation = scar::Rotation::R90;
    }
    return transposed;
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

double copula_transform(const scar::CopulaSpec& spec, double x) {
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return equicorr_transform(spec, x);
    }
    return scar::copula::transform_parameter(
        spec.transform, x, spec.offset);
}

double copula_inverse_transform(const scar::CopulaSpec& spec, double r) {
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return equicorr_inverse_transform(spec, r);
    }
    return scar::copula::inverse_transform_parameter(
        spec.transform,
        r,
        spec.offset,
        spec.family == scar::CopulaFamily::Student);
}

double copula_dtransform(const scar::CopulaSpec& spec, double x) {
    if (spec.family == scar::CopulaFamily::Independent) {
        return 0.0;
    }
    if (spec.family == scar::CopulaFamily::EquicorrGaussian) {
        return equicorr_dtransform(spec, x);
    }
    return scar::copula::d_transform_parameter(spec.transform, x);
}

}  // namespace scar_internal
