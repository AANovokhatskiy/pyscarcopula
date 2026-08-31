#include "scar/copula/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace scar::copula {
constexpr double kLogisticCap = 20.0;
constexpr double kLogisticScale = 2.0;

double logistic_unit(double value) {
    if (value >= 0.0) {
        const double exp_neg = std::exp(-value);
        return 1.0 / (1.0 + exp_neg);
    }
    const double exp_pos = std::exp(value);
    return exp_pos / (1.0 + exp_pos);
}

double logistic_unit_open(double value) {
    return std::clamp(
        logistic_unit(value),
        std::nextafter(0.0, 1.0),
        std::nextafter(1.0, 0.0));
}

double softplus(double value) {
    // Preserve the historical arithmetic and tail approximations: changing
    // their rounding changes finite-difference GAS optimization trajectories.
    // The tail error is at most exp(-20), and these branches prevent overflow.
    if (value > 20.0) {
        return value;
    }
    if (value < -20.0) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

double inverse_softplus(double value) {
    if (value > 20.0) {
        return value;
    }
    // The historical small-value approximation avoids cancellation; its
    // absolute latent error is below 5e-9. Zero/negative inputs retain log's
    // -infinity/NaN behavior; the public parameter inverse applies its floor.
    if (value < 1e-8) {
        return std::log(value);
    }
    return std::log(std::expm1(value));
}

double d_softplus(double value) {
    if (value > 20.0) {
        return 1.0;
    }
    if (value < -20.0) {
        return std::exp(value);
    }
    return 1.0 / (1.0 + std::exp(-value));
}

double transform_parameter(
    Transform transform,
    double value,
    double offset) {

    if (transform == Transform::Softplus) {
        return softplus(value) + offset;
    }
    if (transform == Transform::XTanh) {
        return value * std::tanh(value) + offset;
    }
    if (transform == Transform::Exponential) {
        return std::exp(value) + offset;
    }
    if (transform == Transform::Logistic) {
        return offset
            + kLogisticCap * logistic_unit(value / kLogisticScale);
    }
    if (transform == Transform::GaussianTanh) {
        return 0.9999 * std::tanh(value / 4.0);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double inverse_transform_parameter(
    Transform transform,
    double parameter,
    double offset,
    bool positive_softplus_floor) {

    if (transform == Transform::Softplus) {
        const double value = positive_softplus_floor
            ? std::max(parameter - offset, 1e-15)
            : parameter - offset;
        if (value <= 0.0) {
            return std::log(1e-300);
        }
        return inverse_softplus(value);
    }
    if (transform == Transform::XTanh) {
        // x*tanh(x) is even and therefore has no globally unique inverse.
        // Preserve the historical modulus-based positive-branch
        // approximation used for initialization.
        return std::abs(parameter) + offset;
    }
    if (transform == Transform::Exponential) {
        if (!std::isfinite(parameter) || parameter < offset) {
            throw std::invalid_argument(
                "exponential inverse-transform parameter must be finite "
                "and greater than or equal to the transform offset");
        }
        return std::log(std::max(parameter - offset, 1e-300));
    }
    if (transform == Transform::Logistic) {
        if (!std::isfinite(parameter)
            || parameter < offset
            || parameter > offset + kLogisticCap) {
            throw std::invalid_argument(
                "logistic inverse-transform parameter must be finite and "
                "within [offset, offset + 20]");
        }
        const double probability = std::clamp(
            (parameter - offset) / kLogisticCap,
            1e-15,
            1.0 - 1e-15);
        return kLogisticScale
            * (std::log(probability) - std::log1p(-probability));
    }
    if (transform == Transform::GaussianTanh) {
        const double scaled = std::clamp(
            parameter / 0.9999, -0.9999, 0.9999);
        return 4.0 * std::atanh(scaled);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double d_transform_parameter(Transform transform, double value) {
    if (transform == Transform::Softplus) {
        return d_softplus(value);
    }
    if (transform == Transform::XTanh) {
        const double th = std::tanh(value);
        return th + value * (1.0 - th * th);
    }
    if (transform == Transform::Exponential) {
        return std::exp(value);
    }
    if (transform == Transform::Logistic) {
        const double probability = logistic_unit(value / kLogisticScale);
        return (kLogisticCap / kLogisticScale)
            * probability * (1.0 - probability);
    }
    if (transform == Transform::GaussianTanh) {
        const double th = std::tanh(value / 4.0);
        return 0.9999 * 0.25 * (1.0 - th * th);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace scar::copula
