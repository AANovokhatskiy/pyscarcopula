#pragma once

namespace scar {

/// Mapping used to convert an unconstrained latent state to a parameter.
enum class Transform : int {
    Softplus = 1,
    XTanh = 2,
    GaussianTanh = 3,
    Exponential = 4,
    Logistic = 5,
};

}  // namespace scar

namespace scar::copula {

double softplus(double value);
double inverse_softplus(double value);
double d_softplus(double value);
double logistic_unit(double value);
double logistic_unit_open(double value);
double transform_parameter(Transform transform, double value, double offset);
double inverse_transform_parameter(
    Transform transform,
    double parameter,
    double offset,
    bool positive_softplus_floor = false);
double d_transform_parameter(Transform transform, double value);

}  // namespace scar::copula
