#pragma once

#include <limits>

namespace scar {

/// Standard pair-copula rotations, expressed in degrees.
enum class Rotation : int {
    R0 = 0,
    R90 = 90,
    R180 = 180,
    R270 = 270,
};

}  // namespace scar

namespace scar::copula {

using ConditionalKernel = double (*)(double, double, double);

bool is_valid_rotation(int rotation) noexcept;
void apply_rotation(
    double first,
    double second,
    int rotation,
    double& rotated_first,
    double& rotated_second) noexcept;

template <typename Kernel>
double evaluate_rotated_conditional_with(
    double first, double second, double parameter, int rotation, Kernel kernel) {
    if (!is_valid_rotation(rotation)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double rotated_first = 0.0;
    double rotated_second = 0.0;
    apply_rotation(first, second, rotation, rotated_first, rotated_second);
    const double value = kernel(rotated_first, rotated_second, parameter);
    return rotation == 90 || rotation == 180 ? 1.0 - value : value;
}

double evaluate_rotated_conditional(
    double first,
    double second,
    double parameter,
    int rotation,
    ConditionalKernel kernel);

}  // namespace scar::copula
