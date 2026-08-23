#pragma once

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
double evaluate_rotated_conditional(
    double first,
    double second,
    double parameter,
    int rotation,
    ConditionalKernel kernel);

}  // namespace scar::copula
