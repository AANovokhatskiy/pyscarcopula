#include "scar/copula/rotation.hpp"

#include <limits>

namespace scar::copula {

bool is_valid_rotation(int rotation) noexcept {
    return rotation == 0 || rotation == 90 || rotation == 180 || rotation == 270;
}

void apply_rotation(
    double first,
    double second,
    int rotation,
    double& rotated_first,
    double& rotated_second) noexcept {

    rotated_first = first;
    rotated_second = second;
    if (rotation == 90) {
        rotated_first = 1.0 - first;
    } else if (rotation == 180) {
        rotated_first = 1.0 - first;
        rotated_second = 1.0 - second;
    } else if (rotation == 270) {
        rotated_second = 1.0 - second;
    }
}

double evaluate_rotated_conditional(
    double first,
    double second,
    double parameter,
    int rotation,
    ConditionalKernel kernel) {

    if (kernel == nullptr) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return evaluate_rotated_conditional_with(
        first, second, parameter, rotation, kernel);
}

}  // namespace scar::copula
