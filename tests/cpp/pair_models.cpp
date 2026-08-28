#include "scar/copula/prepared_pair_kernel.hpp"

#include <array>
#include <cmath>

namespace {

struct PairCase {
    scar::CopulaFamily family;
    double parameter;
};

bool close(double first, double second, double tolerance = 2e-8) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

}  // namespace

int run_pair_model_tests() {
    constexpr std::array<PairCase, 6> cases{{
        {scar::CopulaFamily::Independent, 0.0},
        {scar::CopulaFamily::Clayton, 1.5},
        {scar::CopulaFamily::Frank, 2.0},
        {scar::CopulaFamily::Gumbel, 1.5},
        {scar::CopulaFamily::Joe, 1.5},
        {scar::CopulaFamily::Gaussian, 0.35},
    }};
    constexpr double first = 0.37;
    constexpr double second = 0.63;

    for (std::size_t index = 0; index < cases.size(); ++index) {
        const PairCase& test = cases[index];
        const scar::CopulaSpec spec =
            scar::default_pair_copula_spec(test.family);
        const scar::PreparedPairKernel kernel(spec);
        if (!kernel.is_registered()
            || !kernel.is_supported()
            || kernel.family() != test.family) {
            return 1 + static_cast<int>(index) * 10;
        }

        const double log_density = kernel.log_pdf(
            first, second, test.parameter);
        const double conditional = kernel.h(
            first, second, test.parameter);
        const double recovered = kernel.inverse_h(
            conditional, second, test.parameter);
        if (!std::isfinite(log_density)
            || !(conditional > 0.0 && conditional < 1.0)
            || !close(recovered, first)) {
            return 2 + static_cast<int>(index) * 10;
        }

        if (test.family != scar::CopulaFamily::Independent) {
            const double tau = kernel.parameter_to_tau(test.parameter);
            const double recovered_parameter = kernel.tau_to_parameter(tau);
            if (!close(recovered_parameter, test.parameter, 1e-10)) {
                return 3 + static_cast<int>(index) * 10;
            }
        }
    }

    constexpr std::array<scar::CopulaFamily, 3> rotated_families{{
        scar::CopulaFamily::Clayton,
        scar::CopulaFamily::Gumbel,
        scar::CopulaFamily::Joe,
    }};
    constexpr std::array<scar::Rotation, 4> rotations{{
        scar::Rotation::R0,
        scar::Rotation::R90,
        scar::Rotation::R180,
        scar::Rotation::R270,
    }};
    for (std::size_t family_index = 0;
         family_index < rotated_families.size(); ++family_index) {
        for (std::size_t rotation_index = 0;
             rotation_index < rotations.size(); ++rotation_index) {
            scar::CopulaSpec spec = scar::default_pair_copula_spec(
                rotated_families[family_index]);
            spec.rotation = rotations[rotation_index];
            const scar::PreparedPairKernel kernel(spec);
            const double parameter = 1.5;
            const double conditional = kernel.h(first, second, parameter);
            const double recovered = kernel.inverse_h(
                conditional, second, parameter);
            if (!kernel.is_supported() || !close(recovered, first)) {
                return 70
                    + static_cast<int>(family_index) * 10
                    + static_cast<int>(rotation_index);
            }
        }
    }
    return 0;
}
