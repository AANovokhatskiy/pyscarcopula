#include "scar/copula.hpp"
#include "scar/status.hpp"

#include <cmath>
#include <vector>

int main() {
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Independent;

    if (!scar::is_supported(spec)) {
        return 1;
    }

    const scar::Observations observations{{0.25, 0.75}};
    const std::vector<double> parameters{0.0};
    const auto density = scar::copula_pdf(spec, observations, parameters);
    if (density.size() != 1 || std::abs(density.front() - 1.0) > 1e-15) {
        return 2;
    }
    return scar::SCAR_OK;
}

