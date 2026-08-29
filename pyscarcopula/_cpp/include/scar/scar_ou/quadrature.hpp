#pragma once

#include "scar/core/result.hpp"

#include <vector>

namespace scar {

struct OuHermiteRule {
    std::vector<double> nodes;
    std::vector<double> weights;
    std::vector<double> basis;
    int quad_order = 0;
    int basis_order = 0;
};

Result<OuHermiteRule> ou_hermite_rule(int quad_order, int basis_order);
Result<int> ou_default_quad_order(int basis_order);

}  // namespace scar
