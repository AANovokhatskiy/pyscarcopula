#pragma once

#include "scar/core/result.hpp"
#include "scar/scar_jacobi/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

struct JacobiRawBounds {
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};
};

struct JacobiStationaryShape {
    double alpha = 0.0;
    double beta = 0.0;
    std::array<double, 3> dalpha{};
    std::array<double, 3> dbeta{};
};

struct JacobiMemoryEstimate {
    std::size_t elements = 0;
    std::uint64_t bytes = 0;
    std::uint64_t budget_bytes = 0;
    bool within_budget = false;
};

struct JacobiBoundaryValue {
    double value = 0.0;
    bool intervened = false;
};

struct JacobiQuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

/// Gauss-Jacobi nodes on tau in (0, 1), probability weights, and a row-major
/// orthonormal basis plus its derivative with respect to tau.
struct JacobiBasisRule {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> basis_derivative;
    int quad_order = 0;
    int basis_order = 0;
};

/// Parameter-independent tau grid, normalized stationary masses, and their
/// row-major derivatives with respect to `(kappa, m, xi)`.
struct JacobiFixedRule {
    std::vector<double> tau;
    std::vector<double> weights;
    std::vector<double> weight_derivatives;
    int quad_order = 0;
};

using JacobiParamsResult = Result<JacobiParams>;
using JacobiRawParamsResult = Result<std::array<double, 3>>;
using JacobiRawBoundsResult = Result<JacobiRawBounds>;
using JacobiShapeResult = Result<JacobiStationaryShape>;
using JacobiScalarResult = Result<double>;
using JacobiVectorResult = Result<std::vector<double>>;
using JacobiBoundaryResult = Result<JacobiBoundaryValue>;
using JacobiMemoryResult = Result<JacobiMemoryEstimate>;
using JacobiQuadratureResult = Result<JacobiQuadratureRule>;
using JacobiBasisResult = Result<JacobiBasisRule>;
using JacobiFixedRuleResult = Result<JacobiFixedRule>;

}  // namespace scar
