#pragma once

#include <vector>

namespace scar {
struct CopulaSpec;
}

namespace scar::copula::multivariate::correlation {

/// Prepared dense correlation contract based on an inverse Cholesky factor.
struct DenseCorrelation {
    std::vector<double> inverse_cholesky;
    double log_determinant = 0.0;
};

DenseCorrelation& dense(CopulaSpec& spec);
const DenseCorrelation& dense(const CopulaSpec& spec);

}  // namespace scar::copula::multivariate::correlation
