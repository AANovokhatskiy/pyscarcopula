#pragma once

#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"

namespace scar::copula::multivariate::gaussian {

struct DenseModelStorage {
    correlation::DenseCorrelation correlation;
};

struct FactorModelStorage {
    correlation::FactorCorrelation correlation;
};

}  // namespace scar::copula::multivariate::gaussian
