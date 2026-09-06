#pragma once

#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"

namespace scar::copula::multivariate::student {

struct DenseModelStorage {
    correlation::DenseCorrelation correlation;
    PpfCache ppf;
};

struct FactorModelStorage {
    correlation::FactorCorrelation correlation;
    PpfCache ppf;
};

}  // namespace scar::copula::multivariate::student
