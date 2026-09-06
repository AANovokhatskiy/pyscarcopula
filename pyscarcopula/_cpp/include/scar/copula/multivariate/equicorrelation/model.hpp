#pragma once

#include <vector>

namespace scar {
struct CopulaSpec;
}

namespace scar::copula::multivariate::equicorrelation {

struct ObservationCache {
    std::vector<double> sum_scores;
    std::vector<double> sum_squares;
};

struct ModelStorage {
    ObservationCache observations;
};

ObservationCache& observation_cache(CopulaSpec& spec);
const ObservationCache& observation_cache(const CopulaSpec& spec);

}  // namespace scar::copula::multivariate::equicorrelation
