#pragma once

#include <vector>

namespace scar {
struct CopulaSpec;
}

namespace scar::copula::pair {

struct ObservationCache {
    std::vector<double> gaussian_first_scores;
    std::vector<double> gaussian_second_scores;
};

struct ModelStorage {
    ObservationCache observations;
};

ObservationCache& observation_cache(CopulaSpec& spec);
const ObservationCache& observation_cache(const CopulaSpec& spec);

}  // namespace scar::copula::pair
