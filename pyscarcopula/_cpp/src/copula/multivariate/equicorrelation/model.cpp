#include "scar/copula/multivariate/equicorrelation/model.hpp"

#include "scar/copula/model_storage.hpp"
#include "scar/copula/spec.hpp"

#include <stdexcept>

namespace scar::copula::multivariate::equicorrelation {

ObservationCache& observation_cache(CopulaSpec& spec) {
    TypedModelStorage& storage = spec.model_storage();
    auto* model = std::get_if<ModelStorage>(&storage.value);
    if (model == nullptr) {
        throw std::logic_error(
            "copula does not own an equicorrelation cache");
    }
    return model->observations;
}

const ObservationCache& observation_cache(const CopulaSpec& spec) {
    const TypedModelStorage& storage = spec.model_storage();
    const auto* model = std::get_if<ModelStorage>(&storage.value);
    if (model == nullptr) {
        throw std::logic_error(
            "copula does not own an equicorrelation cache");
    }
    return model->observations;
}

}  // namespace scar::copula::multivariate::equicorrelation
