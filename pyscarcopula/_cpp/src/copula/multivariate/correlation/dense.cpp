#include "scar/copula/multivariate/correlation/dense.hpp"

#include "scar/copula/model_storage.hpp"
#include "scar/copula/spec.hpp"

#include <stdexcept>

namespace scar::copula::multivariate::correlation {

DenseCorrelation& dense(CopulaSpec& spec) {
    TypedModelStorage& storage = spec.model_storage();
    if (auto* gaussian = std::get_if<
            gaussian::DenseModelStorage>(&storage.value)) {
        return gaussian->correlation;
    }
    if (auto* student = std::get_if<
            student::DenseModelStorage>(&storage.value)) {
        return student->correlation;
    }
    throw std::logic_error("copula does not own a dense correlation");
}

const DenseCorrelation& dense(const CopulaSpec& spec) {
    const TypedModelStorage& storage = spec.model_storage();
    if (const auto* gaussian = std::get_if<
            gaussian::DenseModelStorage>(&storage.value)) {
        return gaussian->correlation;
    }
    if (const auto* student = std::get_if<
            student::DenseModelStorage>(&storage.value)) {
        return student->correlation;
    }
    throw std::logic_error("copula does not own a dense correlation");
}

}  // namespace scar::copula::multivariate::correlation
