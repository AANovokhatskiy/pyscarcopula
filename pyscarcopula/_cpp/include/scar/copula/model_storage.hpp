#pragma once

#include "scar/copula/multivariate/equicorrelation/model.hpp"
#include "scar/copula/multivariate/gaussian/model.hpp"
#include "scar/copula/multivariate/student/model.hpp"
#include "scar/copula/pair/model.hpp"

#include <variant>

namespace scar {

/// Exactly one family/correlation alternative owns all model-specific state.
struct TypedModelStorage {
    using Alternative = std::variant<
        copula::pair::ModelStorage,
        copula::multivariate::gaussian::DenseModelStorage,
        copula::multivariate::gaussian::FactorModelStorage,
        copula::multivariate::equicorrelation::ModelStorage,
        copula::multivariate::student::DenseModelStorage,
        copula::multivariate::student::FactorModelStorage>;

    Alternative value;
};

}  // namespace scar
