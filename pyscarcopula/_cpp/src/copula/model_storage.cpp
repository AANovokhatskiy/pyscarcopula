#include "scar/copula/model_storage.hpp"
#include "scar/copula/spec.hpp"

#include <utility>

namespace scar {
namespace {

TypedModelStorage make_model_storage(
    CopulaFamily family,
    CorrelationKind correlation_kind) {

    TypedModelStorage storage;
    if (family == CopulaFamily::Student) {
        storage.value = correlation_kind == CorrelationKind::Factor
            ? TypedModelStorage::Alternative{
                copula::multivariate::student::FactorModelStorage{}}
            : TypedModelStorage::Alternative{
                copula::multivariate::student::DenseModelStorage{}};
    } else if (family == CopulaFamily::EquicorrGaussian) {
        storage.value =
            copula::multivariate::equicorrelation::ModelStorage{};
    } else if (family == CopulaFamily::MultivariateGaussian) {
        storage.value = correlation_kind == CorrelationKind::Factor
            ? TypedModelStorage::Alternative{
                copula::multivariate::gaussian::FactorModelStorage{}}
            : TypedModelStorage::Alternative{
                copula::multivariate::gaussian::DenseModelStorage{}};
    } else {
        storage.value = copula::pair::ModelStorage{};
    }
    return storage;
}

bool storage_matches(
    const TypedModelStorage& storage,
    CopulaFamily family,
    CorrelationKind correlation_kind) noexcept {

    if (family == CopulaFamily::Student) {
        return correlation_kind == CorrelationKind::Factor
            ? std::holds_alternative<
                copula::multivariate::student::FactorModelStorage>(
                    storage.value)
            : std::holds_alternative<
                copula::multivariate::student::DenseModelStorage>(
                    storage.value);
    }
    if (family == CopulaFamily::EquicorrGaussian) {
        return std::holds_alternative<
            copula::multivariate::equicorrelation::ModelStorage>(
                storage.value);
    }
    if (family == CopulaFamily::MultivariateGaussian) {
        return correlation_kind == CorrelationKind::Factor
            ? std::holds_alternative<
                copula::multivariate::gaussian::FactorModelStorage>(
                    storage.value)
            : std::holds_alternative<
                copula::multivariate::gaussian::DenseModelStorage>(
                    storage.value);
    }
    return std::holds_alternative<copula::pair::ModelStorage>(
        storage.value);
}

}  // namespace

CopulaSpec::CopulaSpec()
    : model_storage_(std::make_unique<TypedModelStorage>(
          make_model_storage(family, correlation_kind))) {}

CopulaSpec::~CopulaSpec() = default;

CopulaSpec::CopulaSpec(const CopulaSpec& other)
    : family(other.family),
      rotation(other.rotation),
      transform(other.transform),
      offset(other.offset),
      dim(other.dim),
      correlation_kind(other.correlation_kind),
      model_storage_(std::make_unique<TypedModelStorage>(
          other.model_storage())) {}

CopulaSpec::CopulaSpec(CopulaSpec&& other) noexcept = default;

CopulaSpec& CopulaSpec::operator=(const CopulaSpec& other) {
    if (this == &other) {
        return *this;
    }
    family = other.family;
    rotation = other.rotation;
    transform = other.transform;
    offset = other.offset;
    dim = other.dim;
    correlation_kind = other.correlation_kind;
    model_storage_ = std::make_unique<TypedModelStorage>(
        other.model_storage());
    return *this;
}

CopulaSpec& CopulaSpec::operator=(CopulaSpec&& other) noexcept = default;

void CopulaSpec::reset_model_storage() {
    model_storage_ = std::make_unique<TypedModelStorage>(
        make_model_storage(family, correlation_kind));
}

void CopulaSpec::synchronize_model_storage() const {
    if (model_storage_ == nullptr
        || !storage_matches(*model_storage_, family, correlation_kind)) {
        const_cast<CopulaSpec*>(this)->reset_model_storage();
    }
}

TypedModelStorage& CopulaSpec::model_storage() {
    synchronize_model_storage();
    return *model_storage_;
}

const TypedModelStorage& CopulaSpec::model_storage() const {
    synchronize_model_storage();
    return *model_storage_;
}

std::vector<double>& CopulaSpec::dense_inverse_cholesky() {
    return copula::multivariate::correlation::dense(*this).inverse_cholesky;
}

const std::vector<double>& CopulaSpec::dense_inverse_cholesky() const {
    if (correlation_kind == CorrelationKind::Factor) {
        static const std::vector<double> empty;
        return empty;
    }
    return copula::multivariate::correlation::dense(*this).inverse_cholesky;
}

double& CopulaSpec::dense_log_determinant() {
    if (correlation_kind == CorrelationKind::Factor) {
        return copula::multivariate::correlation::factor(*this)
            .log_determinant;
    }
    return copula::multivariate::correlation::dense(*this).log_determinant;
}

double CopulaSpec::dense_log_determinant() const {
    if (correlation_kind == CorrelationKind::Factor) {
        return copula::multivariate::correlation::factor(*this)
            .log_determinant;
    }
    return copula::multivariate::correlation::dense(*this).log_determinant;
}

std::shared_ptr<const FactorCorrelationOperator>& CopulaSpec::factor_operator() {
    return copula::multivariate::correlation::factor(*this).factor;
}

const std::shared_ptr<const FactorCorrelationOperator>&
CopulaSpec::factor_operator() const {
    return copula::multivariate::correlation::factor(*this).factor;
}

std::size_t& CopulaSpec::factor_dimension_tile() {
    return copula::multivariate::correlation::factor(*this).dimension_tile;
}

std::size_t CopulaSpec::factor_dimension_tile() const {
    return copula::multivariate::correlation::factor(*this).dimension_tile;
}

std::int64_t& CopulaSpec::student_ppf_observation_count() {
    return copula::multivariate::student::ppf_cache(*this).observation_count;
}

std::int64_t CopulaSpec::student_ppf_observation_count() const {
    return copula::multivariate::student::ppf_cache(*this).observation_count;
}

std::vector<double>& CopulaSpec::student_ppf_nodes() {
    return copula::multivariate::student::ppf_cache(*this).nodes;
}

const std::vector<double>& CopulaSpec::student_ppf_nodes() const {
    return copula::multivariate::student::ppf_cache(*this).nodes;
}

std::vector<double>& CopulaSpec::student_ppf_table() {
    return copula::multivariate::student::ppf_cache(*this).table;
}

const std::vector<double>& CopulaSpec::student_ppf_table() const {
    return copula::multivariate::student::ppf_cache(*this).table;
}

std::vector<double>& CopulaSpec::pair_gaussian_first_scores() {
    return copula::pair::observation_cache(*this).gaussian_first_scores;
}

const std::vector<double>& CopulaSpec::pair_gaussian_first_scores() const {
    return copula::pair::observation_cache(*this).gaussian_first_scores;
}

std::vector<double>& CopulaSpec::pair_gaussian_second_scores() {
    return copula::pair::observation_cache(*this).gaussian_second_scores;
}

const std::vector<double>& CopulaSpec::pair_gaussian_second_scores() const {
    return copula::pair::observation_cache(*this).gaussian_second_scores;
}

std::vector<double>& CopulaSpec::equicorr_sum_scores() {
    return copula::multivariate::equicorrelation::observation_cache(*this)
        .sum_scores;
}

const std::vector<double>& CopulaSpec::equicorr_sum_scores() const {
    return copula::multivariate::equicorrelation::observation_cache(*this)
        .sum_scores;
}

std::vector<double>& CopulaSpec::equicorr_sum_squares() {
    return copula::multivariate::equicorrelation::observation_cache(*this)
        .sum_squares;
}

const std::vector<double>& CopulaSpec::equicorr_sum_squares() const {
    return copula::multivariate::equicorrelation::observation_cache(*this)
        .sum_squares;
}

}  // namespace scar
