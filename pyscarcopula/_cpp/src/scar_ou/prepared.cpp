#include "scar/ou.hpp"

#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace scar {
namespace {

bool valid_method(const std::string& method) {
    return method == "auto"
        || method == "spectral"
        || method == "local"
        || method == "matrix";
}

int expected_dimension(const CopulaSpec& copula) {
    if (copula.family == CopulaFamily::Student
        || copula.family == CopulaFamily::EquicorrGaussian) {
        return copula.dim;
    }
    return 2;
}

}  // namespace

PreparedScarOuEvaluator::PreparedScarOuEvaluator(
    CopulaSpec copula,
    std::vector<double> observations,
    std::int64_t n_obs,
    int dim,
    OuNumericalConfig config,
    std::string method)
    : copula_(std::move(copula)),
      observations_(std::move(observations)),
      n_obs_(n_obs),
      dim_(dim),
      config_(config),
      method_(std::move(method)) {

    // Prepared objects own the observation memory so ObservationView remains
    // valid across repeated optimizer callbacks after the pybind constructor
    // returns.
    if (!valid_method(method_)) {
        throw std::invalid_argument("unsupported transition_method");
    }
    if (n_obs_ < 0 || dim_ < 2) {
        throw std::invalid_argument("invalid observation shape");
    }
    const int expected_dim = expected_dimension(copula_);
    if (dim_ != expected_dim) {
        throw std::invalid_argument(
            "u dimension does not match CopulaSpec dimension");
    }
    std::size_t n_obs_size = 0;
    std::size_t dim_size = 0;
    std::size_t expected_size = 0;
    if (!scar_internal::checked_nonnegative_size(n_obs_, n_obs_size)
        || !scar_internal::checked_nonnegative_size(dim_, dim_size)
        || !scar_internal::checked_size_mul(
            n_obs_size, dim_size, expected_size)
        || observations_.size() != expected_size) {
        throw std::invalid_argument("u shape is not representable");
    }
    if (!std::all_of(
            observations_.begin(),
            observations_.end(),
            [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("u must contain only finite values");
    }
}

void PreparedScarOuEvaluator::update_student_factor(
    const std::vector<double>& l_inv,
    double log_det) {

    // Joint Student fits mutate only the static correlation factor; PPF cache
    // and observations stay prepared for the lifetime of this evaluator.
    if (copula_.family != CopulaFamily::Student) {
        throw std::invalid_argument(
            "update_student_factor requires a Student copula");
    }
    if (!std::isfinite(log_det)) {
        throw std::invalid_argument("log_det must be finite");
    }
    std::size_t dim_size = 0;
    std::size_t matrix_size = 0;
    if (!scar_internal::checked_nonnegative_size(copula_.dim, dim_size)
        || !scar_internal::checked_size_mul(
            dim_size, dim_size, matrix_size)
        || dim_size < 2
        || l_inv.size() != matrix_size) {
        throw std::invalid_argument(
            "l_inv must have size copula.dim * copula.dim");
    }
    for (double value : l_inv) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("l_inv must contain only finite values");
        }
    }
    copula_.l_inv = l_inv;
    copula_.log_det = log_det;
}

ObservationView PreparedScarOuEvaluator::view() const noexcept {
    return {
        observations_.data(),
        static_cast<std::size_t>(n_obs_),
        dim_,
    };
}

LogLikResult PreparedScarOuEvaluator::loglik(
    const OuParams& params) const {
    return call_loglik(params);
}

GradLogLikResult PreparedScarOuEvaluator::neg_loglik_with_grad(
    const OuParams& params) const {
    return call_no_corr(params);
}

GradLogLikResult PreparedScarOuEvaluator::neg_loglik_with_grad_and_corr(
    const OuParams& params) const {
    return call_full_corr(params);
}

GradLogLikResult
PreparedScarOuEvaluator::neg_loglik_with_grad_and_corr_directional(
    const OuParams& params,
    const std::vector<double>& corr_direction) const {
    return call_directional_corr(params, corr_direction);
}

std::vector<double> PreparedScarOuEvaluator::predictive_mean(
    const OuParams& params,
    OuBackend& backend,
    int& status) const {
    return call_predictive_mean(params, backend, status);
}

std::vector<double> PreparedScarOuEvaluator::mixture_h(
    const OuParams& params,
    OuBackend& backend,
    int& status) const {
    return call_mixture_h(params, backend, status);
}

StateDistribution PreparedScarOuEvaluator::state_distribution(
    const OuParams& params,
    bool horizon_next) const {
    return call_state_distribution(params, horizon_next);
}

LogLikResult PreparedScarOuEvaluator::call_loglik(
    const OuParams& params) const {

    const ObservationView u = view();
    if (method_ == "auto") {
        return evaluator_.loglik_auto(params, copula_, u, config_);
    }
    if (method_ == "spectral") {
        return evaluator_.loglik_spectral(params, copula_, u, config_);
    }
    if (method_ == "local") {
        return evaluator_.loglik_local_gh(params, copula_, u, config_);
    }
    return evaluator_.loglik_matrix(params, copula_, u, config_);
}

GradLogLikResult PreparedScarOuEvaluator::call_no_corr(
    const OuParams& params) const {

    const ObservationView u = view();
    if (method_ == "auto") {
        return evaluator_.neg_loglik_with_grad_auto(
            params, copula_, u, config_);
    }
    if (method_ == "spectral") {
        return evaluator_.neg_loglik_with_grad_spectral(
            params, copula_, u, config_);
    }
    if (method_ == "local") {
        return evaluator_.neg_loglik_with_grad_local_gh(
            params, copula_, u, config_);
    }
    return evaluator_.neg_loglik_with_grad_matrix(
        params, copula_, u, config_);
}

GradLogLikResult PreparedScarOuEvaluator::call_full_corr(
    const OuParams& params) const {

    const ObservationView u = view();
    if (method_ == "auto") {
        return evaluator_.neg_loglik_with_grad_and_corr_auto(
            params, copula_, u, config_);
    }
    if (method_ == "spectral") {
        return evaluator_.neg_loglik_with_grad_and_corr_spectral(
            params, copula_, u, config_);
    }
    if (method_ == "local") {
        return evaluator_.neg_loglik_with_grad_and_corr_local_gh(
            params, copula_, u, config_);
    }
    return evaluator_.neg_loglik_with_grad_and_corr_matrix(
        params, copula_, u, config_);
}

GradLogLikResult PreparedScarOuEvaluator::call_directional_corr(
    const OuParams& params,
    const std::vector<double>& corr_direction) const {

    const ObservationView u = view();
    if (method_ == "auto") {
        return evaluator_.neg_loglik_with_grad_and_corr_directional_auto(
            params, copula_, u, config_, corr_direction);
    }
    if (method_ == "spectral") {
        return evaluator_.neg_loglik_with_grad_and_corr_directional_spectral(
            params, copula_, u, config_, corr_direction);
    }
    if (method_ == "local") {
        return evaluator_.neg_loglik_with_grad_and_corr_directional_local_gh(
            params, copula_, u, config_, corr_direction);
    }
    return evaluator_.neg_loglik_with_grad_and_corr_directional_matrix(
        params, copula_, u, config_, corr_direction);
}

std::vector<double> PreparedScarOuEvaluator::call_predictive_mean(
    const OuParams& params,
    OuBackend& backend,
    int& status) const {

    const ObservationView u = view();
    if (method_ == "local") {
        backend = OuBackend::LocalGh;
        return evaluator_.predictive_mean_local_gh(
            params, copula_, u, config_, status);
    }
    if (method_ == "matrix") {
        backend = OuBackend::Matrix;
        return evaluator_.predictive_mean_matrix(
            params, copula_, u, config_, status);
    }
    return evaluator_.predictive_mean_auto(
        params, copula_, u, config_, backend, status);
}

std::vector<double> PreparedScarOuEvaluator::call_mixture_h(
    const OuParams& params,
    OuBackend& backend,
    int& status) const {

    const ObservationView u = view();
    if (method_ == "local") {
        backend = OuBackend::LocalGh;
        return evaluator_.mixture_h_local_gh(
            params, copula_, u, config_, status);
    }
    if (method_ == "matrix") {
        backend = OuBackend::Matrix;
        return evaluator_.mixture_h_matrix(
            params, copula_, u, config_, status);
    }
    return evaluator_.mixture_h_auto(
        params, copula_, u, config_, backend, status);
}

StateDistribution PreparedScarOuEvaluator::call_state_distribution(
    const OuParams& params,
    bool horizon_next) const {

    const ObservationView u = view();
    if (method_ == "local") {
        return evaluator_.state_distribution_local_gh(
            params, copula_, u, config_, horizon_next);
    }
    if (method_ == "matrix") {
        return evaluator_.state_distribution_matrix(
            params, copula_, u, config_, horizon_next);
    }
    return evaluator_.state_distribution_auto(
        params, copula_, u, config_, horizon_next);
}

}  // namespace scar
