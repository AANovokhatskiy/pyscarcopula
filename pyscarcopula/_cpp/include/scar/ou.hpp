#pragma once

#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/core/span.hpp"
#include "scar/observation.hpp"
#include "scar/scar_ou/result.hpp"
#include "scar/scar_ou/types.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace scar {

OuGridFilterResult filter_ou_grid_emissions(
    const OuParams& params,
    DoubleView emissions,
    std::int64_t n_obs,
    int emission_columns,
    const OuNumericalConfig& config,
    OuBackend backend,
    bool store_predictive = true,
    bool store_filtered = true,
    bool run_backward = true,
    bool run_smoothing = true);

/// Reusable SCAR-OU evaluator.
///
/// Each instance owns mutable scratch buffers and is intentionally not safe
/// for concurrent calls. Use one evaluator per thread.
class ScarOuEvaluator {
public:
    ScarOuEvaluator();
    explicit ScarOuEvaluator(
        const PreparedDynamicEmission* prepared_emission) noexcept;
    ~ScarOuEvaluator();
    ScarOuEvaluator(const ScarOuEvaluator&) = delete;
    ScarOuEvaluator& operator=(const ScarOuEvaluator&) = delete;
    ScarOuEvaluator(ScarOuEvaluator&&) noexcept;
    ScarOuEvaluator& operator=(ScarOuEvaluator&&) noexcept;

    LogLikResult loglik_spectral(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    LogLikResult loglik_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    LogLikResult loglik_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    LogLikResult loglik_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_spectral(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_spectral(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_directional_spectral(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        const std::vector<double>& corr_direction) const;

    GradLogLikResult neg_loglik_with_grad_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_directional_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        const std::vector<double>& corr_direction) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_directional_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        const std::vector<double>& corr_direction) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_directional_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        const std::vector<double>& corr_direction) const;

    GradLogLikResult neg_loglik_with_grad_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult predictive_mean_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult predictive_mean_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult predictive_mean_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult forward_rosenblatt_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult forward_rosenblatt_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult forward_rosenblatt_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult gaussian_rosenblatt_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult gaussian_rosenblatt_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult gaussian_rosenblatt_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult student_rosenblatt_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult student_rosenblatt_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult student_rosenblatt_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult mixture_h_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult mixture_h_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult mixture_h_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult mixture_h_pair_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult mixture_h_pair_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    ScarOuVectorResult mixture_h_pair_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    StateDistribution state_distribution_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        bool horizon_next) const;

    StateDistribution state_distribution_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        bool horizon_next) const;

    StateDistribution state_distribution_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        bool horizon_next) const;

    SmoothedStateDistribution smoothed_state_distribution_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    SmoothedStateDistribution smoothed_state_distribution_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

    SmoothedStateDistribution smoothed_state_distribution_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

private:
    struct Workspace;

    // Private adapters retained while the compiled kernels are migrated away
    // from legacy status out-parameters. They are not part of the caller API.
    std::vector<double> predictive_mean_local_gh(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> predictive_mean_matrix(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> predictive_mean_auto(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, OuBackend&, int&) const;
    std::vector<double> forward_rosenblatt_local_gh(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> forward_rosenblatt_matrix(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> forward_rosenblatt_auto(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, OuBackend&, int&) const;
    std::vector<double> gaussian_rosenblatt_local_gh(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> gaussian_rosenblatt_matrix(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> gaussian_rosenblatt_auto(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, OuBackend&, int&) const;
    std::vector<double> student_rosenblatt_local_gh(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> student_rosenblatt_matrix(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> student_rosenblatt_auto(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, OuBackend&, int&) const;
    std::vector<double> mixture_h_local_gh(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> mixture_h_matrix(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> mixture_h_auto(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, OuBackend&, int&) const;
    std::vector<double> mixture_h_pair_local_gh(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> mixture_h_pair_matrix(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, int&) const;
    std::vector<double> mixture_h_pair_auto(
        const OuParams&, const CopulaSpec&, ObservationView,
        const OuNumericalConfig&, OuBackend&, int&) const;

    const PreparedDynamicEmission& resolve_dynamic_emission(
        const CopulaSpec& copula,
        std::unique_ptr<PreparedDynamicEmission>& owner) const;

    const PreparedDynamicEmission* prepared_emission_ = nullptr;
    Workspace& workspace() const;
    mutable std::unique_ptr<Workspace> workspace_;
};

/// Evaluator that owns validated observations and a fixed numerical setup.
///
/// Prepared evaluators avoid repeated conversion and allocation inside an
/// optimizer loop. Like `ScarOuEvaluator`, an instance is not safe for
/// concurrent calls.
class PreparedScarOuEvaluator {
public:
    PreparedScarOuEvaluator(
        CopulaSpec copula,
        std::vector<double> observations,
        std::int64_t n_obs,
        int dim,
        OuNumericalConfig config,
        std::string method);
    PreparedScarOuEvaluator(
        CopulaSpec copula,
        std::vector<double> equicorr_sums,
        std::vector<double> equicorr_sum_squares,
        OuNumericalConfig config,
        std::string method);

    void update_student_factor(
        const std::vector<double>& l_inv,
        double log_det);

    LogLikResult loglik(
        const OuParams& params) const;

    GradLogLikResult neg_loglik_with_grad(
        const OuParams& params) const;

    GradLogLikResult neg_loglik_with_grad_and_corr(
        const OuParams& params) const;

    GradLogLikResult neg_loglik_with_grad_and_corr_directional(
        const OuParams& params,
        const std::vector<double>& corr_direction) const;

    ScarOuVectorResult predictive_mean(const OuParams& params) const;

    ScarOuVectorResult mixture_h(const OuParams& params) const;

    ScarOuVectorResult mixture_h_pair(const OuParams& params) const;

    StateDistribution state_distribution(
        const OuParams& params,
        bool horizon_next) const;

private:
    ObservationView view() const noexcept;
    LogLikResult call_loglik(const OuParams& params) const;
    GradLogLikResult call_no_corr(const OuParams& params) const;
    GradLogLikResult call_full_corr(const OuParams& params) const;
    GradLogLikResult call_directional_corr(
        const OuParams& params,
        const std::vector<double>& corr_direction) const;
    ScarOuVectorResult call_predictive_mean(const OuParams& params) const;
    ScarOuVectorResult call_mixture_h(const OuParams& params) const;
    ScarOuVectorResult call_mixture_h_pair(const OuParams& params) const;
    StateDistribution call_state_distribution(
        const OuParams& params,
        bool horizon_next) const;

    CopulaSpec copula_;
    PreparedDynamicEmission emission_;
    std::vector<double> observations_;
    std::int64_t n_obs_ = 0;
    int dim_ = 0;
    OuNumericalConfig config_;
    std::string method_;
    mutable std::mutex call_mutex_;
    ScarOuEvaluator evaluator_;
};

}  // namespace scar
