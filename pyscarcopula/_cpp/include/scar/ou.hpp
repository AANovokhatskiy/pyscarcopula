#pragma once

#include "scar/copula.hpp"
#include "scar/observation.hpp"
#include "scar/status.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace scar {

enum class OuBackend : int {
    Spectral = 0,
    LocalGh = 1,
    Matrix = 2,
};

struct OuParams {
    double kappa = 1.0;
    double mu = 0.0;
    double nu = 1.0;
};

struct OuNumericalConfig {
    int K = 300;
    double grid_range = 5.0;
    bool adaptive = true;
    int pts_per_sigma = 4;
    int max_K = 1000;
    double r_gh = 3.0;
    int gh_order = 5;
    double auto_small_kdt = 1e-2;
    int spectral_basis_order = 32;
    int spectral_quad_order = 0;
};

struct LogLikResult {
    double log_likelihood = 0.0;
    OuBackend backend = OuBackend::Spectral;
    int status = 0;
    int fallback_from = -1;
    std::vector<OuBackend> fallback_chain;
    int matrix_fallback_reason = SCAR_FALLBACK_NONE;
};

struct GradLogLikResult {
    double neg_log_likelihood = 0.0;
    std::vector<double> neg_gradient;
    OuBackend backend = OuBackend::Spectral;
    int status = 0;
    int fallback_from = -1;
    std::vector<OuBackend> fallback_chain;
    int matrix_fallback_reason = SCAR_FALLBACK_NONE;
    std::vector<double> neg_corr_gradient;
};

struct StateDistribution {
    std::vector<double> z_grid;
    std::vector<double> prob;
    OuBackend backend = OuBackend::Matrix;
    int status = 0;
};

struct TrajectoryLogPdfResult {
    GridValues log_pdf;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
};

TrajectoryLogPdfResult copula_log_pdf_trajectory_grid(
    const CopulaSpec& copula,
    ObservationView u,
    const double* latent_paths,
    std::size_t n_trajectories);

struct ScarOuGridGradientOperators {
    int K = 0;
    int width = 0;
    bool local = false;
    std::vector<double> dense;
    std::vector<double> dense_grad;
    std::vector<int> cols;
    std::vector<double> vals;
    std::vector<double> grad_vals;
};

struct ScarOuGridGradientWorkspace {
    ScarOuGridGradientOperators op;
    std::vector<double> xi;
    std::vector<double> base_w;
    std::vector<double> pw_const;
    std::vector<double> x_grid;
    std::vector<double> fi;
    std::vector<double> dfi_dx;
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    std::vector<double> beta;
    std::vector<double> c_vals;
    std::vector<double> target;
    std::vector<double> next;
    std::vector<double> dx_dalpha;
    std::vector<double> d_beta;
    std::vector<double> new_d_beta;
    std::vector<double> d_target;
    std::vector<double> contrib;
    std::vector<double> transition_grad;
    std::vector<double> precision;
    std::vector<double> scores;
    std::vector<double> alpha;
    std::vector<double> alpha_source;
    std::vector<double> alpha_next;
};

struct ScarOuSpectralGradientWorkspace {
    int cached_quad_order = 0;
    int cached_basis_order = 0;
    std::vector<double> z;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> weighted_basis;
    std::vector<double> powers;
    std::vector<double> dpowers_dkappa;
    std::vector<double> x_grid;
    std::vector<double> dx_dalpha;
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    std::vector<double> coeff;
    std::vector<double> dcoeff;
    std::vector<double> projected;
    std::vector<double> dprojected;
    std::vector<double> raw;
    std::vector<double> draw;
    std::vector<double> fi_row;
    std::vector<double> dfi_dx_row;
    std::vector<double> precision;
    std::vector<double> scores;
    std::vector<double> corr_coeff;
    std::vector<double> corr_projected;
    std::vector<double> corr_raw;
    std::vector<double> corr_value_projected;
    std::vector<double> corr_dlog_scale;
};

class ScarOuEvaluator {
public:
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

    std::vector<double> predictive_mean_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        int& status) const;

    std::vector<double> predictive_mean_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        int& status) const;

    std::vector<double> predictive_mean_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        OuBackend& backend,
        int& status) const;

    std::vector<double> mixture_h_local_gh(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        int& status) const;

    std::vector<double> mixture_h_matrix(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        int& status) const;

    std::vector<double> mixture_h_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config,
        OuBackend& backend,
        int& status) const;

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

private:
    // Reused by prepared evaluators during one fit/objective loop. These
    // buffers are mutable because likelihood methods keep const call sites;
    // a ScarOuEvaluator instance is therefore intentionally not thread-safe.
    mutable ScarOuGridGradientWorkspace grid_gradient_workspace_;
    mutable ScarOuSpectralGradientWorkspace spectral_gradient_workspace_;
};

class PreparedScarOuEvaluator {
public:
    PreparedScarOuEvaluator(
        CopulaSpec copula,
        std::vector<double> observations,
        std::int64_t n_obs,
        int dim,
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

    std::vector<double> predictive_mean(
        const OuParams& params,
        OuBackend& backend,
        int& status) const;

    std::vector<double> mixture_h(
        const OuParams& params,
        OuBackend& backend,
        int& status) const;

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
    std::vector<double> call_predictive_mean(
        const OuParams& params,
        OuBackend& backend,
        int& status) const;
    std::vector<double> call_mixture_h(
        const OuParams& params,
        OuBackend& backend,
        int& status) const;
    StateDistribution call_state_distribution(
        const OuParams& params,
        bool horizon_next) const;

    CopulaSpec copula_;
    std::vector<double> observations_;
    std::int64_t n_obs_ = 0;
    int dim_ = 0;
    OuNumericalConfig config_;
    std::string method_;
    ScarOuEvaluator evaluator_;
};

}  // namespace scar
