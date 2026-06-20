#pragma once

#include "scar/copula.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

inline constexpr int SCAR_OK = 0;
inline constexpr int SCAR_NULL_POINTER = 1;
inline constexpr int SCAR_INVALID_SIZE = 2;
inline constexpr int SCAR_INVALID_FAMILY = 3;
inline constexpr int SCAR_INVALID_ROTATION = 4;
inline constexpr int SCAR_INVALID_TRANSFORM = 5;
inline constexpr int SCAR_INVALID_PARAMETER = 6;
inline constexpr int SCAR_NUMERICAL_FAILURE = 7;

inline constexpr int SCAR_FALLBACK_NONE = 0;
inline constexpr int SCAR_FALLBACK_FAILED = 1;
inline constexpr int SCAR_FALLBACK_CAPPED = 2;

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

struct ObservationView {
    const double* values = nullptr;
    std::size_t n_obs = 0;
    int dim = 0;

    std::size_t size() const noexcept {
        return n_obs;
    }

    bool empty() const noexcept {
        return n_obs == 0;
    }

    const double* data() const noexcept {
        return values;
    }
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

    GradLogLikResult neg_loglik_with_grad_and_corr_auto(
        const OuParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const OuNumericalConfig& config) const;

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
};

}  // namespace scar
