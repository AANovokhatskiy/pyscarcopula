#pragma once

#include "scar/core/result.hpp"
#include "scar/scar_ou/types.hpp"
#include "scar/status.hpp"

#include <cstdint>
#include <vector>

namespace scar {

/// Vector-valued SCAR-OU operation with backend and failure diagnostics.
struct ScarOuVectorResult {
    std::vector<double> values;
    OuBackend backend = OuBackend::Matrix;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Likelihood value together with backend and fallback diagnostics.
struct LogLikResult {
    double log_likelihood = 0.0;
    OuBackend backend = OuBackend::Spectral;
    Status status = Status::Ok;
    FailureContext failure{};
    std::vector<OuBackend> fallback_chain;
    int matrix_fallback_reason = SCAR_FALLBACK_NONE;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Negative likelihood and gradients with backend diagnostics.
struct GradLogLikResult {
    double neg_log_likelihood = 0.0;
    std::vector<double> neg_gradient;
    OuBackend backend = OuBackend::Spectral;
    Status status = Status::Ok;
    FailureContext failure{};
    std::vector<OuBackend> fallback_chain;
    int matrix_fallback_reason = SCAR_FALLBACK_NONE;
    std::vector<double> neg_corr_gradient;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Discrete posterior or predictive distribution over latent states.
struct StateDistribution {
    std::vector<double> z_grid;
    std::vector<double> prob;
    OuBackend backend = OuBackend::Matrix;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Fixed-draw samples from a discrete/histogram OU state distribution.
struct OuStateSample {
    std::vector<double> values;
    std::int64_t selection_draws_used = 0;
    std::int64_t jitter_draws_used = 0;
};

using OuStateSampleResult = Result<OuStateSample>;

/// Full posterior state distribution for every observation on one OU grid.
struct SmoothedStateDistribution {
    std::vector<double> z_grid;
    std::vector<double> weights;  ///< Row-major `(n_obs, K)` probabilities.
    std::int64_t n_obs = 0;
    int K = 0;
    OuBackend backend = OuBackend::Matrix;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Raw forward/backward filtering over caller-provided OU-grid emissions.
struct OuGridFilterResult {
    std::vector<double> z_grid;
    std::vector<double> predictive_weights;
    std::vector<double> filtered_weights;
    std::vector<double> final_filtered_density;
    std::vector<double> backward_messages;
    std::vector<double> smoothed_weights;
    std::int64_t n_obs = 0;
    int K = 0;
    OuBackend backend = OuBackend::Matrix;
    bool sparse = false;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
