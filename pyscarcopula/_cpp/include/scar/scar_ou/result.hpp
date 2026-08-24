#pragma once

#include "scar/copula/grid_values.hpp"
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

struct TrajectoryLogPdfResult {
    GridValues log_pdf;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int parallel_blocks = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
