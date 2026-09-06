#pragma once

#include "scar/core/result.hpp"

#include <vector>

namespace scar {

struct GasLogLikResult {
    double log_likelihood = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Negative GAS log-likelihood and its optimizer-parameter gradient.
///
/// The numerical differentiation loop is deliberately owned by the C++
/// evaluator so client optimizers never reconstruct model recursion math.
struct GasObjectiveGradientResult {
    double objective = 0.0;
    std::vector<double> gradient;
    std::uint64_t objective_evaluations = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Full filtered GAS paths and their total log-likelihood.
struct GasFilterResult {
    std::vector<double> g_path;
    std::vector<double> r_path;
    std::vector<double> score_path;
    double log_likelihood = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct GasUpdateResult {
    double g_next = 0.0;
    double r = 0.0;
    double r_next = 0.0;
    double log_likelihood = 0.0;
    double score = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct GasStateResult {
    double g = 0.0;
    double parameter = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct GasPredictResult {
    double parameter = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Bivariate GAS samples generated from caller-owned random draws.
struct GasSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct GasPathResult {
    std::vector<double> values;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

struct GasOuInitializationResult {
    double kappa = 1.0;
    double mu = 0.0;
    double nu = 1.0;
    double best_log_likelihood = 0.0;
    double selected_omega = 0.0;
    double selected_gamma = 0.0;
    double selected_beta = 0.0;
    bool grid_candidate_found = false;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

}  // namespace scar
