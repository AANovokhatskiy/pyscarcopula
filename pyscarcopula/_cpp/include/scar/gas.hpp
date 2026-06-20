#pragma once

#include "scar/copula.hpp"
#include "scar/ou.hpp"

#include <cstdint>
#include <vector>

namespace scar {

enum class GasScaling : int {
    Unit = 0,
    Fisher = 1,
};

struct GasParams {
    double omega = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
};

struct GasConfig {
    GasScaling scaling = GasScaling::Unit;
    double score_eps = 1e-4;
    double g_clip = 50.0;
    double score_clip = 100.0;
    double fisher_floor = 1e-6;
    double stationary_beta_tol = 1e-8;
};

struct GasLogLikResult {
    double log_likelihood = 0.0;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
};

struct GasFilterResult {
    std::vector<double> g_path;
    std::vector<double> r_path;
    std::vector<double> score_path;
    double log_likelihood = 0.0;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
};

struct GasUpdateResult {
    double g_next = 0.0;
    double r = 0.0;
    double r_next = 0.0;
    double log_likelihood = 0.0;
    double score = 0.0;
    int status = SCAR_OK;
};

struct GasStateResult {
    double g = 0.0;
    double parameter = 0.0;
    int status = SCAR_OK;
};

struct GasPredictResult {
    double parameter = 0.0;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
};

struct GasPathResult {
    std::vector<double> values;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
};

class GasEvaluator {
public:
    GasStateResult initial_state(
        const GasParams& params,
        const CopulaSpec& copula,
        const GasConfig& config) const;

    GasFilterResult filter(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasLogLikResult log_likelihood(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasLogLikResult negative_log_likelihood(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;

    GasUpdateResult update_one(
        const GasParams& params,
        const CopulaSpec& copula,
        double g,
        double u1,
        double u2,
        const GasConfig& config) const;

    GasUpdateResult update_observation(
        const GasParams& params,
        const CopulaSpec& copula,
        double g,
        ObservationView observation,
        const GasConfig& config) const;

    GasPredictResult predict_parameter(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config,
        bool horizon_next) const;

    GasPathResult h_path(
        const GasParams& params,
        const CopulaSpec& copula,
        ObservationView u,
        const GasConfig& config) const;
};

}  // namespace scar
