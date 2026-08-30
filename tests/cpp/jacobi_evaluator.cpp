#include "scar/jacobi.hpp"

#include "scar/copula/capability.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct PairJacobiCase {
    scar::CopulaFamily family;
    scar::NativeModelId model_id;
    scar::Rotation rotation;
};

bool normalized(const std::vector<double>& values, double tolerance = 2e-12) {
    double total = 0.0;
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            return false;
        }
        total += value;
    }
    return std::abs(total - 1.0) <= tolerance;
}

template <typename Values>
bool finite_values(const Values& values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](double value) { return std::isfinite(value); });
}

scar::JacobiEvaluatorConfig config(
    scar::JacobiTransitionMethod method,
    scar::JacobiTransitionStorage storage) {
    scar::JacobiEvaluatorConfig value;
    value.transition.method = method;
    value.transition.storage = storage;
    value.transition.numerical.n_obs = 8;
    value.transition.numerical.quad_order = 16;
    value.transition.numerical.basis_order = 4;
    value.transition.numerical.gh_order = 3;
    value.transition.numerical.matrix =
        method != scar::JacobiTransitionMethod::SpectralCoeff;
    value.transition.numerical.gradient = true;
    value.transition.derivatives =
        method == scar::JacobiTransitionMethod::LocalFixed;
    return value;
}

}  // namespace

int run_jacobi_evaluator_tests() {
    scar::CopulaSpec copula = scar::default_pair_copula_spec(
        scar::CopulaFamily::Gumbel);
    const std::vector<double> observations{
        0.12, 0.23,
        0.21, 0.74,
        0.33, 0.45,
        0.48, 0.61,
        0.57, 0.32,
        0.69, 0.83,
        0.77, 0.54,
        0.88, 0.91,
    };
    const scar::JacobiParams params{1.2, 0.4, 0.25};
    scar::PreparedScarJacobiEvaluator evaluator(
        copula,
        observations,
        8,
        2,
        config(
            scar::JacobiTransitionMethod::LocalFixed,
            scar::JacobiTransitionStorage::Dense));
    if (evaluator.preparation_count() != 0) {
        return 1;
    }
    const scar::JacobiFilterResult filter = evaluator.filter(params);
    if (!filter.is_ok()
        || filter.value.n_obs != 8
        || filter.value.order != 16
        || filter.value.predicted.size() != 8 * 16
        || filter.value.filtered.size() != 8 * 16
        || filter.value.smoothed.size() != 8 * 16
        || filter.value.scales.size() != 8
        || !normalized(filter.value.current_probability)
        || !normalized(filter.value.next_probability)
        || evaluator.preparation_count() != 1) {
        return 2;
    }
    for (std::size_t row = 0; row < 8; ++row) {
        const std::vector<double> predicted(
            filter.value.predicted.begin() + row * 16,
            filter.value.predicted.begin() + (row + 1) * 16);
        const std::vector<double> filtered(
            filter.value.filtered.begin() + row * 16,
            filter.value.filtered.begin() + (row + 1) * 16);
        const std::vector<double> smoothed(
            filter.value.smoothed.begin() + row * 16,
            filter.value.smoothed.begin() + (row + 1) * 16);
        if (!normalized(predicted)
            || !normalized(filtered)
            || !normalized(smoothed)) {
            return 3;
        }
    }

    const scar::JacobiObjectiveResult objective = evaluator.loglik(params);
    const scar::JacobiObjectiveResult invalid_objective = evaluator.loglik(
        scar::JacobiParams{-1.0, 0.4, 0.25});
    const scar::JacobiEvaluatorVectorResult mean =
        evaluator.predictive_mean(params);
    const scar::JacobiEvaluatorPairResult mixture =
        evaluator.mixture_h_pair(params);
    const scar::JacobiEvaluatorPairResult rosenblatt =
        evaluator.rosenblatt(params);
    const scar::JacobiEvaluatorPairResult gaussian =
        evaluator.gaussian_rosenblatt(params);
    if (!objective.is_ok()
        || !invalid_objective.is_ok()
        || !std::isinf(invalid_objective.value.log_likelihood)
        || !(invalid_objective.value.log_likelihood < 0.0)
        || !std::isinf(invalid_objective.value.objective)
        || !(invalid_objective.value.objective > 0.0)
        || !mean.is_ok() || mean.value.values.size() != 8
        || !mixture.is_ok() || mixture.value.first.size() != 8
        || mixture.value.second.size() != 8
        || !rosenblatt.is_ok() || !gaussian.is_ok()
        || evaluator.preparation_count() != 1
        || std::abs(
            objective.value.log_likelihood
            - filter.value.diagnostics.log_likelihood) > 1e-14) {
        return 4;
    }
    for (std::size_t row = 0; row < 8; ++row) {
        if (!(mixture.value.first[row] > 0.0
              && mixture.value.first[row] < 1.0)
            || !(mixture.value.second[row] > 0.0
                 && mixture.value.second[row] < 1.0)
            || rosenblatt.value.first[row] != observations[2 * row]
            || !std::isfinite(gaussian.value.first[row])
            || !std::isfinite(gaussian.value.second[row])) {
            return 5;
        }
    }

    const scar::JacobiGradientResult gradient =
        evaluator.neg_loglik_with_grad(params);
    if (!gradient.is_ok()
        || !std::isfinite(gradient.value.objective)
        || !std::all_of(
            gradient.value.gradient.begin(), gradient.value.gradient.end(),
            [](double value) { return std::isfinite(value); })
        || std::abs(gradient.value.objective + objective.value.log_likelihood)
            > 2e-12) {
        return 6;
    }

    const scar::JacobiStateDistributionResult current =
        evaluator.state_distribution(
            params, scar::JacobiStateHorizon::Current);
    const scar::JacobiStateDistributionResult next =
        evaluator.state_distribution(params, scar::JacobiStateHorizon::Next);
    if (!current.is_ok() || !next.is_ok()
        || !normalized(current.value.probability)
        || !normalized(next.value.probability)) {
        return 7;
    }
    const scar::JacobiStateDistributionResult conditioned =
        evaluator.condition_state(
            current.value.tau,
            current.value.probability,
            {0.31, 0.79});
    if (!conditioned.is_ok()
        || !normalized(conditioned.value.probability)
        || conditioned.value.probability == current.value.probability) {
        return 8;
    }

    scar::PreparedScarJacobiEvaluator sparse(
        copula,
        observations,
        8,
        2,
        config(
            scar::JacobiTransitionMethod::Local,
            scar::JacobiTransitionStorage::Sparse));
    const scar::JacobiFilterResult sparse_filter = sparse.filter(params);
    if (!sparse_filter.is_ok()
        || !normalized(sparse_filter.value.current_probability)
        || !normalized(sparse_filter.value.next_probability)) {
        return 9;
    }

    scar::PreparedScarJacobiEvaluator coefficient(
        copula,
        observations,
        8,
        2,
        config(
            scar::JacobiTransitionMethod::SpectralCoeff,
            scar::JacobiTransitionStorage::Dense));
    const scar::JacobiFilterResult coefficient_filter =
        coefficient.filter(params);
    const scar::JacobiGradientResult coefficient_gradient =
        coefficient.neg_loglik_with_grad(params);
    if (!coefficient_filter.is_ok()
        || !normalized(coefficient_filter.value.current_probability)
        || !normalized(coefficient_filter.value.next_probability)
        || !coefficient_gradient.is_ok()
        || !std::isfinite(coefficient_gradient.value.objective)
        || !std::all_of(
            coefficient_gradient.value.gradient.begin(),
            coefficient_gradient.value.gradient.end(),
            [](double value) { return std::isfinite(value); })) {
        return 10;
    }

    // The exact 14 registered pair identities must all reach the concrete
    // Jacobi evaluator for objective, gradient, state, and residual paths.
    constexpr std::array<PairJacobiCase, 14> pair_cases{{
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R0},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R90},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R180},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R270},
        {scar::CopulaFamily::Frank,
         scar::NativeModelId::Frank, scar::Rotation::R0},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R0},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R90},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R180},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R270},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R0},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R90},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R180},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R270},
        {scar::CopulaFamily::Gaussian,
         scar::NativeModelId::BivariateGaussian, scar::Rotation::R0},
    }};
    for (std::size_t index = 0; index < pair_cases.size(); ++index) {
        const int base = 20 + static_cast<int>(index) * 5;
        const PairJacobiCase& test = pair_cases[index];
        scar::CopulaSpec identity = scar::default_pair_copula_spec(test.family);
        identity.rotation = test.rotation;
        if (identity.model_descriptor().model_id() != test.model_id) {
            return base;
        }
        for (scar::NativeOperation operation : {
                 scar::NativeOperation::ParameterTransformBoundsInitialization,
                 scar::NativeOperation::RowGridDensityGradient,
                 scar::NativeOperation::LikelihoodObjectiveGradient,
                 scar::NativeOperation::StateFilterSmoother,
                 scar::NativeOperation::RosenblattResidual}) {
            const auto capability = scar::query_capability(
                identity.model_descriptor(),
                operation,
                scar::DynamicsKind::ScarTmJacobi);
            if (!capability.supported || !capability.reason.empty()) {
                return base + 1;
            }
        }
        scar::PreparedScarJacobiEvaluator identity_evaluator(
            identity,
            observations,
            8,
            2,
            config(
                scar::JacobiTransitionMethod::LocalFixed,
                scar::JacobiTransitionStorage::Dense));
        const auto identity_filter = identity_evaluator.filter(params);
        const auto identity_objective = identity_evaluator.loglik(params);
        const auto identity_gradient =
            identity_evaluator.neg_loglik_with_grad(params);
        const auto identity_mixture = identity_evaluator.mixture_h_pair(params);
        const auto identity_rosenblatt = identity_evaluator.rosenblatt(params);
        if (!identity_filter.is_ok() || !identity_objective.is_ok()
            || !identity_gradient.is_ok() || !identity_mixture.is_ok()
            || !identity_rosenblatt.is_ok()
            || !std::isfinite(identity_objective.value.log_likelihood)
            || !std::isfinite(identity_gradient.value.objective)
            || !finite_values(identity_gradient.value.gradient)
            || !normalized(identity_filter.value.current_probability)
            || !normalized(identity_filter.value.next_probability)
            || identity_mixture.value.first.size() != 8
            || identity_mixture.value.second.size() != 8
            || identity_rosenblatt.value.first.size() != 8
            || identity_rosenblatt.value.second.size() != 8
            || !finite_values(identity_mixture.value.first)
            || !finite_values(identity_mixture.value.second)
            || !finite_values(identity_rosenblatt.value.first)
            || !finite_values(identity_rosenblatt.value.second)) {
            return base + 2;
        }
    }

    bool rejected_independent = false;
    try {
        scar::CopulaSpec independent = scar::default_pair_copula_spec(
            scar::CopulaFamily::Independent);
        scar::PreparedScarJacobiEvaluator invalid(
            independent,
            observations,
            8,
            2,
            config(
                scar::JacobiTransitionMethod::Local,
                scar::JacobiTransitionStorage::Dense));
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected_independent = true;
    }
    if (!rejected_independent) {
        return 11;
    }

    for (double value : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
        auto invalid_observations = observations;
        invalid_observations[0] = value;
        bool rejected = false;
        try {
            scar::PreparedScarJacobiEvaluator invalid(
                copula, invalid_observations, 8, 2,
                config(scar::JacobiTransitionMethod::Local,
                       scar::JacobiTransitionStorage::Dense));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            return 12;
        }
    }

    for (const auto& tau : std::vector<std::vector<double>>{
             {-0.1, 0.8}, {0.2, 1.1}, {0.8, 0.2}, {0.2, 0.2}}) {
        const auto invalid = evaluator.condition_state(tau, {0.5, 0.5}, {0.5, 0.5});
        if (invalid.is_ok() || !invalid.value.probability.empty()) {
            return 13;
        }
    }
    for (const auto& probability : std::vector<std::vector<double>>{
             {0.0, 0.0}, {-0.1, 1.1},
             {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}}) {
        if (evaluator.condition_state({0.2, 0.8}, probability, {0.5, 0.5}).is_ok()) {
            return 14;
        }
    }
    for (const auto& observation : std::vector<std::array<double, 2>>{
             {-0.1, 0.5}, {0.5, 1.1}}) {
        if (evaluator.condition_state({0.2, 0.8}, {0.5, 0.5}, observation).is_ok()) {
            return 15;
        }
    }

    // One-row evaluators are used for conditioning, not for a time transition.
    auto single_config = config(scar::JacobiTransitionMethod::Local,
                                scar::JacobiTransitionStorage::Dense);
    single_config.transition.numerical.n_obs = 1;
    scar::PreparedScarJacobiEvaluator single(
        copula, {0.5, 0.5}, 1, 2, single_config);
    const auto reference = single.condition_state({0.2, 0.8}, {0.5, 0.5}, {0.5, 0.5});
    if (!reference.is_ok() || single.loglik(params).is_ok()) {
        return 16;
    }
    for (double scale : {0.2, 2.0, 1e-200, 1e200, 1e-320}) {
        const auto scaled = single.condition_state(
            {0.2, 0.8}, {0.5 * scale, 0.5 * scale}, {0.5, 0.5});
        if (!scaled.is_ok() || !normalized(scaled.value.probability)
            || std::abs(scaled.value.probability[0]
                        - reference.value.probability[0]) > 1e-12) {
            return 17;
        }
    }
    const auto fallback = single.condition_state({0.2, 0.8}, {2.0, 2.0}, {0.0, 1.0});
    if (!fallback.is_ok() || fallback.value.probability != std::vector<double>{0.5, 0.5}) {
        return 18;
    }
    return 0;
}
