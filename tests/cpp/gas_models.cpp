#include "scar/copula.hpp"
#include "scar/copula/capability.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"
#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/gas.hpp"
#include "scar/model_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

struct PairGasCase {
    scar::CopulaFamily family;
    scar::NativeModelId model_id;
    scar::Rotation rotation;
};

scar::DoubleView view(const std::vector<double>& values) {
    return {values.data(), values.size()};
}

bool close(double first, double second, double tolerance = 3e-9) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

bool close_vectors(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double tolerance = 3e-9) {

    if (first.size() != second.size()) {
        return false;
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (!close(first[index], second[index], tolerance)) {
            return false;
        }
    }
    return true;
}

std::vector<double> flatten(const scar::Observations& observations) {
    std::vector<double> output;
    for (const auto& row : observations) {
        output.insert(output.end(), row.begin(), row.end());
    }
    return output;
}

scar::CopulaSpec pair_spec(const PairGasCase& test) {
    scar::CopulaSpec spec = scar::default_pair_copula_spec(test.family);
    spec.rotation = test.rotation;
    return spec;
}

scar::CopulaSpec equicorr_spec(int dimension) {
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::EquicorrGaussian;
    spec.dim = dimension;
    spec.transform = scar::Transform::GaussianTanh;
    spec.correlation_kind = scar::CorrelationKind::Equicorrelation;
    return spec;
}

scar::CopulaSpec dense_student_spec(
    const std::vector<double>& correlation,
    int dimension) {

    const auto prepared = scar::prepare_dense_correlation(
        view(correlation), dimension);
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = dimension;
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0 + 1e-6;
    spec.correlation_kind = scar::CorrelationKind::Fixed;
    spec.dense_inverse_cholesky() = prepared.inverse_cholesky;
    spec.dense_log_determinant() = prepared.log_determinant;
    return spec;
}

scar::CopulaSpec factor_student_spec(
    const std::shared_ptr<const scar::FactorCorrelationOperator>& factor) {

    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = static_cast<int>(factor->dimension());
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0 + 1e-6;
    spec.correlation_kind = scar::CorrelationKind::Factor;
    spec.factor_operator() = factor;
    spec.dense_log_determinant() = factor->logdet();
    return spec;
}

double finite_difference(
    const scar::GasEvaluator& evaluator,
    const scar::GasParams& params,
    const scar::CopulaSpec& spec,
    scar::ObservationView observations,
    const scar::GasConfig& config,
    std::size_t coordinate,
    double step) {

    scar::GasParams plus = params;
    scar::GasParams minus = params;
    double* plus_values[] = {&plus.omega, &plus.gamma, &plus.beta};
    double* minus_values[] = {&minus.omega, &minus.gamma, &minus.beta};
    *plus_values[coordinate] += step;
    *minus_values[coordinate] -= step;
    return (
        evaluator.negative_log_likelihood(
            plus, spec, observations, config).log_likelihood
        - evaluator.negative_log_likelihood(
            minus, spec, observations, config).log_likelihood)
        / (2.0 * step);
}

}  // namespace

int run_gas_model_tests() {
    constexpr std::array<PairGasCase, 14> pair_cases{{
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
    const std::vector<double> pair_values{
        0.20, 0.70,
        0.35, 0.55,
        0.80, 0.25,
        0.60, 0.45,
        0.15, 0.85,
    };
    const scar::ObservationView pair_observations{
        pair_values.data(), 5, 2};
    const scar::GasParams params{0.07, 0.16, 0.68};
    scar::GasConfig unit_config;
    unit_config.optimizer_gradient_eps = 1e-5;
    const scar::GasEvaluator evaluator;

    const auto bounds = scar::default_gas_parameter_bounds();
    const auto initial_point = scar::gas_default_initial_point(0.4);
    if (!bounds.is_ok() || !initial_point.is_ok()
        || bounds.value.lower.size() != 3
        || bounds.value.upper.size() != 3
        || !close_vectors(
            initial_point.value,
            std::vector<double>{0.02, 0.05, 0.95},
            1e-15)) {
        return 1;
    }

    const auto independent = scar::make_typed_model_descriptor(
        scar::NativeModelId::Independent,
        2,
        scar::CorrelationKind::NotApplicable);
    const auto independent_gas = scar::query_capability(
        independent,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::DynamicsKind::Gas);
    if (independent_gas.supported || independent_gas.reason.empty()) {
        return 2;
    }

    // Every exact dynamic pair identity exercises one prepared scalar owner,
    // the row recursion, native objective+gradient, prediction, Rosenblatt,
    // and both h directions obtained from the same filtered parameter path.
    for (std::size_t index = 0; index < pair_cases.size(); ++index) {
        const PairGasCase& test = pair_cases[index];
        const int base = 10 + static_cast<int>(index) * 10;
        const scar::CopulaSpec spec = pair_spec(test);
        const auto descriptor = spec.model_descriptor();
        for (scar::NativeOperation operation : {
                 scar::NativeOperation::ParameterTransformBoundsInitialization,
                 scar::NativeOperation::PointDensityDerivatives,
                 scar::NativeOperation::RowGridDensityGradient,
                 scar::NativeOperation::LikelihoodObjectiveGradient,
                 scar::NativeOperation::StateFilterSmoother,
                 scar::NativeOperation::RosenblattResidual}) {
            const auto capability = scar::query_capability(
                descriptor, operation, scar::DynamicsKind::Gas);
            if (!capability.supported || !capability.reason.empty()) {
                return base;
            }
        }

        scar::PreparedDynamicEmission emission(spec);
        if (emission.kind() != scar::DynamicEmissionKind::Pair
            || !emission.is_supported()) {
            return base + 1;
        }
        const auto state = evaluator.initial_state_prepared(
            params, emission, unit_config);
        auto workspace = emission.make_workspace(true);
        const auto prepared_update = evaluator.update_one_prepared(
            params,
            emission,
            workspace,
            state.g,
            pair_values[0],
            pair_values[1],
            unit_config);
        const auto cold_update = evaluator.update_one(
            params,
            spec,
            state.g,
            pair_values[0],
            pair_values[1],
            unit_config);
        if (!state.is_ok() || !prepared_update.is_ok()
            || !cold_update.is_ok()
            || !close(prepared_update.g_next, cold_update.g_next, 1e-14)
            || !close(prepared_update.r, cold_update.r, 1e-14)
            || !close(prepared_update.score, cold_update.score, 1e-14)) {
            return base + 2;
        }

        const auto filtered = evaluator.filter(
            params, spec, pair_observations, unit_config);
        const auto log_likelihood = evaluator.log_likelihood(
            params, spec, pair_observations, unit_config);
        const auto negative = evaluator.negative_log_likelihood(
            params, spec, pair_observations, unit_config);
        const auto objective_gradient =
            evaluator.negative_log_likelihood_and_gradient(
                params, spec, pair_observations, unit_config);
        if (!filtered.is_ok() || !log_likelihood.is_ok()
            || !negative.is_ok() || !objective_gradient.is_ok()
            || filtered.g_path.size() != 5
            || filtered.r_path.size() != 5
            || filtered.score_path.size() != 4
            || objective_gradient.gradient.size() != 3
            || objective_gradient.objective_evaluations != 7
            || !close(filtered.log_likelihood, log_likelihood.log_likelihood)
            || !close(negative.log_likelihood, -filtered.log_likelihood)
            || !close(
                objective_gradient.objective,
                negative.log_likelihood)) {
            return base + 3;
        }
        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
            const double expected = finite_difference(
                evaluator,
                params,
                spec,
                pair_observations,
                unit_config,
                coordinate,
                4e-5);
            if (!close(
                    objective_gradient.gradient[coordinate],
                    expected,
                    2e-5)) {
                return base + 4;
            }
        }

        const auto current = evaluator.predict_parameter(
            params, spec, pair_observations, unit_config, false);
        const auto next = evaluator.predict_parameter(
            params, spec, pair_observations, unit_config, true);
        const auto last_update = evaluator.update_one(
            params,
            spec,
            filtered.g_path.back(),
            pair_values[pair_values.size() - 2],
            pair_values[pair_values.size() - 1],
            unit_config);
        if (!current.is_ok() || !next.is_ok() || !last_update.is_ok()
            || !close(current.parameter, filtered.r_path.back())
            || !close(next.parameter, last_update.r_next)) {
            return base + 5;
        }

        const auto h_path = evaluator.h_path(
            params, spec, pair_observations, unit_config);
        if (!h_path.is_ok() || h_path.values.size() != 5) {
            return base + 6;
        }
        for (std::size_t row = 0; row < 5; ++row) {
            double first_given_second = 0.0;
            double second_given_first = 0.0;
            emission.h_pair(
                pair_values[2 * row],
                pair_values[2 * row + 1],
                filtered.r_path[row],
                first_given_second,
                second_given_first);
            if (!close(h_path.values[row], second_given_first, 4e-12)
                || !(first_given_second > 0.0
                     && first_given_second < 1.0)
                || !(second_given_first > 0.0
                     && second_given_first < 1.0)) {
                return base + 7;
            }
        }

        if (index == 0) {
            scar::GasConfig fisher = unit_config;
            fisher.scaling = scar::GasScaling::Fisher;
            const auto fisher_filter = evaluator.filter(
                params, spec, pair_observations, fisher);
            const auto fisher_gradient =
                evaluator.negative_log_likelihood_and_gradient(
                    params, spec, pair_observations, fisher);
            const auto initialization = evaluator.ou_initial_point(
                0.2, spec, pair_observations, unit_config);
            if (!fisher_filter.is_ok() || !fisher_gradient.is_ok()
                || !initialization.is_ok()
                || !initialization.grid_candidate_found) {
                return base + 8;
            }
        }
    }

    // Registered multivariate GAS emissions reuse the same evaluator. Their
    // dynamic parameter paths feed the concrete C++ Rosenblatt owners.
    constexpr int dimension = 3;
    const scar::Observations multivariate_observations{
        {0.20, 0.40, 0.60},
        {0.75, 0.35, 0.55},
        {0.10, 0.80, 0.45},
        {0.90, 0.25, 0.70},
    };
    const std::vector<double> flat = flatten(multivariate_observations);
    const scar::ObservationView multivariate_view{
        flat.data(), multivariate_observations.size(), dimension};
    const std::vector<double> identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const auto zero_factor = std::make_shared<scar::FactorCorrelationOperator>(
        std::vector<double>(dimension, 0.0), dimension, 1, 1e-6);
    const std::array<scar::CopulaSpec, 3> multivariate_specs{
        equicorr_spec(dimension),
        dense_student_spec(identity, dimension),
        factor_student_spec(zero_factor),
    };
    for (std::size_t index = 0;
         index < multivariate_specs.size();
         ++index) {
        const scar::CopulaSpec& spec = multivariate_specs[index];
        scar::PreparedDynamicEmission emission(spec);
        const auto filtered = evaluator.filter(
            params, spec, multivariate_view, unit_config);
        const auto objective_gradient =
            evaluator.negative_log_likelihood_and_gradient(
                params, spec, multivariate_view, unit_config);
        const auto update = evaluator.update_observation(
            params,
            spec,
            filtered.g_path.empty() ? 0.0 : filtered.g_path[0],
            {flat.data(), 1, dimension},
            unit_config);
        if (!emission.is_supported() || !filtered.is_ok()
            || !objective_gradient.is_ok() || !update.is_ok()
            || filtered.r_path.size() != multivariate_observations.size()
            || objective_gradient.gradient.size() != 3) {
            return 200 + static_cast<int>(index) * 10;
        }
        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
            const double expected = finite_difference(
                evaluator,
                params,
                spec,
                multivariate_view,
                unit_config,
                coordinate,
                4e-5);
            if (!close(
                    objective_gradient.gradient[coordinate],
                    expected,
                    3e-5)) {
                return 201 + static_cast<int>(index) * 10;
            }
        }

        scar::MultivariateRosenblattResult residuals;
        if (index == 0) {
            residuals = scar::gaussian_rosenblatt_equicorrelation(
                view(filtered.r_path), multivariate_view, 4);
        } else if (index == 1) {
            residuals = scar::student_rosenblatt_dense(
                view(identity),
                dimension,
                multivariate_view,
                view(filtered.r_path),
                4);
        } else {
            residuals = scar::student_rosenblatt_factor(
                *zero_factor,
                multivariate_view,
                view(filtered.r_path),
                4);
        }
        if (!residuals.is_ok()
            || residuals.residuals.size() != flat.size()) {
            return 202 + static_cast<int>(index) * 10;
        }
    }

    // Joint Stochastic-Student shrinkage fitting has one four-coordinate
    // native objective-gradient owner; Python supplies no correlation FD.
    const std::vector<double> base_correlation{
        1.0, 0.30, 0.10,
        0.30, 1.0, -0.15,
        0.10, -0.15, 1.0,
    };
    const double raw_shrinkage = 0.35;
    const auto joint_gradient =
        evaluator.negative_log_likelihood_and_gradient_shrinkage(
            params,
            multivariate_specs[1],
            view(base_correlation),
            raw_shrinkage,
            multivariate_view,
            unit_config);
    if (!joint_gradient.is_ok()
        || joint_gradient.gradient.size() != 4
        || joint_gradient.objective_evaluations != 9) {
        return 230;
    }
    for (std::size_t coordinate = 0; coordinate < 4; ++coordinate) {
        scar::GasParams plus = params;
        scar::GasParams minus = params;
        double plus_raw = raw_shrinkage;
        double minus_raw = raw_shrinkage;
        constexpr double step = 4e-5;
        if (coordinate < 3) {
            double* plus_values[] = {&plus.omega, &plus.gamma, &plus.beta};
            double* minus_values[] = {&minus.omega, &minus.gamma, &minus.beta};
            *plus_values[coordinate] += step;
            *minus_values[coordinate] -= step;
        } else {
            plus_raw += step;
            minus_raw -= step;
        }
        const auto plus_result =
            evaluator.negative_log_likelihood_and_gradient_shrinkage(
                plus,
                multivariate_specs[1],
                view(base_correlation),
                plus_raw,
                multivariate_view,
                unit_config);
        const auto minus_result =
            evaluator.negative_log_likelihood_and_gradient_shrinkage(
                minus,
                multivariate_specs[1],
                view(base_correlation),
                minus_raw,
                multivariate_view,
                unit_config);
        const double expected =
            (plus_result.objective - minus_result.objective) / (2.0 * step);
        if (!plus_result.is_ok() || !minus_result.is_ok()
            || !close(
                joint_gradient.gradient[coordinate], expected, 4e-5)) {
            return 231 + static_cast<int>(coordinate);
        }
    }

    // Native optimizer gradients preserve SciPy's public absolute ``eps``
    // contract and explicitly support the relative-step override.
    const scar::GasParams step_params{0.07, 1.6, 0.68};
    scar::GasConfig absolute_step_config = unit_config;
    absolute_step_config.optimizer_gradient_eps = 2e-2;
    absolute_step_config.optimizer_gradient_relative = false;
    scar::GasConfig relative_step_config = absolute_step_config;
    relative_step_config.optimizer_gradient_relative = true;
    const scar::CopulaSpec step_spec = pair_spec(pair_cases.back());
    const auto absolute_step_gradient =
        evaluator.negative_log_likelihood_and_gradient(
            step_params,
            step_spec,
            pair_observations,
            absolute_step_config);
    const auto relative_step_gradient =
        evaluator.negative_log_likelihood_and_gradient(
            step_params,
            step_spec,
            pair_observations,
            relative_step_config);
    if (!absolute_step_gradient.is_ok()
        || !relative_step_gradient.is_ok()) {
        return 240;
    }
    constexpr std::array<double, 3> step_values{0.07, 1.6, 0.68};
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
        const double absolute_expected = finite_difference(
            evaluator,
            step_params,
            step_spec,
            pair_observations,
            absolute_step_config,
            coordinate,
            absolute_step_config.optimizer_gradient_eps);
        const double relative_expected = finite_difference(
            evaluator,
            step_params,
            step_spec,
            pair_observations,
            relative_step_config,
            coordinate,
            relative_step_config.optimizer_gradient_eps
                * std::max(1.0, std::abs(step_values[coordinate])));
        if (!close(
                absolute_step_gradient.gradient[coordinate],
                absolute_expected,
                1e-12)
            || !close(
                relative_step_gradient.gradient[coordinate],
                relative_expected,
                1e-12)) {
            return 241 + static_cast<int>(coordinate);
        }
    }

    scar::GasConfig invalid_config = unit_config;
    invalid_config.optimizer_gradient_eps = 0.0;
    const scar::CopulaSpec first_spec = pair_spec(pair_cases[0]);
    const auto invalid_gradient =
        evaluator.negative_log_likelihood_and_gradient(
            params, first_spec, pair_observations, invalid_config);
    if (invalid_gradient.status != scar::Status::InvalidParameter
        || !invalid_gradient.gradient.empty()) {
        return 250;
    }

    return 0;
}
