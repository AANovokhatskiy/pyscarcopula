#include "scar/copula.hpp"
#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/gas.hpp"
#include "scar/jacobi.hpp"
#include "scar/model_policy.hpp"
#include "scar/numerical_validation.hpp"
#include "scar/ou.hpp"
#include "scar/rvine.hpp"
#include "scar/scar_ou/quadrature.hpp"
#include "scar/scar_ou/initialization.hpp"
#include "scar/scar_ou/parameterization.hpp"
#include "scar/scar_ou/policy.hpp"
#include "scar/scar_ou/sampling.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {

bool all_zero(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value) || std::abs(value) > 1e-15) {
            return false;
        }
    }
    return true;
}

}  // namespace

int run_application_model_tests() {
    const double statistics_values[] = {1.25, -0.5, 2.0, -0.25};
    const std::int64_t statistics_counts[] = {3, 0, 7};
    const double kendall_first[] = {1.0, 1.0, 2.0, 3.0};
    const double kendall_second[] = {1.0, 2.0, 2.0, 3.0};
    const auto value_sum = scar::sum_values({statistics_values, 4});
    const auto count_sum = scar::sum_int64({statistics_counts, 3});
    const auto bic = scar::information_criterion(
        -12.5, 3, 100, scar::InformationCriterion::Bic);
    const auto tau = scar::kendall_tau(
        {kendall_first, 4}, {kendall_second, 4});
    const auto ranks = scar::dense_ranks_no_ties({statistics_values, 4});
    if (!value_sum.is_ok() || value_sum.value != 2.5
        || !count_sum.is_ok() || count_sum.value != 10
        || !bic.is_ok()
        || std::abs(bic.value - (25.0 + 3.0 * std::log(100.0))) > 1e-15
        || !tau.is_ok() || std::abs(tau.value - 0.8) > 1e-15
        || !ranks.is_ok()
        || ranks.value != std::vector<std::int64_t>{3, 1, 4, 2}) {
        return 16;
    }
    const double fixed_normals[] = {0.5, -1.0, 0.25};
    const auto trajectory = scar::sample_ou_trajectory(
        scar::OuParams{1.4, 0.2, 0.9}, scar::DoubleView{fixed_normals, 3});
    const double rho = std::exp(-0.7);
    const double sigma = std::sqrt(0.81 / 2.8 * (1.0 - rho * rho));
    const double x0 = 0.2 + 0.9 / std::sqrt(2.8) * 0.5;
    const double x1 = 0.2 + rho * (x0 - 0.2) - sigma;
    if (!trajectory.is_ok() || trajectory.values.size() != 3
        || std::abs(trajectory.values[0] - x0) > 1e-15
        || std::abs(trajectory.values[1] - x1) > 1e-15
        || std::abs(trajectory.values[2] - (0.2 + rho * (x1 - 0.2) + sigma * 0.25)) > 1e-15) {
        return 5;
    }
    const auto invalid_trajectory = scar::sample_ou_trajectory(
        scar::OuParams{-1.0, 0.2, 0.9}, scar::DoubleView{fixed_normals, 3});
    const auto empty_trajectory = scar::sample_ou_trajectory(
        scar::OuParams{1.4, 0.2, 0.9}, scar::DoubleView{});
    if (invalid_trajectory.status != scar::Status::InvalidParameter
        || !empty_trajectory.is_ok() || !empty_trajectory.values.empty()) {
        return 6;
    }
    const auto stationary = scar::sample_ou_stationary(
        scar::OuParams{2.0, 0.3, 4.0}, scar::DoubleView{fixed_normals, 3});
    const double stationary_sigma = 4.0 / std::sqrt(4.0);
    if (!stationary.is_ok() || stationary.values.size() != 3
        || std::abs(stationary.values[0]
            - (0.3 + stationary_sigma * fixed_normals[0])) > 1e-15
        || std::abs(stationary.values[1]
            - (0.3 + stationary_sigma * fixed_normals[1])) > 1e-15
        || std::abs(stationary.values[2]
            - (0.3 + stationary_sigma * fixed_normals[2])) > 1e-15) {
        return 12;
    }
    const std::vector<double> state_grid{-1.0, 0.0, 2.0};
    const std::vector<double> state_probability{0.2, 0.3, 0.5};
    const std::vector<double> state_selection{0.0, 0.2, 0.499999, 0.5, 0.999};
    const auto grid_sample = scar::sample_ou_state_distribution(
        {state_grid.data(), state_grid.size()},
        {state_probability.data(), state_probability.size()},
        {state_selection.data(), state_selection.size()},
        {},
        scar::OuStateSamplingMode::Grid);
    if (!grid_sample.is_ok()
        || grid_sample.value.values
            != std::vector<double>{-1.0, 0.0, 0.0, 2.0, 2.0}
        || grid_sample.value.selection_draws_used != 5
        || grid_sample.value.jitter_draws_used != 0) {
        return 13;
    }
    const std::vector<double> histogram_selection{0.1, 0.6};
    const std::vector<double> histogram_jitter{0.5, 0.25};
    const auto histogram_sample = scar::sample_ou_state_distribution(
        {state_grid.data(), state_grid.size()},
        {state_probability.data(), state_probability.size()},
        {histogram_selection.data(), histogram_selection.size()},
        {histogram_jitter.data(), histogram_jitter.size()},
        scar::OuStateSamplingMode::Histogram);
    if (!histogram_sample.is_ok()
        || histogram_sample.value.values != std::vector<double>{-0.75, 1.25}
        || histogram_sample.value.selection_draws_used != 2
        || histogram_sample.value.jitter_draws_used != 2) {
        return 14;
    }
    const double condition_observation[] = {0.25, 0.75};
    scar::CopulaSpec independent_state;
    independent_state.family = scar::CopulaFamily::Independent;
    const scar::StateDistribution conditioned =
        scar::condition_ou_state_distribution(
            independent_state,
            {state_grid.data(), state_grid.size()},
            {state_probability.data(), state_probability.size()},
            scar::ObservationView{condition_observation, 1, 2});
    if (!conditioned.is_ok()
        || conditioned.z_grid != state_grid
        || conditioned.prob.size() != state_probability.size()
        || std::abs(conditioned.prob[0] - state_probability[0]) > 1e-15
        || std::abs(conditioned.prob[1] - state_probability[1]) > 1e-15
        || std::abs(conditioned.prob[2] - state_probability[2]) > 1e-15) {
        return 15;
    }
    const auto rule = scar::ou_hermite_rule(3, 3);
    if (!rule.is_ok() || rule.value.nodes.size() != 3
        || std::abs(rule.value.nodes[0] + std::sqrt(3.0)) > 1e-14
        || std::abs(rule.value.weights[1] - 2.0 / 3.0) > 1e-14
        || scar::ou_default_quad_order(32).value != 80
        || scar::ou_default_quad_order(0).is_ok()
        || scar::ou_hermite_rule(2, 3).is_ok()) {
        return 7;
    }
    scar::CopulaSpec independent;
    independent.family = scar::CopulaFamily::Independent;
    const scar::Observations observations{
        {0.25, 0.75},
        {0.60, 0.40},
        {0.30, 0.80},
    };
    scar::StaticCopulaEvaluator static_model(independent, observations, 2);
    const scar::StaticObjectiveResult objective =
        static_model.objective_value(0.0);
    if (!objective.is_ok()
        || std::abs(objective.negative_log_likelihood) > 1e-15
        || !all_zero(static_model.log_pdf_rows(0.0))) {
        return 1;
    }

    const std::vector<double> observation_values{
        0.25, 0.75,
        0.60, 0.40,
        0.30, 0.80,
    };
    const scar::ObservationView observation_view{
        observation_values.data(), 3, 2};
    const scar::GasEvaluator gas;
    const scar::GasFilterResult gas_result = gas.filter(
        scar::GasParams{0.1, 0.2, 0.3},
        independent,
        observation_view,
        scar::GasConfig{});
    if (!gas_result.is_ok()
        || gas_result.g_path.size() != 3
        || gas_result.r_path.size() != 3
        || gas_result.score_path.size() != 2
        || std::abs(gas_result.log_likelihood) > 1e-15) {
        return 2;
    }

    std::vector<double> emissions(3 * 9, 1.0);
    scar::OuNumericalConfig ou_config;
    ou_config.K = 9;
    ou_config.grid_range = 3.0;
    ou_config.adaptive = false;
    ou_config.grid_method = scar::OuGridMethod::Dense;
    ou_config.n_threads = 2;
    const scar::OuGridFilterResult ou_result =
        scar::filter_ou_grid_emissions(
            scar::OuParams{0.9, 0.0, 1.0},
            scar::DoubleView{emissions.data(), emissions.size()},
            3,
            9,
            ou_config,
            scar::OuBackend::Matrix);
    if (!ou_result.is_ok()
        || ou_result.n_obs != 3
        || ou_result.K != 9
        || ou_result.smoothed_weights.size() != 3 * 9) {
        return 3;
    }

    scar::RVineDensityPlan plan;
    plan.dimension = 2;
    plan.node_count = 2;
    plan.input_nodes = {0, 1};
    plan.edge_indices = {0};
    plan.input1_nodes = {0};
    plan.input2_nodes = {1};
    plan.output1_nodes = {-1};
    plan.output2_nodes = {-1};
    plan.transposed = {0};
    plan.residual_nodes = {0, 1};
    plan.used_edges = {0};
    plan.affected_operation_offsets = {0, 1, 2};
    plan.affected_operations = {0, 0};
    plan.affected_node_offsets = {0, 1, 2};
    plan.affected_nodes = {0, 1};

    scar::rvine::EdgeSpec edge;
    edge.copula = independent;
    edge.parameter_source = scar::rvine::ParameterSource::None;
    edge.parameter_index = -1;
    edge.parameter_free = true;
    const std::vector<scar::rvine::EdgeSpec> edges{edge};
    const scar::rvine::ParameterPack parameters{
        scar::DoubleView{}, scar::DoubleView{}, 3, 0};
    const scar::rvine::DensityResult vine_result =
        scar::rvine::log_pdf_rows(
            plan,
            edges,
            parameters,
            scar::DoubleView{
                observation_values.data(), observation_values.size()},
            3,
            2,
            2);
    if (!vine_result.is_ok()
        || vine_result.log_pdf.size() != 3
        || !all_zero(vine_result.log_pdf)
        || vine_result.diagnostics.independence_fast_paths != 3) {
        return 4;
    }
    const scar::rvine::RosenblattResult vine_trace =
        scar::rvine::rosenblatt_transform(
            plan,
            edges,
            parameters,
            scar::DoubleView{
                observation_values.data(), observation_values.size()},
            3,
            2,
            2,
            true);
    if (!vine_trace.is_ok()
        || vine_trace.node_count != 2
        || vine_trace.node_values != observation_values) {
        return 14;
    }

    scar::CopulaSpec gumbel;
    gumbel.family = scar::CopulaFamily::Gumbel;
    const auto public_bounds =
        scar::model_public_parameter_bounds(gumbel);
    const auto student_policy =
        scar::student_fit_parameter_policy(3, false);
    const auto gas_default = scar::gas_default_initial_point(2.0);
    if (!public_bounds.is_ok()
        || public_bounds.value.lower != std::vector<double>{1.0001}
        || public_bounds.value.upper.size() != 1
        || !std::isinf(public_bounds.value.upper[0])
        || !student_policy.is_ok()
        || student_policy.value.initial != 5.0
        || student_policy.value.lower != 2.001
        || student_policy.value.upper != 10000.0
        || !gas_default.is_ok()
        || gas_default.value != std::vector<double>{0.1, 0.05, 0.95}) {
        return 8;
    }

    const auto optimizer_failure = scar::optimizer_failure_evaluation(
        {2.0, 1.0}, {1.0, 1.0}, 400.0, true);
    const auto sized_optimizer_failure =
        scar::optimizer_failure_evaluation_for_size(3, 123.0);
    const auto optimizer_failure_value = scar::optimizer_failure_objective();
    const auto optimizer_scale = scar::optimizer_unit_scale(
        {-4.0, 0.25, 2.0});
    const auto optimizer_projection = scar::project_optimizer_point(
        {-4.0, 0.25, 2.0},
        {-3.0, -std::numeric_limits<double>::infinity(), -1.0},
        {3.0, std::numeric_limits<double>::infinity(), 1.5});
    if (!optimizer_failure.is_ok()
        || optimizer_failure.value.objective != 400.0
        || optimizer_failure.value.gradient
            != std::vector<double>{20.0, 0.0}
        || !sized_optimizer_failure.is_ok()
        || sized_optimizer_failure.value.objective != 123.0
        || sized_optimizer_failure.value.gradient
            != std::vector<double>{0.0, 0.0, 0.0}
        || !optimizer_failure_value.is_ok()
        || optimizer_failure_value.value != 1e10
        || !optimizer_scale.is_ok()
        || optimizer_scale.value != std::vector<double>{4.0, 1.0, 2.0}
        || !optimizer_projection.is_ok()
        || optimizer_projection.value
            != std::vector<double>{-3.0, 0.25, 1.5}) {
        return 17;
    }

    const auto ou_kappa_dt = scar::ou_kappa_dt(2.0, 5);
    const auto ou_local = scar::ou_auto_backend(0.01, 11, 0.01);
    const auto ou_spectral = scar::ou_auto_backend(1.0, 11, 0.01);
    const auto ou_basis = scar::ou_adaptive_spectral_basis_order(0.01, 11);
    const auto ou_quad = scar::ou_resolve_quad_order(32, 0, false);
    if (!ou_kappa_dt.is_ok() || ou_kappa_dt.value != 0.5
        || !ou_local.is_ok() || ou_local.value != scar::OuBackend::LocalGh
        || !ou_spectral.is_ok()
        || ou_spectral.value != scar::OuBackend::Spectral
        || !ou_basis.is_ok() || ou_basis.value != 128
        || !ou_quad.is_ok() || ou_quad.value != 80) {
        return 18;
    }

    scar::RVineDensityPlan policy_plan;
    policy_plan.dimension = 1;
    policy_plan.node_count = 1;
    policy_plan.edge_indices.assign(100, 0);
    policy_plan.affected_operation_offsets = {0, 85};
    const auto mcmc_policy = scar::rvine::mcmc_policy(
        policy_plan,
        {0},
        2,
        true,
        scar::rvine::MCMCDensityAlgorithm::Auto,
        1024U * 1024U * 1024U);
    const auto mcmc_defaults = scar::rvine::mcmc_default_steps(4);
    if (!mcmc_policy.is_ok()
        || mcmc_policy.value.density_algorithm
            != scar::rvine::MCMCDensityAlgorithm::Incremental
        || !mcmc_defaults.is_ok()
        || mcmc_defaults.value.n_steps != 120
        || mcmc_defaults.value.burnin_steps != 40) {
        return 19;
    }

    const auto fixed_shape = scar::build_fixed_jacobi_shape_rule(
        2.0, 3.0, 16, 16U * 1024U * 1024U);
    const auto capped_basis = scar::resolve_jacobi_basis_order(128, 48);
    const auto horizon_steps = scar::jacobi_horizon_steps(11);
    if (!fixed_shape.is_ok()
        || fixed_shape.value.tau.size() != 16
        || !capped_basis.is_ok() || capped_basis.value != 48
        || !horizon_steps.is_ok() || horizon_steps.value != 10) {
        return 20;
    }

    const std::vector<double> ou_physical{2.0, 0.3, 4.0, 0.25};
    const auto ou_optimizer = scar::ou_to_log_stationary(ou_physical);
    const auto ou_recovered = scar::ou_from_log_stationary(ou_optimizer.value);
    const std::vector<double> ou_physical_gradient{1.0, 0.5, 3.0, 0.75};
    const auto ou_optimizer_gradient = scar::ou_gradient_to_log_stationary(
        ou_physical, ou_physical_gradient);
    const auto ou_recovered_gradient = scar::ou_gradient_from_log_stationary(
        ou_physical, ou_optimizer_gradient.value);
    const std::vector<double> coordinate_scale{2.0, 4.0, 0.5, 8.0};
    const auto ou_scaled = scar::physical_to_optimizer_scaled(
        ou_physical, coordinate_scale);
    const auto ou_unscaled = scar::optimizer_scaled_to_physical(
        ou_scaled.value, coordinate_scale);
    if (!ou_optimizer.is_ok() || !ou_recovered.is_ok()
        || !ou_optimizer_gradient.is_ok() || !ou_recovered_gradient.is_ok()
        || !ou_scaled.is_ok() || !ou_unscaled.is_ok()
        || std::abs(ou_optimizer.value[0] - std::log(2.0)) > 1e-15
        || std::abs(ou_optimizer.value[2] - std::log(2.0)) > 1e-15
        || ou_recovered.value != ou_physical
        || ou_optimizer_gradient.value
            != std::vector<double>{8.0, 0.5, 12.0, 0.75}
        || ou_recovered_gradient.value != ou_physical_gradient
        || ou_unscaled.value != ou_physical) {
        return 10;
    }

    scar::CopulaSpec equicorr;
    equicorr.family = scar::CopulaFamily::EquicorrGaussian;
    equicorr.transform = scar::Transform::GaussianTanh;
    equicorr.dim = 3;
    const scar::Observations equicorr_observations{
        {0.25, 0.50, 0.75},
        {0.60, 0.40, 0.30},
    };
    const scar::StaticCopulaEvaluator equicorr_model(
        equicorr, equicorr_observations, 2);
    const double equicorr_raw = 0.2;
    const double equicorr_parameter = scar_internal::equicorr_transform(
        equicorr, equicorr_raw);
    const double equicorr_derivative = scar_internal::equicorr_dtransform(
        equicorr, equicorr_raw);
    const auto equicorr_transformed =
        equicorr_model.transformed_objective(equicorr_raw);
    const auto equicorr_physical = equicorr_model.objective(
        equicorr_parameter);
    if (!equicorr_transformed.is_ok() || !equicorr_physical.is_ok()
        || std::abs(equicorr_transformed.negative_log_likelihood
            - equicorr_physical.negative_log_likelihood) > 1e-14
        || std::abs(equicorr_transformed.negative_gradient
            - equicorr_physical.negative_gradient * equicorr_derivative)
            > 1e-14) {
        return 11;
    }

    const auto heuristic =
        scar::ou_heuristic_initial_point(21, 0.4, 0.95, 0.3);
    const auto jacobi_initial =
        scar::jacobi_initial_point(1.0, true, 1e-6);
    const auto gas_initial = gas.ou_initial_point(
        0.2, independent, observation_view, scar::GasConfig{});
    if (!heuristic.is_ok()
        || std::abs(
            heuristic.value.params.kappa + 20.0 * std::log(0.95)) > 1e-14
        || heuristic.value.params.mu != 0.4
        || !jacobi_initial.is_ok()
        || jacobi_initial.value.kappa != 1.0
        || jacobi_initial.value.m != 1.0 - 1e-6
        || jacobi_initial.value.xi != 0.2
        || !gas_initial.is_ok()
        || !gas_initial.grid_candidate_found
        || std::abs(gas_initial.mu - 0.2) > 1e-15) {
        return 9;
    }

    const double clip_input[] = {-1.0, 0.5, 2.0};
    const auto clipped = scar::clip_open_unit(
        scar::DoubleView{clip_input, 3}, 1e-10);
    const double fit_input[] = {0.2, 0.4, 0.2, 0.7};
    const auto fit_validation = scar::validate_fit_observations(
        scar::DoubleView{fit_input, 4}, 2, 2);
    const double invalid_sum[] = {10.0};
    const double invalid_sum2[] = {1.0};
    const auto prepared_validation =
        scar::validate_equicorr_prepared_statistics(
            scar::DoubleView{invalid_sum, 1},
            scar::DoubleView{invalid_sum2, 1},
            3,
            1e-10);
    if (!clipped.is_ok()
        || clipped.values != std::vector<double>{1e-10, 0.5, 1.0 - 1e-10}
        || fit_validation.code
            != scar::NumericalValidationCode::ConstantColumn
        || prepared_validation.code
            != scar::NumericalValidationCode::CauchyBound) {
        return 12;
    }

    const double final_parameters[] = {1.0, 0.0, 1.0};
    const double final_lower[] = {0.001, -10.0, 0.001};
    const double final_upper[] = {10.0, 10.0, 10.0};
    const double final_gradient[] = {0.0, 0.0, 0.0};
    const auto final_validation = scar::validate_ou_final_fit(
        scar::DoubleView{final_parameters, 3},
        scar::DoubleView{final_parameters, 3},
        scar::DoubleView{final_lower, 3},
        scar::DoubleView{final_upper, 3},
        2.0,
        2.0,
        scar::DoubleView{final_gradient, 3},
        true,
        "cpp",
        "",
        100,
        true,
        false,
        0.0,
        1e-3,
        1e-10,
        1e5);
    if (!final_validation.reasons.empty()
        || !final_validation.has_ou_diagnostics
        || !final_validation.has_projected_gradient
        || final_validation.projected_gradient_norm != 0.0) {
        return 13;
    }
    return 0;
}
