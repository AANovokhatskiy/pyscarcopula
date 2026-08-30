#include "scar/copula/model_statistics.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/dynamic_rvine.hpp"
#include "scar/gas_rvine.hpp"
#include "scar/model_policy.hpp"
#include "scar/rvine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

scar::DoubleView view(const std::vector<double>& values) {
    return {values.data(), values.size()};
}

bool finite_values(const std::vector<double>& values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](double value) { return std::isfinite(value); });
}

bool close_vectors(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double tolerance = 2e-13) {

    if (first.size() != second.size()) {
        return false;
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (!std::isfinite(first[index]) || !std::isfinite(second[index])
            || std::abs(first[index] - second[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

scar::RVineDensityPlan c_vine_density_plan() {
    scar::RVineDensityPlan plan;
    plan.dimension = 4;
    plan.node_count = 16;
    plan.input_nodes = {0, 1, 2, 3};
    plan.edge_indices = {0, 1, 2, 3, 4, 5};
    plan.input1_nodes = {0, 0, 0, 7, 7, 11};
    plan.input2_nodes = {2, 1, 3, 5, 9, 13};
    plan.output1_nodes = {4, 6, 8, 10, 12, 14};
    plan.output2_nodes = {5, 7, 9, 11, 13, 15};
    plan.transposed = {0, 0, 0, 0, 0, 0};
    plan.residual_nodes = {14, 12, 8, 3};
    plan.used_edges = {0, 1, 2, 3, 4, 5};
    plan.affected_operation_offsets = {0, 6, 10, 13, 16};
    plan.affected_operations = {
        0, 1, 2, 3, 4, 5,
        1, 3, 4, 5,
        0, 3, 5,
        2, 4, 5,
    };
    plan.affected_node_offsets = {0, 13, 22, 29, 36};
    plan.affected_nodes = {
        0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        1, 6, 7, 10, 11, 12, 13, 14, 15,
        2, 4, 5, 10, 11, 14, 15,
        3, 8, 9, 12, 13, 14, 15,
    };
    return plan;
}

scar::RVineDensityPlan d_vine_density_plan() {
    scar::RVineDensityPlan plan;
    plan.dimension = 4;
    plan.node_count = 16;
    plan.input_nodes = {0, 1, 2, 3};
    plan.edge_indices = {0, 1, 2, 3, 4, 5};
    plan.input1_nodes = {0, 1, 2, 4, 6, 10};
    plan.input2_nodes = {1, 2, 3, 7, 9, 13};
    plan.output1_nodes = {4, 6, 8, 10, 12, 14};
    plan.output2_nodes = {5, 7, 9, 11, 13, 15};
    plan.transposed = {0, 0, 0, 0, 0, 0};
    plan.residual_nodes = {14, 12, 8, 3};
    plan.used_edges = {0, 1, 2, 3, 4, 5};
    plan.affected_operation_offsets = {0, 3, 8, 13, 16};
    plan.affected_operations = {
        0, 3, 5,
        0, 1, 3, 4, 5,
        1, 2, 3, 4, 5,
        2, 4, 5,
    };
    plan.affected_node_offsets = {0, 7, 18, 29, 36};
    plan.affected_nodes = {
        0, 4, 5, 10, 11, 14, 15,
        1, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15,
        2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        3, 8, 9, 12, 13, 14, 15,
    };
    return plan;
}

scar::RVineDensityPlan r_vine_density_plan() {
    scar::RVineDensityPlan plan;
    plan.dimension = 3;
    plan.node_count = 9;
    plan.input_nodes = {0, 1, 2};
    plan.edge_indices = {1, 0, 2};
    plan.input1_nodes = {1, 0, 5};
    plan.input2_nodes = {2, 1, 4};
    plan.output1_nodes = {3, 5, 7};
    plan.output2_nodes = {4, 6, 8};
    plan.transposed = {0, 0, 0};
    plan.residual_nodes = {7, 3, 2};
    plan.used_edges = {0, 1, 2};
    plan.affected_operation_offsets = {0, 2, 5, 7};
    plan.affected_operations = {1, 2, 0, 1, 2, 0, 2};
    plan.affected_node_offsets = {0, 5, 12, 17};
    plan.affected_nodes = {
        0, 5, 6, 7, 8,
        1, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 7, 8,
    };
    return plan;
}

scar::RVineTraversalPlan r_vine_traversal_plan() {
    scar::RVineTraversalPlan plan;
    plan.dimension = 3;
    plan.node_count = 7;
    plan.last_uniform_column = 2;
    plan.last_output_node = 0;
    plan.output_nodes = {4, 1, 0};
    plan.column_uniforms = {1, 0};
    plan.inverse_offsets = {0, 1, 2};
    plan.inverse_edges = {0, 1};
    plan.inverse_partner_nodes = {0, 1};
    plan.inverse_output_nodes = {1, 4};
    plan.inverse_transposed = {0, 0};
    plan.forward_offsets = {0, 1, 2};
    plan.forward_edges = {0, 1};
    plan.forward_leaf_nodes = {1, 4};
    plan.forward_partner_nodes = {0, 1};
    plan.forward_leaf_output_nodes = {2, 5};
    plan.forward_partner_output_nodes = {3, 6};
    plan.forward_transposed = {0, 0};
    plan.update_u1_nodes = {1, 4};
    plan.update_u2_nodes = {0, 1};
    return plan;
}

scar::RVineConditionalPlan suffix_conditional_plan() {
    scar::RVineConditionalPlan plan;
    plan.dimension = 3;
    plan.node_count = 11;
    plan.given_variables = {2};
    plan.given_nodes = {0};
    plan.uniform_nodes = {1, 5};
    plan.node_sources = {1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0};
    plan.node_source_indices = {2, 1, -1, -1, -1, 0, -1, -1, -1, -1, -1};
    plan.opcodes = {3, 2, 3, 3, 2, 2};
    plan.edge_indices = {1, 1, 2, 0, 0, 2};
    plan.input1_nodes = {1, 2, 5, 6, 7, 6};
    plan.input2_nodes = {0, 0, 4, 2, 2, 4};
    plan.output1_nodes = {2, 3, 6, 7, 6, 9};
    plan.output2_nodes = {-1, 4, -1, -1, 8, 10};
    plan.transposed = {0, 0, 0, 0, 0, 0};
    plan.output_nodes = {7, 2, 0};
    plan.used_edges = {0, 1, 2};
    return plan;
}

scar::RVineConditionalPlan dag_conditional_plan() {
    scar::RVineConditionalPlan plan;
    plan.dimension = 3;
    plan.node_count = 11;
    plan.given_variables = {1};
    plan.given_nodes = {0};
    plan.uniform_nodes = {1, 5};
    plan.node_sources = {1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0};
    plan.node_source_indices = {1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1};
    plan.opcodes = {3, 1, 1, 3, 3, 1, 1, 1};
    plan.edge_indices = {1, 1, 1, 2, 0, 0, 2, 2};
    plan.input1_nodes = {1, 0, 2, 5, 6, 0, 6, 4};
    plan.input2_nodes = {0, 2, 0, 4, 0, 7, 4, 6};
    plan.output1_nodes = {2, 3, 4, 6, 7, 8, 9, 10};
    plan.output2_nodes = {-1, -1, -1, -1, -1, -1, -1, -1};
    plan.transposed = {0, 0, 0, 0, 0, 0, 0, 0};
    plan.output_nodes = {7, 0, 2};
    plan.used_edges = {0, 1, 2};
    return plan;
}

scar::rvine::EdgeSpec static_edge(
    scar::CopulaFamily family,
    scar::Rotation rotation,
    scar::rvine::ParameterSource source,
    int parameter_index,
    bool parameter_free) {

    scar::rvine::EdgeSpec edge;
    edge.copula = scar::default_pair_copula_spec(family);
    edge.copula.rotation = rotation;
    edge.parameter_source = source;
    edge.parameter_index = parameter_index;
    edge.parameter_free = parameter_free;
    return edge;
}

std::vector<scar::rvine::EdgeSpec> independent_edges(std::size_t count) {
    std::vector<scar::rvine::EdgeSpec> edges;
    edges.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        edges.push_back(static_edge(
            scar::CopulaFamily::Independent,
            scar::Rotation::R0,
            scar::rvine::ParameterSource::None,
            -1,
            true));
    }
    return edges;
}

scar::JacobiEvaluatorConfig jacobi_config() {
    scar::JacobiEvaluatorConfig config;
    config.transition.method = scar::JacobiTransitionMethod::LocalFixed;
    config.transition.storage = scar::JacobiTransitionStorage::Dense;
    config.transition.numerical.n_obs = 5;
    config.transition.numerical.quad_order = 16;
    config.transition.numerical.basis_order = 4;
    config.transition.numerical.gh_order = 3;
    config.transition.numerical.matrix = true;
    config.transition.numerical.gradient = true;
    config.transition.derivatives = true;
    return config;
}

}  // namespace

int run_vine_model_tests() {
    const scar::RVineDensityPlan c_plan = c_vine_density_plan();
    const scar::RVineDensityPlan d_plan = d_vine_density_plan();
    const scar::RVineDensityPlan r_plan = r_vine_density_plan();
    const auto six_independent = independent_edges(6);
    const auto three_independent = independent_edges(3);
    if (!scar::rvine::validate_density_plan(c_plan, six_independent.size())
        || !scar::rvine::validate_density_plan(d_plan, six_independent.size())
        || !scar::rvine::validate_density_plan(r_plan, three_independent.size())) {
        return 1;
    }
    scar::RVineDensityPlan invalid_plan = r_plan;
    invalid_plan.transposed[0] = 2;
    if (scar::rvine::validate_density_plan(
            invalid_plan, three_independent.size())) {
        return 2;
    }

    const std::vector<double> four_dimensional{
        0.20, 0.45, 0.70, 0.30,
        0.35, 0.60, 0.25, 0.80,
        0.80, 0.30, 0.55, 0.40,
        0.65, 0.75, 0.40, 0.15,
        0.15, 0.50, 0.85, 0.60,
    };
    const scar::rvine::ParameterPack empty_four{{}, {}, 5, 0};
    const auto c_density = scar::rvine::log_pdf_rows(
        c_plan,
        six_independent,
        empty_four,
        view(four_dimensional),
        5,
        4,
        2);
    const auto d_density = scar::rvine::log_pdf_rows(
        d_plan,
        six_independent,
        empty_four,
        view(four_dimensional),
        5,
        4,
        2);
    if (!c_density.is_ok() || !d_density.is_ok()
        || c_density.log_pdf != std::vector<double>(5, 0.0)
        || d_density.log_pdf != std::vector<double>(5, 0.0)
        || c_density.diagnostics.independence_fast_paths != 30
        || d_density.diagnostics.independence_fast_paths != 30) {
        return 3;
    }

    // Mixed-family static R-vine with both scalar and row-path parameters.
    std::vector<scar::rvine::EdgeSpec> edges{
        static_edge(
            scar::CopulaFamily::Clayton,
            scar::Rotation::R90,
            scar::rvine::ParameterSource::RowPath,
            0,
            false),
        static_edge(
            scar::CopulaFamily::Gaussian,
            scar::Rotation::R0,
            scar::rvine::ParameterSource::Scalar,
            0,
            false),
        static_edge(
            scar::CopulaFamily::Independent,
            scar::Rotation::R0,
            scar::rvine::ParameterSource::None,
            -1,
            true),
    };
    const std::vector<double> scalar_parameters{-0.35};
    const std::vector<double> row_parameters{0.70, 0.75, 0.80, 0.85, 0.90};
    const scar::rvine::ParameterPack parameters{
        view(scalar_parameters), view(row_parameters), 5, 1};
    const std::vector<double> observations{
        0.20, 0.45, 0.70,
        0.35, 0.60, 0.25,
        0.80, 0.30, 0.55,
        0.65, 0.75, 0.40,
        0.15, 0.50, 0.85,
    };
    const auto density_one = scar::rvine::log_pdf_rows(
        r_plan, edges, parameters, view(observations), 5, 3, 1);
    const auto density_two = scar::rvine::log_pdf_rows(
        r_plan, edges, parameters, view(observations), 5, 3, 2);
    const auto residual = scar::rvine::rosenblatt_transform(
        r_plan, edges, parameters, view(observations), 5, 3, 2, true);
    if (!density_one.is_ok() || !density_two.is_ok() || !residual.is_ok()) {
        return 4;
    }
    if (density_one.log_pdf.size() != 5
        || residual.residuals.size() != observations.size()
        || residual.node_values.size() != 5 * 9) {
        return 12;
    }
    if (!finite_values(density_one.log_pdf)
        || !finite_values(residual.residuals)
        || !finite_values(residual.node_values)) {
        return 13;
    }
    if (!close_vectors(density_one.log_pdf, density_two.log_pdf)) {
        return 14;
    }
    // Generic unconditional traversal and the GAS-specialized stateful path.
    const scar::RVineTraversalPlan traversal = r_vine_traversal_plan();
    std::vector<scar::rvine::EdgeSpec> traversal_edges{
        edges[0], edges[1],
    };
    traversal_edges[0].parameter_source = scar::rvine::ParameterSource::Scalar;
    traversal_edges[0].parameter_index = 0;
    traversal_edges[1].parameter_index = 1;
    const std::vector<double> traversal_parameters{0.8, -0.35};
    const scar::rvine::ParameterPack traversal_pack{
        view(traversal_parameters), {}, 5, 0};
    const auto sample_one = scar::rvine::sample(
        traversal,
        traversal_edges,
        traversal_pack,
        view(observations),
        5,
        3,
        1);
    const auto sample_two = scar::rvine::sample(
        traversal,
        traversal_edges,
        traversal_pack,
        view(observations),
        5,
        3,
        2);
    if (!scar::rvine::validate_traversal_plan(
            traversal, traversal_edges.size())
        || !sample_one.is_ok() || !sample_two.is_ok()
        || sample_one.values.size() != observations.size()
        || !finite_values(sample_one.values)
        || !close_vectors(sample_one.values, sample_two.values)) {
        return 5;
    }

    std::vector<scar::GasRvineEdge> gas_edges(2);
    gas_edges[0].copula = traversal_edges[0].copula;
    gas_edges[0].gas_params = {0.05, 0.10, 0.70};
    gas_edges[0].dynamic = true;
    gas_edges[1].copula = traversal_edges[1].copula;
    gas_edges[1].dynamic = false;
    std::vector<double> gas_parameter_paths(5 * 2, 0.0);
    for (std::size_t row = 0; row < 5; ++row) {
        gas_parameter_paths[row * 2] = 0.8;
        gas_parameter_paths[row * 2 + 1] = -0.35;
    }
    const auto gas_sample = scar::gas_rvine_sample(
        gas_edges,
        traversal,
        observations.data(),
        5,
        3,
        gas_parameter_paths.data(),
        5,
        2);
    if (!gas_sample.is_ok()
        || gas_sample.values.size() != observations.size()
        || !finite_values(gas_sample.values)) {
        return 6;
    }

    // One native composition executes GAS, OU, and Jacobi edges in the same
    // R-vine Rosenblatt traversal.
    std::vector<scar::DynamicRvineEdge> dynamic_edges(3);
    dynamic_edges[0].edge = static_edge(
        scar::CopulaFamily::Clayton,
        scar::Rotation::R0,
        scar::rvine::ParameterSource::None,
        -1,
        false);
    dynamic_edges[0].dynamics = scar::DynamicRvineKind::Gas;
    dynamic_edges[0].gas_params = {0.05, 0.10, 0.70};
    dynamic_edges[1].edge = static_edge(
        scar::CopulaFamily::Gaussian,
        scar::Rotation::R0,
        scar::rvine::ParameterSource::None,
        -1,
        false);
    dynamic_edges[1].dynamics = scar::DynamicRvineKind::ScarOu;
    dynamic_edges[1].ou_params = {1.1, 0.1, 0.7};
    dynamic_edges[1].ou_config.K = 17;
    dynamic_edges[1].ou_config.max_K = 17;
    dynamic_edges[1].ou_config.adaptive = false;
    dynamic_edges[1].ou_config.grid_method = scar::OuGridMethod::Dense;
    dynamic_edges[1].ou_method = "matrix";
    dynamic_edges[2].edge = static_edge(
        scar::CopulaFamily::Gumbel,
        scar::Rotation::R180,
        scar::rvine::ParameterSource::None,
        -1,
        false);
    dynamic_edges[2].dynamics = scar::DynamicRvineKind::ScarJacobi;
    dynamic_edges[2].jacobi_params = {1.2, 0.4, 0.25};
    dynamic_edges[2].jacobi_config = jacobi_config();
    const scar::rvine::ParameterPack dynamic_parameters{{}, {}, 5, 0};
    const auto dynamic = scar::dynamic_rvine_rosenblatt_transform(
        r_plan,
        dynamic_edges,
        dynamic_parameters,
        view(observations),
        5,
        3,
        2,
        true);
    if (!dynamic.is_ok()
        || dynamic.residuals.size() != observations.size()
        || dynamic.node_values.size() != 5 * 9
        || !std::isfinite(dynamic.log_likelihood)
        || !finite_values(dynamic.residuals)
        || !finite_values(dynamic.node_values)
        || dynamic.h_pair_operations != 15) {
        return 7;
    }

    // Suffix and arbitrary-DAG conditional programs use the same edge pack.
    const scar::RVineConditionalPlan suffix = suffix_conditional_plan();
    const scar::RVineConditionalPlan dag = dag_conditional_plan();
    const std::vector<double> suffix_given{0.5, 0.5, 0.4};
    const std::vector<double> dag_given{0.5, 0.4, 0.5};
    const auto suffix_result = scar::rvine::conditional_sample(
        suffix,
        edges,
        parameters,
        view(suffix_given),
        view(observations),
        5,
        3,
        2,
        true);
    const auto dag_result = scar::rvine::conditional_sample(
        dag,
        edges,
        parameters,
        view(dag_given),
        view(observations),
        5,
        3,
        2,
        false);
    if (!scar::rvine::validate_conditional_plan(suffix, edges.size())
        || !scar::rvine::validate_conditional_plan(dag, edges.size())
        || !suffix_result.is_ok() || !dag_result.is_ok()
        || suffix_result.values.size() != observations.size()
        || dag_result.values.size() != observations.size()
        || suffix_result.operation_inputs.size() != 6 * 5 * 2
        || !finite_values(suffix_result.values)
        || !finite_values(dag_result.values)
        || !finite_values(suffix_result.operation_inputs)) {
        return 8;
    }

    // Arbitrary conditional MCMC has exact fixed-draw consumption and chunk
    // continuation, for both full and incremental density algorithms.
    const std::vector<double> mcmc_row_parameters{0.78, 0.84};
    const scar::rvine::ParameterPack mcmc_parameters{
        view(scalar_parameters), view(mcmc_row_parameters), 2, 1};
    const std::vector<double> state{
        0.25, 0.40, 0.70,
        0.65, 0.40, 0.30,
    };
    const auto initial_density = scar::rvine::log_pdf_rows(
        r_plan, edges, mcmc_parameters, view(state), 2, 3, 1);
    const std::vector<double> given{0.4};
    const std::vector<int> given_indices{1};
    const std::vector<int> free_indices{0, 2};
    const std::vector<double> proposals{
        0.10, 0.90,
        0.20, 0.80,
        0.30, 0.70,
        0.40, 0.60,
    };
    const std::vector<double> acceptance{
        0.01, 0.99,
        0.25, 0.75,
        0.50, 0.50,
        0.90, 0.10,
    };
    const auto full = scar::rvine::mcmc_chunk(
        r_plan,
        edges,
        mcmc_parameters,
        given_indices,
        view(given),
        free_indices,
        view(state),
        2,
        3,
        view(initial_density.log_pdf),
        0,
        view(proposals),
        4,
        2,
        view(acceptance),
        4,
        2,
        2,
        scar::rvine::MCMCDensityAlgorithm::Incremental);
    const auto full_recompute = scar::rvine::mcmc_chunk(
        r_plan,
        edges,
        mcmc_parameters,
        given_indices,
        view(given),
        free_indices,
        view(state),
        2,
        3,
        view(initial_density.log_pdf),
        0,
        view(proposals),
        4,
        2,
        view(acceptance),
        4,
        2,
        1,
        scar::rvine::MCMCDensityAlgorithm::FullRecompute);
    const auto first_chunk = scar::rvine::mcmc_chunk(
        r_plan,
        edges,
        mcmc_parameters,
        given_indices,
        view(given),
        free_indices,
        view(state),
        2,
        3,
        view(initial_density.log_pdf),
        0,
        {proposals.data(), 4},
        2,
        2,
        {acceptance.data(), 4},
        2,
        2,
        1,
        scar::rvine::MCMCDensityAlgorithm::Incremental);
    const auto second_chunk = scar::rvine::mcmc_chunk(
        r_plan,
        edges,
        mcmc_parameters,
        given_indices,
        view(given),
        free_indices,
        view(first_chunk.state),
        2,
        3,
        view(first_chunk.log_pdf),
        2,
        {proposals.data() + 4, 4},
        2,
        2,
        {acceptance.data() + 4, 4},
        2,
        2,
        1,
        scar::rvine::MCMCDensityAlgorithm::Incremental);
    if (!initial_density.is_ok()) {
        return 9;
    }
    if (!full.is_ok()) {
        return 16;
    }
    if (!full_recompute.is_ok()) {
        return 17;
    }
    if (!first_chunk.is_ok()) {
        return 18;
    }
    if (!second_chunk.is_ok()) {
        return 19;
    }
    if (full.proposal_draws_used != 8
        || full.acceptance_draws_used != 8
        || first_chunk.proposal_draws_used != 4
        || second_chunk.proposal_draws_used != 4) {
        return 20;
    }
    if (!close_vectors(full.state, full_recompute.state)
        || !close_vectors(full.log_pdf, full_recompute.log_pdf)) {
        return 21;
    }
    if (!close_vectors(full.state, second_chunk.state)
        || !close_vectors(full.log_pdf, second_chunk.log_pdf)) {
        return 22;
    }

    // Native selection primitives: pair association, candidate likelihood
    // accumulation, and information-score reduction.
    const std::vector<double> first{1.0, 1.0, 2.0, 3.0};
    const std::vector<double> second{1.0, 2.0, 2.0, 3.0};
    const auto tau = scar::kendall_tau(view(first), view(second));
    const auto score = scar::add_scores(1.25, -0.5);
    const auto bic = scar::information_criterion(
        -12.5, 3, 100, scar::InformationCriterion::Bic);
    if (!tau.is_ok() || std::abs(tau.value - 0.8) > 1e-15
        || !score.is_ok() || score.value != 0.75
        || !bic.is_ok()
        || std::abs(bic.value - (25.0 + 3.0 * std::log(100.0))) > 1e-15) {
        return 10;
    }

    // Exact Kendall limits must reach finite native MLE initialization.
    const auto gaussian_spec = scar::default_pair_copula_spec(
        scar::CopulaFamily::Gaussian);
    for (double endpoint : {-1.0, 1.0}) {
        const auto signed_tau = scar::tau_for_itau(endpoint, true);
        const auto positive_tau = scar::tau_for_itau(endpoint, false);
        const auto initial = scar::pair_mle_initial_parameter(
            gaussian_spec, endpoint);
        if (!signed_tau.is_ok() || signed_tau.value != endpoint
            || !positive_tau.is_ok() || positive_tau.value != 1.0
            || !initial.is_ok() || initial.value != endpoint * 0.9999) {
            return 23;
        }
    }
    for (const auto family : {scar::CopulaFamily::Clayton, scar::CopulaFamily::Frank,
                             scar::CopulaFamily::Gumbel, scar::CopulaFamily::Joe}) {
        auto spec = scar::default_pair_copula_spec(family);
        const auto initial = scar::pair_mle_initial_parameter(spec, 1.0);
        spec.transform = scar::Transform::Logistic;
        const auto bounded = scar::pair_mle_initial_parameter(spec, 1.0);
        const auto domain = scar::model_public_parameter_bounds(spec);
        if (initial.status != scar::Status::NumericalFailure || !bounded.is_ok()
            || !domain.is_ok()
            || bounded.value != domain.value.upper[0]) {
            return 24;
        }
    }
    for (double invalid_tau : {1.001, -1.001,
                              std::numeric_limits<double>::quiet_NaN()}) {
        if (scar::pair_mle_initial_parameter(gaussian_spec, invalid_tau).is_ok()) {
            return 25;
        }
    }

    scar::rvine::EdgeSpec unsupported;
    unsupported.copula.family = scar::CopulaFamily::Student;
    unsupported.copula.dim = 3;
    std::vector<scar::rvine::PreparedEdge> prepared;
    if (scar::rvine::prepare_edges({unsupported}, prepared)
        == scar::SCAR_OK) {
        return 11;
    }
    return 0;
}
