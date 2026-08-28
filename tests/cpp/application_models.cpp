#include "scar/copula.hpp"
#include "scar/gas.hpp"
#include "scar/ou.hpp"
#include "scar/rvine.hpp"

#include <cmath>
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
    return 0;
}
