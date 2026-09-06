#include "module.hpp"

#include "array.hpp"
#include "scar/jacobi.hpp"

#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

template <typename ResultType>
py::dict status_dict(const ResultType& result) {
    py::dict output;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_row"] = result.failure.row;
    output["failure_coordinate"] = result.failure.coordinate;
    output["failure_operation"] = result.failure.operation;
    return output;
}

py::dict transition_diagnostics_to_dict(
    const scar::JacobiTransitionDiagnostics& value) {

    py::dict output;
    output["dt"] = value.dt;
    output["alpha"] = value.alpha;
    output["beta"] = value.beta;
    output["raw_min_entry"] = value.raw_min_entry;
    output["raw_negative_mass"] = value.raw_negative_mass;
    output["min_entry"] = value.min_entry;
    output["max_row_sum_error_before_normalization"] =
        value.max_row_sum_error_before_normalization;
    output["max_row_sum_error"] = value.max_row_sum_error;
    output["stationary_error"] = value.stationary_error;
    output["probability_cleanup_negative_mass"] =
        value.probability_cleanup_negative_mass;
    output["probability_min_entry_before_cleanup"] =
        value.probability_min_entry_before_cleanup;
    output["clipped_negative"] = value.clipped_negative;
    output["probability_cleanup_applied"] =
        value.probability_cleanup_applied;
    output["gh_order"] = value.gh_order;
    output["method_requested"] = static_cast<int>(value.method_requested);
    output["method_used"] = static_cast<int>(value.method_used);
    output["storage"] = static_cast<int>(value.storage);
    output["correction"] = static_cast<int>(value.correction);
    output["spectral_status"] = static_cast<int>(value.spectral_status);
    output["nnz"] = value.nnz;
    output["max_width"] = value.max_width;
    output["retained_bytes"] = value.retained_bytes;
    output["dense_bytes"] = value.dense_bytes;
    output["estimated_workspace_bytes"] =
        value.estimated_workspace_bytes;
    output["memory_budget_bytes"] = value.memory_budget_bytes;
    output["mean_accepted_off_diagonal_mass"] =
        value.mean_accepted_off_diagonal_mass;
    output["mean_proposed_off_diagonal_mass"] =
        value.mean_proposed_off_diagonal_mass;
    output["acceptance_mass_ratio"] = value.acceptance_mass_ratio;
    output["min_row_acceptance_ratio"] =
        value.min_row_acceptance_ratio;
    output["mean_stay_probability"] = value.mean_stay_probability;
    output["max_stay_probability"] = value.max_stay_probability;
    output["reverse_missing_edge_fraction"] =
        value.reverse_missing_edge_fraction;
    output["detailed_balance_error"] = value.detailed_balance_error;
    output["ipfp_iterations"] = value.ipfp_iterations;
    output["ipfp_stationary_residual"] = value.ipfp_stationary_residual;
    output["ipfp_kl_divergence"] = value.ipfp_kl_divergence;
    output["ipfp_max_probability_change"] =
        value.ipfp_max_probability_change;
    return output;
}

py::dict trajectory_to_dict(const scar::JacobiTrajectoryResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["draws_used"] = result.value.draws_used;
    output["normal_draws_used"] = result.value.normal_draws_used;
    output["euler_steps"] = result.value.euler_steps;
    output["boundary_interventions"] =
        result.value.boundary_interventions;
    output["final_lamperti_value"] = result.value.final_lamperti_value;
    output["diagnostics"] = transition_diagnostics_to_dict(
        result.value.transition);
    return output;
}

py::dict state_sample_to_dict(const scar::JacobiStateSampleResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["parameters"] = vector_to_array(result.value.parameters);
    output["selection_draws_used"] = result.value.selection_draws_used;
    output["jitter_draws_used"] = result.value.jitter_draws_used;
    return output;
}

py::dict histogram_cells_to_dict(
    const scar::JacobiHistogramCellsResult& result) {

    py::dict output = status_dict(result);
    output["left"] = vector_to_array(result.value.left);
    output["right"] = vector_to_array(result.value.right);
    return output;
}

}  // namespace

void bind_jacobi_sampling(py::module_& m) {
    py::enum_<scar::JacobiStateSamplingMode>(m, "JacobiStateSamplingMode")
        .value("Grid", scar::JacobiStateSamplingMode::Grid)
        .value("Histogram", scar::JacobiStateSamplingMode::Histogram);

    py::class_<scar::JacobiLampertiSamplingConfig>(
            m, "JacobiLampertiSamplingConfig")
        .def(py::init<>())
        .def_readwrite(
            "n_obs", &scar::JacobiLampertiSamplingConfig::n_obs)
        .def_readwrite(
            "substeps", &scar::JacobiLampertiSamplingConfig::substeps)
        .def_readwrite(
            "interior_eps",
            &scar::JacobiLampertiSamplingConfig::interior_eps)
        .def_readwrite(
            "boundary", &scar::JacobiLampertiSamplingConfig::boundary);

    m.def(
        "jacobi_sample_grid_trajectory",
        [](const scar::JacobiParams& params,
           const scar::JacobiTransitionConfig& config,
           Float64Array uniforms) {
            const std::vector<double> draws = flat_vector_from_array(
                uniforms, "uniforms");
            scar::JacobiTrajectoryResult result;
            { py::gil_scoped_release release;
              result = scar::sample_jacobi_grid_trajectory(
                  params, config, draws); }
            return trajectory_to_dict(result);
        },
        py::arg("params"),
        py::arg("config"),
        py::arg("uniforms"));

    m.def(
        "jacobi_sample_prepared_sparse_trajectory",
        [](Float64Array tau,
           Float64Array weights,
           py::array_t<std::int64_t,
               py::array::c_style | py::array::forcecast> indices,
           Float64Array probabilities,
           py::array_t<std::int64_t,
               py::array::c_style | py::array::forcecast> counts,
           Float64Array uniforms) {
            const py::buffer_info index_info = indices.request();
            const py::buffer_info probability_info = probabilities.request();
            const py::buffer_info count_info = counts.request();
            if (index_info.ndim != 2
                || probability_info.ndim != 2
                || count_info.ndim != 1
                || index_info.shape != probability_info.shape
                || count_info.shape[0] != index_info.shape[0]) {
                throw std::invalid_argument(
                    "sparse transition arrays have inconsistent shapes");
            }
            scar::JacobiSparseTransition transition;
            transition.order = static_cast<int>(index_info.shape[0]);
            transition.max_width = static_cast<int>(index_info.shape[1]);
            const std::size_t size = static_cast<std::size_t>(
                index_info.shape[0] * index_info.shape[1]);
            const auto* index_data = static_cast<const std::int64_t*>(
                index_info.ptr);
            const auto* probability_data = static_cast<const double*>(
                probability_info.ptr);
            const auto* count_data = static_cast<const std::int64_t*>(
                count_info.ptr);
            transition.indices.assign(index_data, index_data + size);
            transition.probabilities.assign(
                probability_data, probability_data + size);
            transition.counts.assign(
                count_data,
                count_data + static_cast<std::size_t>(transition.order));
            const std::vector<double> tau_values = flat_vector_from_array(
                tau, "tau");
            const std::vector<double> weight_values = flat_vector_from_array(
                weights, "weights");
            const std::vector<double> draws = flat_vector_from_array(
                uniforms, "uniforms");
            scar::JacobiTrajectoryResult result;
            { py::gil_scoped_release release;
              result = scar::sample_prepared_jacobi_sparse_trajectory(
                  tau_values, weight_values, transition, draws); }
            return trajectory_to_dict(result);
        },
        py::arg("tau"),
        py::arg("weights"),
        py::arg("indices"),
        py::arg("probabilities"),
        py::arg("counts"),
        py::arg("uniforms"));

    m.def(
        "jacobi_sample_lamperti_chunk",
        [](const scar::JacobiParams& params,
           const scar::JacobiLampertiSamplingConfig& config,
           double initial_lamperti_value,
           Float64Array normal_draws) {
            const std::vector<double> draws = flat_vector_from_array(
                normal_draws, "normal_draws");
            scar::JacobiTrajectoryResult result;
            { py::gil_scoped_release release;
              result = scar::sample_jacobi_lamperti_chunk(
                  params, config, initial_lamperti_value, draws); }
            return trajectory_to_dict(result);
        },
        py::arg("params"),
        py::arg("config"),
        py::arg("initial_lamperti_value"),
        py::arg("normal_draws"));

    m.def(
        "jacobi_sample_state_distribution",
        [](const scar::CopulaSpec& copula,
           py::handle tau_input,
           py::handle probability_input,
           py::handle selection_input,
           py::handle jitter_input,
           scar::JacobiStateSamplingMode mode,
           double theta_cap) {
            const Float64Array tau = real_float64_array_from_object(
                tau_input, "tau");
            const Float64Array probability = real_float64_array_from_object(
                probability_input, "probability");
            const Float64Array selection_draws = real_float64_array_from_object(
                selection_input, "selection_draws");
            const Float64Array jitter_draws = real_float64_array_from_object(
                jitter_input, "jitter_draws");
            const std::vector<double> tau_values = flat_vector_from_array(
                tau, "tau");
            const std::vector<double> probability_values =
                flat_vector_from_array(probability, "probability");
            const std::vector<double> selection_values =
                flat_vector_from_array(selection_draws, "selection_draws");
            const std::vector<double> jitter_values = flat_vector_from_array(
                jitter_draws, "jitter_draws");
            scar::JacobiStateSampleResult result;
            { py::gil_scoped_release release;
              result = scar::sample_jacobi_state_distribution(
                  copula,
                  tau_values,
                  probability_values,
                  selection_values,
                  jitter_values,
                  mode,
                  theta_cap); }
            return state_sample_to_dict(result);
        },
        py::arg("copula"),
        py::arg("tau"),
        py::arg("probability"),
        py::arg("selection_draws"),
        py::arg("jitter_draws"),
        py::arg("mode"),
        py::arg("theta_cap") = std::numeric_limits<double>::quiet_NaN());

    m.def(
        "jacobi_state_histogram_cells",
        [](Float64Array tau,
           py::array_t<std::int64_t,
               py::array::c_style | py::array::forcecast> indices) {
            const py::buffer_info index_info = indices.request();
            if (index_info.ndim != 1) {
                throw std::invalid_argument("indices must be one-dimensional");
            }
            const auto* index_data = static_cast<const std::int64_t*>(
                index_info.ptr);
            const std::vector<std::int64_t> index_values(
                index_data,
                index_data + static_cast<std::size_t>(index_info.shape[0]));
            const std::vector<double> tau_values = flat_vector_from_array(
                tau, "tau");
            scar::JacobiHistogramCellsResult result;
            { py::gil_scoped_release release;
              result = scar::jacobi_state_histogram_cells(
                  tau_values, index_values); }
            return histogram_cells_to_dict(result);
        },
        py::arg("tau"),
        py::arg("indices"));
}

}  // namespace pyscarcopula::bindings
