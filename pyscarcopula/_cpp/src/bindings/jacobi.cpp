#include "module.hpp"

#include "array.hpp"
#include "scar/jacobi.hpp"

#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
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

py::dict params_to_dict(const scar::JacobiParamsResult& result) {
    py::dict output = status_dict(result);
    output["values"] = py::make_tuple(
        result.value.kappa, result.value.m, result.value.xi);
    return output;
}

py::dict raw_to_dict(const scar::JacobiRawParamsResult& result) {
    py::dict output = status_dict(result);
    output["values"] = result.value;
    return output;
}

py::dict bounds_to_dict(const scar::JacobiRawBoundsResult& result) {
    py::dict output = status_dict(result);
    output["lower"] = result.value.lower;
    output["upper"] = result.value.upper;
    return output;
}

py::dict shape_to_dict(const scar::JacobiShapeResult& result) {
    py::dict output = status_dict(result);
    output["alpha"] = result.value.alpha;
    output["beta"] = result.value.beta;
    output["dalpha"] = result.value.dalpha;
    output["dbeta"] = result.value.dbeta;
    return output;
}

py::dict scalar_to_dict(const scar::JacobiScalarResult& result) {
    py::dict output = status_dict(result);
    output["value"] = result.value;
    return output;
}

py::dict vector_to_dict(const scar::JacobiVectorResult& result) {
    py::dict output = status_dict(result);
    output["values"] = vector_to_array(result.value);
    return output;
}

py::dict memory_to_dict(const scar::JacobiMemoryResult& result) {
    py::dict output = status_dict(result);
    output["elements"] = result.value.elements;
    output["bytes"] = result.value.bytes;
    output["budget_bytes"] = result.value.budget_bytes;
    output["within_budget"] = result.value.within_budget;
    return output;
}

py::dict boundary_to_dict(const scar::JacobiBoundaryResult& result) {
    py::dict output = status_dict(result);
    output["value"] = result.value.value;
    output["intervened"] = result.value.intervened;
    return output;
}

py::dict quadrature_to_dict(const scar::JacobiQuadratureResult& result) {
    py::dict output = status_dict(result);
    output["nodes"] = vector_to_array(result.value.nodes);
    output["weights"] = vector_to_array(result.value.weights);
    return output;
}

py::dict basis_to_dict(const scar::JacobiBasisResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["basis"] = matrix_to_array(
        result.value.basis,
        static_cast<std::size_t>(result.value.quad_order),
        static_cast<std::size_t>(result.value.basis_order));
    output["basis_derivative"] = matrix_to_array(
        result.value.basis_derivative,
        static_cast<std::size_t>(result.value.quad_order),
        static_cast<std::size_t>(result.value.basis_order));
    return output;
}

py::dict fixed_rule_to_dict(const scar::JacobiFixedRuleResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["weight_derivatives"] = matrix_to_array(
        result.value.weight_derivatives,
        3,
        static_cast<std::size_t>(result.value.quad_order));
    return output;
}

py::array_t<double> tensor3_to_array(
    const std::vector<double>& values,
    std::size_t first,
    std::size_t second,
    std::size_t third) {
    const std::size_t expected = first * second * third;
    if (values.size() != expected) {
        throw std::invalid_argument(
            "native tensor result has an inconsistent shape");
    }
    py::array_t<double> output({
        static_cast<py::ssize_t>(first),
        static_cast<py::ssize_t>(second),
        static_cast<py::ssize_t>(third),
    });
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::array_t<std::int64_t> int64_matrix_to_array(
    const std::vector<std::int64_t>& values,
    std::size_t rows,
    std::size_t columns) {
    if (values.size() != rows * columns) {
        throw std::invalid_argument(
            "native integer matrix result has an inconsistent shape");
    }
    py::array_t<std::int64_t> output({
        static_cast<py::ssize_t>(rows),
        static_cast<py::ssize_t>(columns),
    });
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::array_t<std::int64_t> int64_vector_to_array(
    const std::vector<std::int64_t>& values) {
    py::array_t<std::int64_t> output(
        static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), output.mutable_data());
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
    output["ipfp_stationary_residual"] =
        value.ipfp_stationary_residual;
    output["ipfp_kl_divergence"] = value.ipfp_kl_divergence;
    output["ipfp_max_probability_change"] =
        value.ipfp_max_probability_change;
    return output;
}

py::dict dense_transition_to_dict(
    const scar::JacobiDenseTransitionResult& result) {
    py::dict output = status_dict(result);
    const std::size_t order = result.value.order > 0
        ? static_cast<std::size_t>(result.value.order) : 0;
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["probabilities"] = matrix_to_array(
        result.value.probabilities, order, order);
    output["derivatives"] = tensor3_to_array(
        result.value.derivatives,
        result.value.derivatives.empty() ? 0 : 3,
        order,
        order);
    output["spectral_powers"] =
        vector_to_array(result.value.spectral_powers);
    output["diagnostics"] =
        transition_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict coefficient_transition_to_dict(
    const scar::JacobiCoefficientTransitionResult& result) {
    py::dict output = status_dict(result);
    const std::size_t quad_order = result.value.quad_order > 0
        ? static_cast<std::size_t>(result.value.quad_order) : 0;
    const std::size_t basis_order = result.value.basis_order > 0
        ? static_cast<std::size_t>(result.value.basis_order) : 0;
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["basis"] = matrix_to_array(
        result.value.basis, quad_order, basis_order);
    output["spectral_powers"] =
        vector_to_array(result.value.spectral_powers);
    output["diagnostics"] =
        transition_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict sparse_transition_to_dict(
    const scar::JacobiSparseTransitionResult& result) {
    py::dict output = status_dict(result);
    const std::size_t order = result.value.order > 0
        ? static_cast<std::size_t>(result.value.order) : 0;
    const std::size_t width = result.value.max_width > 0
        ? static_cast<std::size_t>(result.value.max_width) : 0;
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["indices"] = int64_matrix_to_array(
        result.value.indices, order, width);
    output["probabilities"] = matrix_to_array(
        result.value.probabilities, order, width);
    output["counts"] = int64_vector_to_array(result.value.counts);
    output["derivatives"] = tensor3_to_array(
        result.value.derivatives,
        result.value.derivatives.empty() ? 0 : 3,
        order,
        width);
    output["diagnostics"] =
        transition_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict horizon_to_dict(const scar::JacobiHorizonDiagnostics& value) {
    py::dict output;
    output["steps"] = value.steps;
    output["one_step_stationary_tv"] = value.one_step_stationary_tv;
    output["full_horizon_stationary_tv"] =
        value.full_horizon_stationary_tv;
    output["target_mean"] = value.target_mean;
    output["propagated_mean"] = value.propagated_mean;
    output["target_variance"] = value.target_variance;
    output["propagated_variance"] = value.propagated_variance;
    output["relative_variance_error"] = value.relative_variance_error;
    output["conditional_mean_rmse"] = value.conditional_mean_rmse;
    output["conditional_mean_max_error"] =
        value.conditional_mean_max_error;
    output["lag_one_correlation"] = value.lag_one_correlation;
    output["target_lag_one_correlation"] =
        value.target_lag_one_correlation;
    output["lag_one_correlation_error"] =
        value.lag_one_correlation_error;
    return output;
}

scar::JacobiSparseTransition sparse_transition_from_arrays(
    py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast> indices,
    Float64Array probabilities,
    py::array_t<std::int64_t,
        py::array::c_style | py::array::forcecast> counts) {
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
    const auto* index_data = static_cast<const std::int64_t*>(index_info.ptr);
    const auto* probability_data =
        static_cast<const double*>(probability_info.ptr);
    const auto* count_data = static_cast<const std::int64_t*>(count_info.ptr);
    transition.indices.assign(index_data, index_data + size);
    transition.probabilities.assign(
        probability_data, probability_data + size);
    transition.counts.assign(
        count_data, count_data + static_cast<std::size_t>(transition.order));
    return transition;
}

py::dict filter_diagnostics_to_dict(
    const scar::JacobiFilterDiagnostics& value) {
    py::dict output = transition_diagnostics_to_dict(value.transition);
    output["n_obs"] = value.n_obs;
    output["order"] = value.order;
    output["log_likelihood"] = value.log_likelihood;
    output["minimum_scale"] = value.minimum_scale;
    output["maximum_scale"] = value.maximum_scale;
    output["max_predictive_mass_error"] =
        value.max_predictive_mass_error;
    output["max_filtered_mass_error"] = value.max_filtered_mass_error;
    output["max_smoothed_mass_error"] = value.max_smoothed_mass_error;
    output["preparation_generation"] = value.preparation_generation;
    return output;
}

py::dict filter_result_to_dict(const scar::JacobiFilterResult& result) {
    py::dict output = status_dict(result);
    const std::size_t rows = result.value.n_obs > 0
        ? static_cast<std::size_t>(result.value.n_obs) : 0;
    const std::size_t columns = result.value.order > 0
        ? static_cast<std::size_t>(result.value.order) : 0;
    output["tau"] = vector_to_array(result.value.tau);
    output["theta"] = vector_to_array(result.value.theta);
    output["emissions"] = matrix_to_array(
        result.value.emissions, rows, columns);
    output["predicted"] = matrix_to_array(
        result.value.predicted, rows, columns);
    output["filtered"] = matrix_to_array(
        result.value.filtered, rows, columns);
    output["smoothed"] = matrix_to_array(
        result.value.smoothed, rows, columns);
    output["scales"] = vector_to_array(result.value.scales);
    output["current_probability"] =
        vector_to_array(result.value.current_probability);
    output["next_probability"] =
        vector_to_array(result.value.next_probability);
    output["diagnostics"] =
        filter_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict objective_result_to_dict(const scar::JacobiObjectiveResult& result) {
    py::dict output = status_dict(result);
    output["log_likelihood"] = result.value.log_likelihood;
    output["objective"] = result.value.objective;
    output["diagnostics"] =
        filter_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict gradient_result_to_dict(const scar::JacobiGradientResult& result) {
    py::dict output = status_dict(result);
    output["objective"] = result.value.objective;
    output["gradient"] = result.value.gradient;
    output["diagnostics"] =
        filter_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict evaluator_vector_to_dict(
    const scar::JacobiEvaluatorVectorResult& result) {
    py::dict output = status_dict(result);
    output["values"] = vector_to_array(result.value.values);
    output["diagnostics"] =
        filter_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict evaluator_pair_to_dict(
    const scar::JacobiEvaluatorPairResult& result) {
    py::dict output = status_dict(result);
    output["first"] = vector_to_array(result.value.first);
    output["second"] = vector_to_array(result.value.second);
    output["diagnostics"] =
        filter_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

py::dict state_result_to_dict(
    const scar::JacobiStateDistributionResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["probability"] = vector_to_array(result.value.probability);
    output["horizon"] = static_cast<int>(result.value.horizon);
    output["diagnostics"] =
        filter_diagnostics_to_dict(result.value.diagnostics);
    return output;
}

std::unique_ptr<scar::PreparedScarJacobiEvaluator>
make_prepared_scar_jacobi_evaluator(
    scar::CopulaSpec copula,
    Float64Array observations,
    const scar::JacobiEvaluatorConfig& config) {

    const py::buffer_info info = observations.request();
    if (info.ndim != 2 || info.shape[1] != 2 || info.shape[0] < 1) {
        throw std::invalid_argument(
            "u must be a 2D float64 array with shape (n, 2), n >= 1");
    }
    const auto n_obs = static_cast<std::int64_t>(info.shape[0]);
    std::vector<double> values = flat_vector_from_array(observations, "u");
    return std::make_unique<scar::PreparedScarJacobiEvaluator>(
        std::move(copula),
        std::move(values),
        n_obs,
        2,
        config);
}

}  // namespace

void bind_jacobi(py::module_& m) {
    py::enum_<scar::JacobiBoundaryPolicy>(m, "JacobiBoundaryPolicy")
        .value("Reflect", scar::JacobiBoundaryPolicy::Reflect)
        .value("Clip", scar::JacobiBoundaryPolicy::Clip);

    py::enum_<scar::JacobiTransitionMethod>(m, "JacobiTransitionMethod")
        .value("Auto", scar::JacobiTransitionMethod::Auto)
        .value("SpectralMatrix", scar::JacobiTransitionMethod::SpectralMatrix)
        .value("Local", scar::JacobiTransitionMethod::Local)
        .value("LocalFixed", scar::JacobiTransitionMethod::LocalFixed)
        .value("SpectralCoeff", scar::JacobiTransitionMethod::SpectralCoeff);

    py::enum_<scar::JacobiTransitionStorage>(m, "JacobiTransitionStorage")
        .value("Dense", scar::JacobiTransitionStorage::Dense)
        .value("Sparse", scar::JacobiTransitionStorage::Sparse);

    py::enum_<scar::JacobiStationarityCorrection>(
            m, "JacobiStationarityCorrection")
        .value("None_", scar::JacobiStationarityCorrection::None)
        .value(
            "MetropolisHastings",
            scar::JacobiStationarityCorrection::MetropolisHastings)
        .value("IpFp", scar::JacobiStationarityCorrection::IpFp);

    py::enum_<scar::JacobiStateHorizon>(m, "JacobiStateHorizon")
        .value("Current", scar::JacobiStateHorizon::Current)
        .value("Next", scar::JacobiStateHorizon::Next);

    py::class_<scar::JacobiParams>(m, "JacobiParams")
        .def(py::init<>())
        .def_readwrite("kappa", &scar::JacobiParams::kappa)
        .def_readwrite("m", &scar::JacobiParams::m)
        .def_readwrite("xi", &scar::JacobiParams::xi);

    py::class_<scar::JacobiParameterBounds>(m, "JacobiParameterBounds")
        .def(py::init<>())
        .def_readwrite(
            "kappa_lower", &scar::JacobiParameterBounds::kappa_lower)
        .def_readwrite(
            "kappa_upper", &scar::JacobiParameterBounds::kappa_upper)
        .def_readwrite("xi_lower", &scar::JacobiParameterBounds::xi_lower)
        .def_readwrite("xi_upper", &scar::JacobiParameterBounds::xi_upper)
        .def_readwrite("tau_eps", &scar::JacobiParameterBounds::tau_eps);

    py::class_<scar::JacobiNumericalConfig>(m, "JacobiNumericalConfig")
        .def(py::init<>())
        .def_readwrite("quad_order", &scar::JacobiNumericalConfig::quad_order)
        .def_readwrite(
            "basis_order", &scar::JacobiNumericalConfig::basis_order)
        .def_readwrite("gh_order", &scar::JacobiNumericalConfig::gh_order)
        .def_readwrite("n_obs", &scar::JacobiNumericalConfig::n_obs)
        .def_readwrite("matrix", &scar::JacobiNumericalConfig::matrix)
        .def_readwrite("gradient", &scar::JacobiNumericalConfig::gradient)
        .def_readwrite(
            "memory_budget_bytes",
            &scar::JacobiNumericalConfig::memory_budget_bytes)
        .def_readwrite("tau_eps", &scar::JacobiNumericalConfig::tau_eps)
        .def_readwrite("theta_cap", &scar::JacobiNumericalConfig::theta_cap)
        .def_readwrite(
            "stationary_shape_max",
            &scar::JacobiNumericalConfig::stationary_shape_max)
        .def_readwrite(
            "lamperti_eps", &scar::JacobiNumericalConfig::lamperti_eps)
        .def_readwrite("boundary", &scar::JacobiNumericalConfig::boundary);

    py::class_<scar::JacobiTransitionConfig>(m, "JacobiTransitionConfig")
        .def(py::init<>())
        .def_readwrite("numerical", &scar::JacobiTransitionConfig::numerical)
        .def_readwrite("method", &scar::JacobiTransitionConfig::method)
        .def_readwrite("storage", &scar::JacobiTransitionConfig::storage)
        .def_readwrite("correction", &scar::JacobiTransitionConfig::correction)
        .def_readwrite(
            "negative_mass_tolerance",
            &scar::JacobiTransitionConfig::negative_mass_tolerance)
        .def_readwrite(
            "clip_negative", &scar::JacobiTransitionConfig::clip_negative)
        .def_readwrite(
            "derivatives", &scar::JacobiTransitionConfig::derivatives)
        .def_readwrite(
            "ipfp_tolerance", &scar::JacobiTransitionConfig::ipfp_tolerance)
        .def_readwrite(
            "ipfp_max_iterations",
            &scar::JacobiTransitionConfig::ipfp_max_iterations);

    py::class_<scar::JacobiAdaptiveThresholds>(
            m, "JacobiAdaptiveThresholds")
        .def(py::init<>())
        .def_readwrite(
            "max_full_horizon_tv",
            &scar::JacobiAdaptiveThresholds::max_full_horizon_tv)
        .def_readwrite(
            "max_relative_variance_error",
            &scar::JacobiAdaptiveThresholds::max_relative_variance_error)
        .def_readwrite(
            "max_conditional_mean_rmse",
            &scar::JacobiAdaptiveThresholds::max_conditional_mean_rmse)
        .def_readwrite(
            "max_lag_one_correlation_error",
            &scar::JacobiAdaptiveThresholds::
                max_lag_one_correlation_error);

    py::class_<scar::JacobiEvaluatorConfig>(m, "JacobiEvaluatorConfig")
        .def(py::init<>())
        .def_readwrite(
            "transition", &scar::JacobiEvaluatorConfig::transition)
        .def_readwrite(
            "finite_difference_relative_step",
            &scar::JacobiEvaluatorConfig::finite_difference_relative_step);

    py::class_<scar::PreparedScarJacobiEvaluator>(
            m,
            "PreparedScarJacobiEvaluator",
            "Prepared native Jacobi objective/filter/state evaluator.")
        .def(py::init(&make_prepared_scar_jacobi_evaluator))
        .def(
            "filter",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiFilterResult result;
                { py::gil_scoped_release release;
                  result = evaluator.filter(params); }
                return filter_result_to_dict(result);
            })
        .def(
            "loglik",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiObjectiveResult result;
                { py::gil_scoped_release release;
                  result = evaluator.loglik(params); }
                return objective_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiGradientResult result;
                { py::gil_scoped_release release;
                  result = evaluator.neg_loglik_with_grad(params); }
                return gradient_result_to_dict(result);
            })
        .def(
            "predictive_mean",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiEvaluatorVectorResult result;
                { py::gil_scoped_release release;
                  result = evaluator.predictive_mean(params); }
                return evaluator_vector_to_dict(result);
            })
        .def(
            "mixture_h",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiEvaluatorVectorResult result;
                { py::gil_scoped_release release;
                  result = evaluator.mixture_h(params); }
                return evaluator_vector_to_dict(result);
            })
        .def(
            "mixture_h_pair",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiEvaluatorPairResult result;
                { py::gil_scoped_release release;
                  result = evaluator.mixture_h_pair(params); }
                return evaluator_pair_to_dict(result);
            })
        .def(
            "rosenblatt",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiEvaluatorPairResult result;
                { py::gil_scoped_release release;
                  result = evaluator.rosenblatt(params); }
                return evaluator_pair_to_dict(result);
            })
        .def(
            "gaussian_rosenblatt",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params) {
                scar::JacobiEvaluatorPairResult result;
                { py::gil_scoped_release release;
                  result = evaluator.gaussian_rosenblatt(params); }
                return evaluator_pair_to_dict(result);
            })
        .def(
            "state_distribution",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               const scar::JacobiParams& params,
               scar::JacobiStateHorizon horizon) {
                scar::JacobiStateDistributionResult result;
                { py::gil_scoped_release release;
                  result = evaluator.state_distribution(params, horizon); }
                return state_result_to_dict(result);
            })
        .def(
            "condition_state",
            [](const scar::PreparedScarJacobiEvaluator& evaluator,
               Float64Array tau,
               Float64Array probability,
               Float64Array observation,
               scar::JacobiStateHorizon horizon) {
                const std::vector<double> tau_values =
                    flat_vector_from_array(tau, "tau");
                const std::vector<double> probability_values =
                    flat_vector_from_array(probability, "probability");
                const std::vector<double> observation_values =
                    flat_vector_from_array(observation, "observation");
                if (observation_values.size() != 2) {
                    throw std::invalid_argument(
                        "observation must contain exactly two values");
                }
                scar::JacobiStateDistributionResult result;
                { py::gil_scoped_release release;
                  result = evaluator.condition_state(
                      tau_values,
                      probability_values,
                      {observation_values[0], observation_values[1]},
                      horizon); }
                return state_result_to_dict(result);
            },
            py::arg("tau"),
            py::arg("probability"),
            py::arg("observation"),
            py::arg("horizon") = scar::JacobiStateHorizon::Current)
        .def_property_readonly(
            "preparation_count",
            &scar::PreparedScarJacobiEvaluator::preparation_count);

    m.def("jacobi_raw_to_physical", [](
            const std::array<double, 3>& raw) {
        scar::JacobiParamsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_raw_to_physical(raw); }
        return params_to_dict(result);
    });
    m.def("jacobi_physical_to_raw", [](
            const scar::JacobiParams& params, double tau_eps) {
        scar::JacobiRawParamsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_physical_to_raw(params, tau_eps); }
        return raw_to_dict(result);
    });
    m.def("jacobi_gradient_to_raw", [](
            const scar::JacobiParams& params,
            const std::array<double, 3>& gradient) {
        scar::JacobiRawParamsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_gradient_to_raw(params, gradient); }
        return raw_to_dict(result);
    }, py::arg("params"), py::arg("gradient"));
    m.def("jacobi_raw_bounds", [](
            const scar::JacobiParameterBounds& bounds) {
        scar::JacobiRawBoundsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_raw_bounds(bounds); }
        return bounds_to_dict(result);
    });
    m.def("jacobi_initial_point",
        [](const std::optional<double>& tau, double tau_eps) {
            return params_to_dict(scar::jacobi_initial_point(
                tau.value_or(0.0), tau.has_value(), tau_eps));
        }, py::arg("tau"), py::arg("tau_eps"));
    m.def("jacobi_stationary_shape", [](
            const scar::JacobiParams& params) {
        scar::JacobiShapeResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_stationary_shape(params); }
        return shape_to_dict(result);
    });
    m.def("jacobi_sample_stationary", [](
            const scar::JacobiParams& params,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                uniforms) {
        const std::vector<double> draws = flat_vector_from_array(
            uniforms, "uniforms");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::sample_jacobi_stationary(params, draws); }
        return vector_to_dict(result);
    }, py::arg("params"), py::arg("uniforms"));
    m.def("jacobi_validate_params", [](
            const scar::JacobiParams& params, double shape_max) {
        int status = 0;
        { py::gil_scoped_release release;
          status = static_cast<int>(
              scar::validate_jacobi_params(params, shape_max)); }
        return status;
    });
    m.def("jacobi_validate_config", [](
            const scar::JacobiNumericalConfig& config) {
        int status = 0;
        { py::gil_scoped_release release;
          status = static_cast<int>(scar::validate_jacobi_config(config)); }
        return status;
    });
    m.def("jacobi_resolve_dt", [](std::int64_t n_obs) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_resolve_dt(n_obs); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_estimate_workspace", [](
            const scar::JacobiNumericalConfig& config) {
        scar::JacobiMemoryResult result;
        { py::gil_scoped_release release;
          result = scar::estimate_jacobi_workspace(config); }
        return memory_to_dict(result);
    });
    m.def("jacobi_estimate_sampling_workspace", [](
            std::int64_t n, const scar::JacobiNumericalConfig& config) {
        scar::JacobiMemoryResult result;
        { py::gil_scoped_release release;
          result = scar::estimate_jacobi_sampling_workspace(n, config); }
        return memory_to_dict(result);
    });
    m.def("jacobi_lamperti_values", [](
            Float64Array tau, double xi) {
        const std::vector<double> values = flat_vector_from_array(tau, "tau");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_lamperti_values(values, xi); }
        return vector_to_dict(result);
    });
    m.def("jacobi_inverse_lamperti_values", [](
            Float64Array values, double xi) {
        const std::vector<double> input =
            flat_vector_from_array(values, "values");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_inverse_lamperti_values(input, xi); }
        return vector_to_dict(result);
    });
    m.def("jacobi_lamperti_drift_values", [](
            const scar::JacobiParams& params,
            Float64Array tau,
            double interior_eps) {
        const std::vector<double> values = flat_vector_from_array(tau, "tau");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_lamperti_drift_values(
              params, values, interior_eps); }
        return vector_to_dict(result);
    }, py::arg("params"), py::arg("tau"), py::arg("interior_eps") = 0.0);
    m.def("jacobi_apply_boundary", [](
            double value,
            double upper,
            scar::JacobiBoundaryPolicy policy) {
        scar::JacobiBoundaryResult result;
        { py::gil_scoped_release release;
          result = scar::apply_jacobi_boundary(value, upper, policy); }
        return boundary_to_dict(result);
    });
    m.def("jacobi_log_beta", [](double alpha, double beta) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_log_beta(alpha, beta); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_digamma", [](double value) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_digamma(value); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_trigamma", [](double value) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_trigamma(value); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_gauss_hermite_rule", [](
            int order, std::uint64_t memory_budget_bytes) {
        scar::JacobiQuadratureResult result;
        { py::gil_scoped_release release;
          result = scar::gauss_hermite_probability_rule(
              order, memory_budget_bytes); }
        return quadrature_to_dict(result);
    }, py::arg("order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_gauss_jacobi_rule", [](
            double alpha,
            double beta,
            int order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiQuadratureResult result;
        { py::gil_scoped_release release;
          result = scar::gauss_jacobi_probability_rule(
              alpha, beta, order, memory_budget_bytes); }
        return quadrature_to_dict(result);
    }, py::arg("alpha"), py::arg("beta"), py::arg("order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_build_rule", [](
            double alpha,
            double beta,
            int quad_order,
            int basis_order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiBasisResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_rule(
              alpha, beta, quad_order, basis_order, memory_budget_bytes); }
        return basis_to_dict(result);
    }, py::arg("alpha"), py::arg("beta"), py::arg("quad_order"),
       py::arg("basis_order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_build_fixed_rule", [](
            const scar::JacobiParams& params,
            int quad_order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiFixedRuleResult result;
        { py::gil_scoped_release release;
          result = scar::build_fixed_jacobi_rule(
              params, quad_order, memory_budget_bytes); }
        return fixed_rule_to_dict(result);
    }, py::arg("params"), py::arg("quad_order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_build_fixed_shape_rule", [](
            double alpha,
            double beta,
            int quad_order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiFixedRuleResult result;
        { py::gil_scoped_release release;
          result = scar::build_fixed_jacobi_shape_rule(
              alpha, beta, quad_order, memory_budget_bytes); }
        return fixed_rule_to_dict(result);
    }, py::arg("alpha"), py::arg("beta"), py::arg("quad_order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_resolve_basis_order", [](
            int requested_basis_order, int quad_order) {
        const scar::JacobiIntResult result = scar::resolve_jacobi_basis_order(
            requested_basis_order, quad_order);
        py::dict output = status_dict(result);
        output["order"] = result.value;
        return output;
    }, py::arg("requested_basis_order"), py::arg("quad_order"));
    m.def("jacobi_horizon_steps", [](std::int64_t n_obs) {
        const scar::JacobiIntResult result = scar::jacobi_horizon_steps(n_obs);
        py::dict output = status_dict(result);
        output["steps"] = result.value;
        return output;
    }, py::arg("n_obs"));
    m.def("jacobi_evaluate_polynomials", [](
            double x,
            double alpha,
            double beta,
            int order,
            bool derivative) {
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::evaluate_jacobi_polynomials(
              x, alpha, beta, order, derivative); }
        return vector_to_dict(result);
    }, py::arg("x"), py::arg("alpha"), py::arg("beta"),
       py::arg("order"), py::arg("derivative") = false);
    m.def("jacobi_transition_powers", [](
            const scar::JacobiParams& params,
            std::int64_t n_obs,
            int basis_order) {
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_transition_powers(
              params, n_obs, basis_order); }
        return vector_to_dict(result);
    });
    m.def("jacobi_default_quad_order", [](int basis_order) {
        scar::JacobiIntResult result;
        { py::gil_scoped_release release;
          result = scar::default_jacobi_quad_order(basis_order); }
        py::dict output = status_dict(result);
        output["value"] = result.value;
        return output;
    });
    m.def("jacobi_estimate_sparse_workspace", [](
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiMemoryResult result;
        { py::gil_scoped_release release;
          result = scar::estimate_jacobi_sparse_workspace(config); }
        return memory_to_dict(result);
    });
    m.def("jacobi_estimate_sparse_storage", [](
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiMemoryResult result;
        { py::gil_scoped_release release;
          result = scar::estimate_jacobi_sparse_storage(config); }
        return memory_to_dict(result);
    });
    m.def("jacobi_build_coefficient_transition", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiCoefficientTransitionResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_coefficient_transition(
              params, config); }
        return coefficient_transition_to_dict(result);
    });
    m.def("jacobi_apply_coefficient_transition", [](
            Float64Array powers,
            Float64Array coefficients) {
        scar::JacobiCoefficientTransition transition;
        transition.spectral_powers = flat_vector_from_array(
            powers, "powers");
        transition.basis_order = static_cast<int>(
            transition.spectral_powers.size());
        const std::vector<double> coefficient_values =
            flat_vector_from_array(coefficients, "coefficients");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::apply_jacobi_coefficient_transition(
              transition, coefficient_values); }
        return vector_to_dict(result);
    });
    m.def("jacobi_build_spectral_transition", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiDenseTransitionResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_spectral_transition(params, config); }
        return dense_transition_to_dict(result);
    });
    m.def("jacobi_build_local_transition", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiDenseTransitionResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_local_transition(params, config); }
        return dense_transition_to_dict(result);
    });
    m.def("jacobi_build_fixed_transition", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiDenseTransitionResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_fixed_transition(params, config); }
        return dense_transition_to_dict(result);
    });
    m.def("jacobi_build_dense_transition", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiDenseTransitionResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_dense_transition(params, config); }
        return dense_transition_to_dict(result);
    });
    m.def("jacobi_build_sparse_transition", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config) {
        scar::JacobiSparseTransitionResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_sparse_transition(params, config); }
        return sparse_transition_to_dict(result);
    });
    m.def("jacobi_validate_sparse_transition", [](
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> indices,
            Float64Array probabilities,
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> counts) {
        const scar::JacobiSparseTransition transition =
            sparse_transition_from_arrays(indices, probabilities, counts);
        switch (scar::validate_jacobi_sparse_transition(transition)) {
        case scar::JacobiSparseValidationCode::Ok:
            return;
        case scar::JacobiSparseValidationCode::InvalidShape:
            throw py::value_error(
                "indices and probabilities must have the same 2D shape");
        case scar::JacobiSparseValidationCode::InvalidCount:
            throw py::value_error(
                "counts are outside the sparse row width");
        case scar::JacobiSparseValidationCode::InvalidProbability:
            throw py::value_error(
                "probabilities must be finite and non-negative");
        case scar::JacobiSparseValidationCode::IndexOutOfRange:
            throw py::value_error(
                "active sparse indices are out of range");
        case scar::JacobiSparseValidationCode::IndicesNotIncreasing:
            throw py::value_error(
                "active sparse indices must be strictly increasing");
        case scar::JacobiSparseValidationCode::RowSum:
            throw py::value_error(
                "sparse transition rows must sum to one");
        }
        throw py::value_error("invalid sparse transition");
    });
    m.def("jacobi_sparse_left_multiply", [](
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> indices,
            Float64Array probabilities,
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> counts,
            Float64Array values) {
        const scar::JacobiSparseTransition transition =
            sparse_transition_from_arrays(indices, probabilities, counts);
        const std::vector<double> vector =
            flat_vector_from_array(values, "values");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_sparse_left_multiply(transition, vector); }
        return vector_to_dict(result);
    });
    m.def("jacobi_sparse_to_dense", [](
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> indices,
            Float64Array probabilities,
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> counts) {
        const scar::JacobiSparseTransition transition =
            sparse_transition_from_arrays(indices, probabilities, counts);
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_sparse_to_dense(transition); }
        py::dict output = status_dict(result);
        output["values"] = result.is_ok()
            ? matrix_to_array(
                result.value,
                static_cast<std::size_t>(transition.order),
                static_cast<std::size_t>(transition.order))
            : vector_to_array(result.value);
        return output;
    });
    m.def("jacobi_sparse_full_horizon_diagnostics", [](
            const scar::JacobiParams& params,
            Float64Array tau,
            Float64Array weights,
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> indices,
            Float64Array probabilities,
            py::array_t<std::int64_t,
                py::array::c_style | py::array::forcecast> counts,
            std::int64_t steps) {
        const std::vector<double> tau_values =
            flat_vector_from_array(tau, "tau");
        const std::vector<double> weight_values =
            flat_vector_from_array(weights, "weights");
        const scar::JacobiSparseTransition transition =
            sparse_transition_from_arrays(indices, probabilities, counts);
        scar::JacobiHorizonResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_sparse_full_horizon_diagnostics(
              params,
              tau_values,
              weight_values,
              transition,
              steps); }
        py::dict output = status_dict(result);
        output["diagnostics"] = horizon_to_dict(result.value);
        return output;
    });
    m.def("jacobi_select_sparse_order", [](
            const scar::JacobiParams& params,
            const scar::JacobiTransitionConfig& config,
            const std::vector<int>& quad_orders,
            const scar::JacobiAdaptiveThresholds& thresholds,
            bool require_pass) {
        scar::JacobiAdaptiveSelectionResult result;
        { py::gil_scoped_release release;
          result = scar::select_sparse_jacobi_order(
              params, config, quad_orders, thresholds, require_pass); }
        py::dict output = status_dict(result);
        scar::JacobiSparseTransitionResult transition_result;
        transition_result.value = result.value.transition;
        py::dict transition = sparse_transition_to_dict(transition_result);
        output["transition"] = transition;
        output["selected_quad_order"] = result.value.selected_quad_order;
        output["passed"] = result.value.passed;
        output["exhausted"] = result.value.exhausted;
        py::list candidates;
        for (const scar::JacobiAdaptiveCandidate& candidate
             : result.value.candidates) {
            py::dict record;
            record["quad_order"] = candidate.quad_order;
            record["passed"] = candidate.passed;
            record["memory_limited"] = candidate.memory_limited;
            record["status"] = static_cast<int>(candidate.status);
            record["retained_bytes"] = candidate.retained_bytes;
            record["diagnostics"] = horizon_to_dict(candidate.diagnostics);
            candidates.append(record);
        }
        output["candidates"] = candidates;
        return output;
    });
}

}  // namespace pyscarcopula::bindings
