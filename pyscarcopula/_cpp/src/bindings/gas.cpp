#include "array.hpp"
#include "module.hpp"

#include "scar/gas.hpp"
#include "scar/gas_rvine.hpp"

#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

py::dict gas_loglik_result_to_dict(const scar::GasLogLikResult& result) {
    py::dict output;
    output["log_likelihood"] = result.log_likelihood;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    return output;
}

py::dict gas_filter_result_to_dict(const scar::GasFilterResult& result) {
    py::dict output;
    output["g_path"] = vector_to_array(result.g_path);
    output["r_path"] = vector_to_array(result.r_path);
    output["score_path"] = vector_to_array(result.score_path);
    output["log_likelihood"] = result.log_likelihood;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    return output;
}

py::dict gas_update_result_to_dict(const scar::GasUpdateResult& result) {
    py::dict output;
    output["g_next"] = result.g_next;
    output["r"] = result.r;
    output["r_next"] = result.r_next;
    output["log_likelihood"] = result.log_likelihood;
    output["score"] = result.score;
    output["status"] = static_cast<int>(result.status);
    return output;
}

py::dict gas_state_result_to_dict(const scar::GasStateResult& result) {
    py::dict output;
    output["g"] = result.g;
    output["parameter"] = result.parameter;
    output["status"] = static_cast<int>(result.status);
    return output;
}

py::dict gas_predict_result_to_dict(const scar::GasPredictResult& result) {
    py::dict output;
    output["parameter"] = result.parameter;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    return output;
}

py::dict gas_path_result_to_dict(const scar::GasPathResult& result) {
    py::dict output;
    output["values"] = vector_to_array(result.values);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    return output;
}

scar::ObservationView set_equicorr_prepared(
    scar::CopulaSpec& copula,
    py::array_t<double, py::array::c_style | py::array::forcecast> sum_z,
    py::array_t<double, py::array::c_style | py::array::forcecast> sum_z2) {

    copula.equicorr_sum_scores() = vector_from_array(sum_z);
    copula.equicorr_sum_squares() = vector_from_array(sum_z2);
    if (copula.equicorr_sum_scores().empty()
        || copula.equicorr_sum_squares().size()
            != copula.equicorr_sum_scores().size()) {
        throw std::invalid_argument(
            "prepared Equicorr statistics must be non-empty and equal-sized");
    }
    return {
        nullptr,
        copula.equicorr_sum_scores().size(),
        copula.dim,
    };
}

}  // namespace

void bind_gas(py::module_& m) {
    py::enum_<scar::GasScaling>(
        m, "GasScaling", "Scaling applied to the GAS score.")
        .value("Unit", scar::GasScaling::Unit)
        .value("Fisher", scar::GasScaling::Fisher);

    py::class_<scar::GasParams>(
        m, "GasParams", "Parameters of the score-driven GAS recursion.")
        .def(py::init<>())
        .def_readwrite("omega", &scar::GasParams::omega)
        .def_readwrite("gamma", &scar::GasParams::gamma)
        .def_readwrite("beta", &scar::GasParams::beta);

    py::class_<scar::GasConfig>(
        m, "GasConfig", "Numerical safeguards and GAS score scaling.")
        .def(py::init<>())
        .def_readwrite("scaling", &scar::GasConfig::scaling)
        .def_readwrite("score_eps", &scar::GasConfig::score_eps)
        .def_readwrite("g_clip", &scar::GasConfig::g_clip)
        .def_readwrite("score_clip", &scar::GasConfig::score_clip)
        .def_readwrite("fisher_floor", &scar::GasConfig::fisher_floor)
        .def_readwrite(
            "stationary_beta_tol",
            &scar::GasConfig::stationary_beta_tol);

    py::class_<scar::GasRvineEdge>(
        m, "GasRvineEdge", "Native GAS R-vine edge specification.")
        .def(py::init<>())
        .def_readwrite("copula", &scar::GasRvineEdge::copula)
        .def_readwrite("gas_params", &scar::GasRvineEdge::gas_params)
        .def_readwrite("gas_config", &scar::GasRvineEdge::gas_config)
        .def_readwrite("dynamic", &scar::GasRvineEdge::dynamic);

    m.def(
        "gas_rvine_sample",
        [](const std::vector<scar::GasRvineEdge>& edges,
           const scar::RVineTraversalPlan& plan,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> uniforms,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> parameter_paths) {
            const py::buffer_info uniforms_info = uniforms.request();
            const py::buffer_info parameters_info = parameter_paths.request();
            if (uniforms_info.ndim != 2
                || parameters_info.ndim != 2) {
                throw std::invalid_argument(
                    "uniforms and parameter_paths must be 2D arrays");
            }
            scar::GasRvineSampleResult result;
            {
                py::gil_scoped_release release;
                result = scar::gas_rvine_sample(
                    edges,
                    plan,
                    static_cast<const double*>(uniforms_info.ptr),
                    static_cast<std::int64_t>(uniforms_info.shape[0]),
                    static_cast<std::int64_t>(uniforms_info.shape[1]),
                    static_cast<const double*>(parameters_info.ptr),
                    static_cast<std::int64_t>(parameters_info.shape[0]),
                    static_cast<std::int64_t>(parameters_info.shape[1]));
            }
            py::dict out;
            out["values"] = vector_to_array(result.values);
            out["n_rows"] = result.n_rows;
            out["dimension"] = result.dimension;
            out["status"] = static_cast<int>(result.status);
            out["failure_row"] = result.failure.row;
            out["failure_edge"] = result.failure.edge;
            return out;
        },
        py::arg("edges"),
        py::arg("plan"),
        py::arg("uniforms"),
        py::arg("parameter_paths"));

    py::class_<scar::GasEvaluator>(
        m, "GasEvaluator", "Native evaluator for GAS copula dynamics.")
        .def(py::init<>())
        .def(
            "initial_state",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               const scar::GasConfig& config) {
                return gas_state_result_to_dict(
                    evaluator.initial_state(params, copula, config));
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("config"))
        .def(
            "filter",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(
                        copula.model_descriptor().expected_dimension(), u);
                scar::GasFilterResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.filter(params, copula, obs, config);
                }
                return gas_filter_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("u"),
            py::arg("config"))
        .def(
            "filter_equicorr_prepared",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               scar::CopulaSpec copula,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z2,
               const scar::GasConfig& config) {
                const auto obs = set_equicorr_prepared(
                    copula, sum_z, sum_z2);
                scar::GasFilterResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.filter(
                        params, copula, obs, config);
                }
                return gas_filter_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("sum_z"),
            py::arg("sum_z2"),
            py::arg("config"))
        .def(
            "log_likelihood",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(
                        copula.model_descriptor().expected_dimension(), u);
                scar::GasLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.log_likelihood(
                        params, copula, obs, config);
                }
                return gas_loglik_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("u"),
            py::arg("config"))
        .def(
            "log_likelihood_equicorr_prepared",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               scar::CopulaSpec copula,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z2,
               const scar::GasConfig& config) {
                const auto obs = set_equicorr_prepared(
                    copula, sum_z, sum_z2);
                scar::GasLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.log_likelihood(
                        params, copula, obs, config);
                }
                return gas_loglik_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("sum_z"),
            py::arg("sum_z2"),
            py::arg("config"))
        .def(
            "negative_log_likelihood",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(
                        copula.model_descriptor().expected_dimension(), u);
                scar::GasLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.negative_log_likelihood(
                        params, copula, obs, config);
                }
                return gas_loglik_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("u"),
            py::arg("config"))
        .def(
            "negative_log_likelihood_equicorr_prepared",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               scar::CopulaSpec copula,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z2,
               const scar::GasConfig& config) {
                const auto obs = set_equicorr_prepared(
                    copula, sum_z, sum_z2);
                scar::GasLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.negative_log_likelihood(
                        params, copula, obs, config);
                }
                return gas_loglik_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("sum_z"),
            py::arg("sum_z2"),
            py::arg("config"))
        .def(
            "update_one",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               double g,
               double u1,
               double u2,
               const scar::GasConfig& config) {
                scar::GasUpdateResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.update_one(
                        params, copula, g, u1, u2, config);
                }
                return gas_update_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("g"),
            py::arg("u1"),
            py::arg("u2"),
            py::arg("config"))
        .def(
            "update_observation",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               double g,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> observation,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(
                        copula.model_descriptor().expected_dimension(),
                        observation);
                scar::GasUpdateResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.update_observation(
                        params, copula, g, obs, config);
                }
                return gas_update_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("g"),
            py::arg("observation"),
            py::arg("config"))
        .def(
            "predict_parameter",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config,
               bool horizon_next) {
                const scar::ObservationView obs =
                    observation_view_from_array(
                        copula.model_descriptor().expected_dimension(), u);
                scar::GasPredictResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.predict_parameter(
                        params, copula, obs, config, horizon_next);
                }
                return gas_predict_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("u"),
            py::arg("config"),
            py::arg("horizon_next"))
        .def(
            "predict_parameter_equicorr_prepared",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               scar::CopulaSpec copula,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> sum_z2,
               const scar::GasConfig& config,
               bool horizon_next) {
                const auto obs = set_equicorr_prepared(
                    copula, sum_z, sum_z2);
                scar::GasPredictResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.predict_parameter(
                        params, copula, obs, config, horizon_next);
                }
                return gas_predict_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("sum_z"),
            py::arg("sum_z2"),
            py::arg("config"),
            py::arg("horizon_next"))
        .def(
            "h_path",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(
                        copula.model_descriptor().expected_dimension(), u);
                scar::GasPathResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.h_path(
                        params, copula, obs, config);
                }
                return gas_path_result_to_dict(result);
            },
            py::arg("params"),
            py::arg("copula"),
            py::arg("u"),
            py::arg("config"));
}

}  // namespace pyscarcopula::bindings
