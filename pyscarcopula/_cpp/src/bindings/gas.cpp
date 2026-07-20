#include "common.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_gas(py::module_& m) {
    py::class_<scar::GasParams>(m, "GasParams")
        .def(py::init<>())
        .def_readwrite("omega", &scar::GasParams::omega)
        .def_readwrite("gamma", &scar::GasParams::gamma)
        .def_readwrite("beta", &scar::GasParams::beta);

    py::class_<scar::GasConfig>(m, "GasConfig")
        .def(py::init<>())
        .def_readwrite("scaling", &scar::GasConfig::scaling)
        .def_readwrite("score_eps", &scar::GasConfig::score_eps)
        .def_readwrite("g_clip", &scar::GasConfig::g_clip)
        .def_readwrite("score_clip", &scar::GasConfig::score_clip)
        .def_readwrite("fisher_floor", &scar::GasConfig::fisher_floor)
        .def_readwrite(
            "stationary_beta_tol",
            &scar::GasConfig::stationary_beta_tol);

    py::class_<scar::GasRvineEdge>(m, "GasRvineEdge")
        .def(py::init<>())
        .def_readwrite("copula", &scar::GasRvineEdge::copula)
        .def_readwrite("gas_params", &scar::GasRvineEdge::gas_params)
        .def_readwrite("gas_config", &scar::GasRvineEdge::gas_config)
        .def_readwrite("dynamic", &scar::GasRvineEdge::dynamic);

    py::class_<scar::GasRvinePlan>(m, "GasRvinePlan")
        .def(py::init<>())
        .def_readwrite("dimension", &scar::GasRvinePlan::dimension)
        .def_readwrite("node_count", &scar::GasRvinePlan::node_count)
        .def_readwrite(
            "last_uniform_column",
            &scar::GasRvinePlan::last_uniform_column)
        .def_readwrite(
            "last_output_node",
            &scar::GasRvinePlan::last_output_node)
        .def_readwrite("output_nodes", &scar::GasRvinePlan::output_nodes)
        .def_readwrite("column_uniforms", &scar::GasRvinePlan::column_uniforms)
        .def_readwrite("inverse_offsets", &scar::GasRvinePlan::inverse_offsets)
        .def_readwrite("inverse_edges", &scar::GasRvinePlan::inverse_edges)
        .def_readwrite(
            "inverse_partner_nodes",
            &scar::GasRvinePlan::inverse_partner_nodes)
        .def_readwrite(
            "inverse_output_nodes",
            &scar::GasRvinePlan::inverse_output_nodes)
        .def_readwrite("forward_offsets", &scar::GasRvinePlan::forward_offsets)
        .def_readwrite("forward_edges", &scar::GasRvinePlan::forward_edges)
        .def_readwrite(
            "forward_leaf_nodes",
            &scar::GasRvinePlan::forward_leaf_nodes)
        .def_readwrite(
            "forward_partner_nodes",
            &scar::GasRvinePlan::forward_partner_nodes)
        .def_readwrite(
            "forward_leaf_output_nodes",
            &scar::GasRvinePlan::forward_leaf_output_nodes)
        .def_readwrite(
            "forward_partner_output_nodes",
            &scar::GasRvinePlan::forward_partner_output_nodes)
        .def_readwrite(
            "update_u1_nodes",
            &scar::GasRvinePlan::update_u1_nodes)
        .def_readwrite(
            "update_u2_nodes",
            &scar::GasRvinePlan::update_u2_nodes);

    m.def(
        "gas_rvine_sample",
        [](const std::vector<scar::GasRvineEdge>& edges,
           const scar::GasRvinePlan& plan,
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
            out["status"] = result.status;
            out["failure_row"] = result.failure_row;
            out["failure_edge"] = result.failure_edge;
            return out;
        },
        py::arg("edges"),
        py::arg("plan"),
        py::arg("uniforms"),
        py::arg("parameter_paths"));

    py::class_<scar::GasEvaluator>(m, "GasEvaluator")
        .def(py::init<>())
        .def(
            "initial_state",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               const scar::GasConfig& config) {
                return gas_state_result_to_dict(
                    evaluator.initial_state(params, copula, config));
            })
        .def(
            "filter",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                scar::GasFilterResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.filter(params, copula, obs, config);
                }
                return gas_filter_result_to_dict(result);
            })
        .def(
            "log_likelihood",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                scar::GasLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.log_likelihood(
                        params, copula, obs, config);
                }
                return gas_loglik_result_to_dict(result);
            })
        .def(
            "negative_log_likelihood",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                scar::GasLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.negative_log_likelihood(
                        params, copula, obs, config);
                }
                return gas_loglik_result_to_dict(result);
            })
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
            })
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
                    observation_view_from_array(copula, observation);
                scar::GasUpdateResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.update_observation(
                        params, copula, g, obs, config);
                }
                return gas_update_result_to_dict(result);
            })
        .def(
            "predict_parameter",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config,
               bool horizon_next) {
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
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
            "h_path",
            [](const scar::GasEvaluator& evaluator,
               const scar::GasParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::GasConfig& config) {
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                scar::GasPathResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.h_path(
                        params, copula, obs, config);
                }
                return gas_path_result_to_dict(result);
            });
}

}  // namespace pyscarcopula::bindings
