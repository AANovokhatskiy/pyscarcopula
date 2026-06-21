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
