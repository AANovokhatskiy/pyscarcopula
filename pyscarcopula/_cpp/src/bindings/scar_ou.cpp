#include "common.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_scar_ou(py::module_& m) {
    py::class_<scar::ScarOuEvaluator>(m, "ScarOuEvaluator")
        .def(py::init<>())
        .def(
            "loglik_spectral",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return loglik_result_to_dict(
                    evaluator.loglik_spectral(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "loglik_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return loglik_result_to_dict(
                    evaluator.loglik_local_gh(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "loglik_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return loglik_result_to_dict(
                    evaluator.loglik_matrix(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "loglik_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return loglik_result_to_dict(
                    evaluator.loglik_auto(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_spectral",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_spectral(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_local_gh(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_matrix(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_and_corr_spectral",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_spectral(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_and_corr_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_local_gh(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_and_corr_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_matrix(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_and_corr_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_auto(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "neg_loglik_with_grad_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_auto(
                        params, copula, observation_view_from_array(copula, u),
                        config));
            })
        .def(
            "predictive_mean_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                std::vector<double> values = evaluator.predictive_mean_local_gh(
                    params, copula, obs, config, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::LocalGh));
            })
        .def(
            "predictive_mean_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                std::vector<double> values = evaluator.predictive_mean_matrix(
                    params, copula, obs, config, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::Matrix));
            })
        .def(
            "predictive_mean_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                std::vector<double> values = evaluator.predictive_mean_auto(
                    params, copula, obs, config, backend, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "mixture_h_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                std::vector<double> values = evaluator.mixture_h_local_gh(
                    params, copula, obs, config, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::LocalGh));
            })
        .def(
            "mixture_h_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                std::vector<double> values = evaluator.mixture_h_matrix(
                    params, copula, obs, config, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::Matrix));
            })
        .def(
            "mixture_h_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                const scar::ObservationView obs =
                    observation_view_from_array(copula, u);
                std::vector<double> values = evaluator.mixture_h_auto(
                    params, copula, obs, config, backend, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "state_distribution_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                return state_distribution_to_dict(
                    evaluator.state_distribution_local_gh(
                        params, copula, observation_view_from_array(copula, u),
                        config, horizon_next));
            })
        .def(
            "state_distribution_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                return state_distribution_to_dict(
                    evaluator.state_distribution_matrix(
                        params, copula, observation_view_from_array(copula, u),
                        config, horizon_next));
            })
        .def(
            "state_distribution_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                return state_distribution_to_dict(
                    evaluator.state_distribution_auto(
                        params, copula, observation_view_from_array(copula, u),
                        config, horizon_next));
            });
}

}  // namespace pyscarcopula::bindings
