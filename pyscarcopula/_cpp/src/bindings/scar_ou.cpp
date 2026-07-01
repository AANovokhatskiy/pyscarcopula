#include "common.hpp"

#include <string>
#include <utility>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

scar::PreparedScarOuEvaluator make_prepared_scar_ou_evaluator(
    scar::CopulaSpec copula,
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    const scar::OuNumericalConfig& config,
    const std::string& method) {

    const py::buffer_info info = u.request();
    if (info.ndim != 2 || info.shape[1] < 2) {
        throw std::invalid_argument(
            "u must be a 2D float64 array with shape (n, d), d >= 2");
    }
    const auto n_obs = static_cast<std::int64_t>(info.shape[0]);
    const auto dim = static_cast<int>(info.shape[1]);
    std::vector<double> observations = flat_vector_from_array(u, "u");
    return scar::PreparedScarOuEvaluator(
        std::move(copula),
        std::move(observations),
        n_obs,
        dim,
        config,
        method);
}

}  // namespace

void bind_scar_ou(py::module_& m) {
    py::class_<scar::PreparedScarOuEvaluator>(m, "PreparedScarOuEvaluator")
        .def(py::init(&make_prepared_scar_ou_evaluator))
        .def(
            "update_student_factor",
            [](scar::PreparedScarOuEvaluator& evaluator,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   l_inv,
               double log_det) {
                evaluator.update_student_factor(
                    vector_from_array(l_inv), log_det);
            })
        .def(
            "loglik",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                return loglik_result_to_dict(evaluator.loglik(params));
            })
        .def(
            "neg_loglik_with_grad",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad(params));
            })
        .def(
            "neg_loglik_with_grad_and_corr",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr(params));
            })
        .def(
            "neg_loglik_with_grad_and_corr_directional",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_directional(
                        params, vector_from_array(corr_direction)));
            })
        .def(
            "predictive_mean",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                std::vector<double> values =
                    evaluator.predictive_mean(params, backend, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "mixture_h",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                std::vector<double> values =
                    evaluator.mixture_h(params, backend, status);
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "state_distribution",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               bool horizon_next) {
                return state_distribution_to_dict(
                    evaluator.state_distribution(params, horizon_next));
            });

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
            "neg_loglik_with_grad_and_corr_directional_spectral",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                const std::vector<double> direction =
                    vector_from_array(corr_direction);
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_directional_spectral(
                        params, copula, observation_view_from_array(copula, u),
                        config, direction));
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
            "neg_loglik_with_grad_and_corr_directional_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                const std::vector<double> direction =
                    vector_from_array(corr_direction);
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_directional_local_gh(
                        params, copula, observation_view_from_array(copula, u),
                        config, direction));
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
            "neg_loglik_with_grad_and_corr_directional_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                const std::vector<double> direction =
                    vector_from_array(corr_direction);
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_directional_matrix(
                        params, copula, observation_view_from_array(copula, u),
                        config, direction));
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
            "neg_loglik_with_grad_and_corr_directional_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                const std::vector<double> direction =
                    vector_from_array(corr_direction);
                return grad_loglik_result_to_dict(
                    evaluator.neg_loglik_with_grad_and_corr_directional_auto(
                        params, copula, observation_view_from_array(copula, u),
                        config, direction));
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
