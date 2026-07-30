#include "common.hpp"

#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

std::unique_ptr<scar::PreparedScarOuEvaluator>
make_prepared_scar_ou_evaluator(
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
    observation_view_from_array(copula, u);
    std::vector<double> observations = flat_vector_from_array(u, "u");
    return std::make_unique<scar::PreparedScarOuEvaluator>(
        std::move(copula),
        std::move(observations),
        n_obs,
        dim,
        config,
        method);
}

std::unique_ptr<scar::PreparedScarOuEvaluator>
make_prepared_equicorr_scar_ou_evaluator(
    scar::CopulaSpec copula,
    py::array_t<double, py::array::c_style | py::array::forcecast> sum_z,
    py::array_t<double, py::array::c_style | py::array::forcecast> sum_z2,
    const scar::OuNumericalConfig& config,
    const std::string& method) {

    return std::make_unique<scar::PreparedScarOuEvaluator>(
        std::move(copula),
        vector_from_array(sum_z),
        vector_from_array(sum_z2),
        config,
        method);
}

template <typename Callback>
auto with_observation_view_without_gil(
    const scar::CopulaSpec& copula,
    py::array_t<double, py::array::c_style | py::array::forcecast>& u,
    Callback&& callback) {

    const scar::ObservationView obs =
        observation_view_from_array(copula, u);
    py::gil_scoped_release release;
    return std::forward<Callback>(callback)(obs);
}

struct NativeSparseTransition {
    int K = 0;
    scar_internal::SparseTransitionMatrix matrix;
};

std::unique_ptr<NativeSparseTransition> make_sparse_transition(
    py::array_t<
        double,
        py::array::c_style | py::array::forcecast> z,
    double rho,
    double sigma_cond,
    py::array_t<
        double,
        py::array::c_style | py::array::forcecast> trap_w,
    int K,
    int band) {

    const std::vector<double> z_values = vector_from_array(std::move(z));
    const std::vector<double> weights =
        vector_from_array(std::move(trap_w));
    auto transition = std::make_unique<NativeSparseTransition>();
    transition->K = K;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = scar_internal::build_sparse_transition_matrix(
            z_values,
            rho,
            sigma_cond,
            weights,
            K,
            band,
            transition->matrix);
    }
    if (!ok) {
        throw std::invalid_argument(
            "invalid sparse transition matrix parameters");
    }
    return transition;
}

}  // namespace

void bind_scar_ou(py::module_& m) {
    py::class_<NativeSparseTransition>(
        m,
        "_SparseTransitionOperator",
        "Native CSR Gaussian-grid transition operator.")
        .def(py::init(&make_sparse_transition),
        py::arg("z"),
        py::arg("rho"),
        py::arg("sigma_cond"),
        py::arg("trap_w"),
        py::arg("K"),
        py::arg("band"))
        .def(
            "matvec",
            [](const NativeSparseTransition& transition,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> values) {
                const std::vector<double> input =
                    vector_from_array(std::move(values));
                if (input.size()
                    != static_cast<std::size_t>(transition.K)) {
                    throw std::invalid_argument(
                        "values length must equal sparse transition K");
                }
                std::vector<double> output(input.size(), 0.0);
                {
                    py::gil_scoped_release release;
                    scar_internal::sparse_matvec(
                        transition.matrix,
                        transition.K,
                        input,
                        output);
                }
                return vector_to_array(output);
            },
            py::arg("values"))
        .def(
            "rmatvec",
            [](const NativeSparseTransition& transition,
               py::array_t<
                   double,
                   py::array::c_style | py::array::forcecast> values) {
                const std::vector<double> input =
                    vector_from_array(std::move(values));
                if (input.size()
                    != static_cast<std::size_t>(transition.K)) {
                    throw std::invalid_argument(
                        "values length must equal sparse transition K");
                }
                std::vector<double> output(input.size(), 0.0);
                {
                    py::gil_scoped_release release;
                    scar_internal::sparse_transpose_matvec(
                        transition.matrix,
                        transition.K,
                        input,
                        output);
                }
                return vector_to_array(output);
            },
            py::arg("values"))
        .def_property_readonly(
            "nnz",
            [](const NativeSparseTransition& transition) {
                return transition.matrix.data.size();
            });

    py::class_<scar::PreparedScarOuEvaluator>(m, "PreparedScarOuEvaluator")
        .def(py::init(&make_prepared_scar_ou_evaluator))
        .def(py::init(&make_prepared_equicorr_scar_ou_evaluator))
        .def(
            "update_student_factor",
            [](scar::PreparedScarOuEvaluator& evaluator,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   l_inv,
               double log_det) {
                const std::vector<double> factor = vector_from_array(l_inv);
                py::gil_scoped_release release;
                evaluator.update_student_factor(factor, log_det);
            })
        .def(
            "loglik",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                scar::LogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.loglik(params);
                }
                return loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                scar::GradLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.neg_loglik_with_grad(params);
                }
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_and_corr",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                scar::GradLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.neg_loglik_with_grad_and_corr(params);
                }
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_and_corr_directional",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                const std::vector<double> direction =
                    vector_from_array(corr_direction);
                scar::GradLogLikResult result;
                {
                    py::gil_scoped_release release;
                    result =
                        evaluator.neg_loglik_with_grad_and_corr_directional(
                            params, direction);
                }
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "predictive_mean",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                std::vector<double> values;
                {
                    py::gil_scoped_release release;
                    values = evaluator.predictive_mean(params, backend, status);
                }
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "mixture_h",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                std::vector<double> values;
                {
                    py::gil_scoped_release release;
                    values = evaluator.mixture_h(params, backend, status);
                }
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "mixture_h_pair",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                std::vector<double> values;
                {
                    py::gil_scoped_release release;
                    values = evaluator.mixture_h_pair(params, backend, status);
                }
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "state_distribution",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               bool horizon_next) {
                scar::StateDistribution result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.state_distribution(params, horizon_next);
                }
                return state_distribution_to_dict(result);
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
                const scar::LogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.loglik_spectral(
                                params, copula, obs, config);
                        });
                return loglik_result_to_dict(result);
            })
        .def(
            "loglik_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::LogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.loglik_local_gh(
                                params, copula, obs, config);
                        });
                return loglik_result_to_dict(result);
            })
        .def(
            "loglik_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::LogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.loglik_matrix(
                                params, copula, obs, config);
                        });
                return loglik_result_to_dict(result);
            })
        .def(
            "loglik_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::LogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.loglik_auto(
                                params, copula, obs, config);
                        });
                return loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_spectral",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_spectral(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_local_gh(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_matrix(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_and_corr_spectral",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_and_corr_spectral(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
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
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator
                                .neg_loglik_with_grad_and_corr_directional_spectral(
                                    params, copula, obs, config, direction);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_and_corr_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_and_corr_local_gh(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
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
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator
                                .neg_loglik_with_grad_and_corr_directional_local_gh(
                                    params, copula, obs, config, direction);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_and_corr_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_and_corr_matrix(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
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
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator
                                .neg_loglik_with_grad_and_corr_directional_matrix(
                                    params, copula, obs, config, direction);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_and_corr_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_and_corr_auto(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
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
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator
                                .neg_loglik_with_grad_and_corr_directional_auto(
                                    params, copula, obs, config, direction);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "neg_loglik_with_grad_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::GradLogLikResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.neg_loglik_with_grad_auto(
                                params, copula, obs, config);
                        });
                return grad_loglik_result_to_dict(result);
            })
        .def(
            "predictive_mean_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.predictive_mean_local_gh(
                                params, copula, obs, config, status);
                        });
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
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.predictive_mean_matrix(
                                params, copula, obs, config, status);
                        });
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
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.predictive_mean_auto(
                                params, copula, obs, config, backend, status);
                        });
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
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_local_gh(
                                params, copula, obs, config, status);
                        });
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
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_matrix(
                                params, copula, obs, config, status);
                        });
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
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_auto(
                                params, copula, obs, config, backend, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "mixture_h_pair_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_pair_local_gh(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::LocalGh));
            })
        .def(
            "mixture_h_pair_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_pair_matrix(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::Matrix));
            })
        .def(
            "mixture_h_pair_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                scar::OuBackend backend = scar::OuBackend::Matrix;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_pair_auto(
                                params, copula, obs, config, backend, status);
                        });
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
                const scar::StateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.state_distribution_local_gh(
                                params, copula, obs, config, horizon_next);
                        });
                return state_distribution_to_dict(result);
            })
        .def(
            "state_distribution_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                const scar::StateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.state_distribution_matrix(
                                params, copula, obs, config, horizon_next);
                        });
                return state_distribution_to_dict(result);
            })
        .def(
            "state_distribution_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                const scar::StateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.state_distribution_auto(
                                params, copula, obs, config, horizon_next);
                        });
                return state_distribution_to_dict(result);
            });
}

}  // namespace pyscarcopula::bindings
