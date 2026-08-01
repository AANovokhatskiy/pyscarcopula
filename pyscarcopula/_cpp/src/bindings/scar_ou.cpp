#include "common.hpp"

#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <cstring>
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

py::array_t<double> filter_matrix_to_array(
    const std::vector<double>& values,
    std::int64_t rows,
    int columns) {

    if (values.empty()) {
        return py::array_t<double>(
            py::array::ShapeContainer{
                static_cast<py::ssize_t>(0),
                static_cast<py::ssize_t>(columns)});
    }
    py::array_t<double> out(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(rows),
            static_cast<py::ssize_t>(columns)});
    std::memcpy(
        out.mutable_data(),
        values.data(),
        values.size() * sizeof(double));
    return out;
}

py::dict run_ou_grid_filter_engine(
    const scar::OuParams& params,
    py::array_t<
        double,
        py::array::c_style | py::array::forcecast> emissions,
    const scar::OuNumericalConfig& config,
    scar::OuBackend backend,
    bool store_predictive,
    bool store_filtered,
    bool run_backward,
    bool run_smoothing) {

    const py::buffer_info info = emissions.request();
    if (info.ndim != 2 || info.shape[0] < 2 || info.shape[1] < 2) {
        throw std::invalid_argument(
            "emissions must have shape (n_obs, K), with both dimensions >= 2");
    }
    const std::int64_t n_obs = static_cast<std::int64_t>(info.shape[0]);
    scar_internal::OuGrid grid;
    scar_internal::GridTransitionOperator transition;
    scar_internal::ForwardFilterOptions options;
    options.store_predictive_weights = store_predictive;
    options.store_filtered_weights = store_filtered;
    scar_internal::ForwardFilterResult forward;
    scar_internal::BackwardFilterResult backward;
    scar_internal::SmoothedStateResult smoothed;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = scar_internal::build_ou_grid(
                params.kappa,
                params.mu,
                params.nu,
                n_obs,
                config.K,
                config.grid_range,
                config.adaptive,
                config.pts_per_sigma,
                config.max_K,
                grid)
            && info.shape[1] == grid.K
            && scar_internal::build_grid_transition_operator(
                grid,
                backend,
                config.grid_method,
                config.gh_order,
                transition)
            && scar_internal::forward_filter_emissions(
                grid,
                transition,
                static_cast<const double*>(info.ptr),
                n_obs,
                options,
                forward)
            && (!run_backward
                || scar_internal::backward_filter_emissions(
                    grid,
                    transition,
                    static_cast<const double*>(info.ptr),
                    n_obs,
                    backward))
            && (!run_smoothing
                || scar_internal::smooth_state_emissions(
                    grid,
                    transition,
                    static_cast<const double*>(info.ptr),
                    n_obs,
                    smoothed));
    }
    if (!ok) {
        throw std::runtime_error("native OU grid filtering failed");
    }

    py::dict out;
    out["z_grid"] = vector_to_array(grid.x_grid);
    out["predictive_weights"] = filter_matrix_to_array(
        forward.predictive_weights, n_obs, grid.K);
    out["filtered_weights"] = filter_matrix_to_array(
        forward.filtered_weights, n_obs, grid.K);
    out["final_filtered_density"] =
        vector_to_array(forward.final_filtered_density);
    out["backward_messages"] = filter_matrix_to_array(
        backward.messages, n_obs, grid.K);
    out["smoothed_weights"] = filter_matrix_to_array(
        smoothed.weights, n_obs, grid.K);
    out["backend"] = static_cast<int>(backend);
    out["sparse"] = !transition.local_gh && transition.matrix.sparse;
    return out;
}

}  // namespace

void bind_scar_ou(py::module_& m) {
    m.def(
        "_ou_grid_filter_engine",
        &run_ou_grid_filter_engine,
        py::arg("params"),
        py::arg("emissions"),
        py::arg("config"),
        py::arg("backend"),
        py::arg("store_predictive") = true,
        py::arg("store_filtered") = true,
        py::arg("run_backward") = true,
        py::arg("run_smoothing") = true,
        "Private native forward/backward engine used by SCAR-OU endpoints.");

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
            "forward_rosenblatt_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.forward_rosenblatt_local_gh(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::LocalGh));
            })
        .def(
            "forward_rosenblatt_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.forward_rosenblatt_matrix(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::Matrix));
            })
        .def(
            "forward_rosenblatt_auto",
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
                            return evaluator.forward_rosenblatt_auto(
                                params, copula, obs, config, backend, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "gaussian_rosenblatt_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.gaussian_rosenblatt_local_gh(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::LocalGh));
            })
        .def(
            "gaussian_rosenblatt_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.gaussian_rosenblatt_matrix(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::Matrix));
            })
        .def(
            "gaussian_rosenblatt_auto",
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
                            return evaluator.gaussian_rosenblatt_auto(
                                params, copula, obs, config, backend, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(backend));
            })
        .def(
            "student_rosenblatt_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.student_rosenblatt_local_gh(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::LocalGh));
            })
        .def(
            "student_rosenblatt_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                std::vector<double> values =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.student_rosenblatt_matrix(
                                params, copula, obs, config, status);
                        });
                return vector_result_to_dict(
                    values, status, static_cast<int>(scar::OuBackend::Matrix));
            })
        .def(
            "student_rosenblatt_auto",
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
                            return evaluator.student_rosenblatt_auto(
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
            })
        .def(
            "smoothed_state_distribution_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::SmoothedStateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator
                                .smoothed_state_distribution_local_gh(
                                    params, copula, obs, config);
                        });
                return smoothed_state_distribution_to_dict(result);
            })
        .def(
            "smoothed_state_distribution_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::SmoothedStateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.smoothed_state_distribution_matrix(
                                params, copula, obs, config);
                        });
                return smoothed_state_distribution_to_dict(result);
            })
        .def(
            "smoothed_state_distribution_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::SmoothedStateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.smoothed_state_distribution_auto(
                                params, copula, obs, config);
                        });
                return smoothed_state_distribution_to_dict(result);
            });
}

}  // namespace pyscarcopula::bindings
