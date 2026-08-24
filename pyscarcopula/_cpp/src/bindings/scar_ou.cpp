#include "array.hpp"
#include "module.hpp"

#include "scar/ou.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

py::list backend_chain_to_list(const std::vector<scar::OuBackend>& chain) {
    py::list output;
    for (const scar::OuBackend backend : chain) {
        output.append(static_cast<int>(backend));
    }
    return output;
}

py::dict loglik_result_to_dict(const scar::LogLikResult& result) {
    py::dict output;
    output["log_likelihood"] = result.log_likelihood;
    output["backend"] = static_cast<int>(result.backend);
    output["status"] = static_cast<int>(result.status);
    output["fallback_from"] = result.failure.fallback_from;
    output["fallback_chain"] = backend_chain_to_list(result.fallback_chain);
    output["matrix_fallback_reason"] = result.matrix_fallback_reason;
    return output;
}

py::dict grad_loglik_result_to_dict(const scar::GradLogLikResult& result) {
    py::dict output;
    output["neg_log_likelihood"] = result.neg_log_likelihood;
    output["neg_gradient"] = vector_to_array(result.neg_gradient);
    output["backend"] = static_cast<int>(result.backend);
    output["status"] = static_cast<int>(result.status);
    output["fallback_from"] = result.failure.fallback_from;
    output["fallback_chain"] = backend_chain_to_list(result.fallback_chain);
    output["matrix_fallback_reason"] = result.matrix_fallback_reason;
    output["neg_corr_gradient"] =
        vector_to_array(result.neg_corr_gradient);
    return output;
}

py::dict vector_result_to_dict(const scar::ScarOuVectorResult& result) {
    py::dict output;
    output["values"] = vector_to_array(result.values);
    output["status"] = static_cast<int>(result.status);
    output["backend"] = static_cast<int>(result.backend);
    return output;
}

py::dict state_distribution_to_dict(const scar::StateDistribution& result) {
    py::dict output;
    output["z_grid"] = vector_to_array(result.z_grid);
    output["prob"] = vector_to_array(result.prob);
    output["status"] = static_cast<int>(result.status);
    output["backend"] = static_cast<int>(result.backend);
    return output;
}

py::dict smoothed_state_distribution_to_dict(
    const scar::SmoothedStateDistribution& result) {

    py::dict output;
    output["z_grid"] = vector_to_array(result.z_grid);
    output["weights"] = matrix_to_array(
        result.weights,
        static_cast<std::size_t>(result.n_obs),
        static_cast<std::size_t>(result.K));
    output["status"] = static_cast<int>(result.status);
    output["backend"] = static_cast<int>(result.backend);
    return output;
}

py::dict trajectory_log_pdf_result_to_dict(
    const scar::TrajectoryLogPdfResult& result) {

    py::dict output;
    output["log_pdf"] = matrix_to_array(
        result.log_pdf.values,
        static_cast<std::size_t>(result.log_pdf.n_obs),
        static_cast<std::size_t>(result.log_pdf.n_grid));
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_blocks"] = result.parallel_blocks;
    return output;
}

py::dict hermite_rule_cache_info_to_dict(
    const scar_internal::HermiteRuleCacheInfo& info) {

    py::dict output;
    output["entries"] = info.entries;
    output["bytes"] = info.bytes;
    output["max_entries"] = info.max_entries;
    output["max_bytes"] = info.max_bytes;
    output["hits"] = info.hits;
    output["misses"] = info.misses;
    output["insertions"] = info.insertions;
    output["evictions"] = info.evictions;
    output["oversized_skips"] = info.oversized_skips;
    output["duplicate_builds"] = info.duplicate_builds;
    return output;
}

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
    observation_view_from_array(
        copula.model_descriptor().expected_dimension(), u);
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
        observation_view_from_array(
            copula.model_descriptor().expected_dimension(), u);
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

    return matrix_to_array(
        values,
        values.empty() ? 0 : static_cast<std::size_t>(rows),
        static_cast<std::size_t>(columns));
}

py::dict ou_grid_filter_result_to_dict(
    const scar::OuGridFilterResult& result) {

    py::dict out;
    out["z_grid"] = vector_to_array(result.z_grid);
    out["predictive_weights"] = filter_matrix_to_array(
        result.predictive_weights, result.n_obs, result.K);
    out["filtered_weights"] = filter_matrix_to_array(
        result.filtered_weights, result.n_obs, result.K);
    out["final_filtered_density"] =
        vector_to_array(result.final_filtered_density);
    out["backward_messages"] = filter_matrix_to_array(
        result.backward_messages, result.n_obs, result.K);
    out["smoothed_weights"] = filter_matrix_to_array(
        result.smoothed_weights, result.n_obs, result.K);
    out["n_obs"] = result.n_obs;
    out["K"] = result.K;
    out["backend"] = static_cast<int>(result.backend);
    out["sparse"] = result.sparse;
    out["status"] = static_cast<int>(result.status);
    out["failure_index"] = result.failure.index;
    out["failure_row"] = result.failure.row;
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
    if (static_cast<std::uint64_t>(info.shape[0])
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())
        || info.shape[1]
            > static_cast<py::ssize_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "emissions shape is not representable by the native API");
    }
    const std::int64_t n_obs = static_cast<std::int64_t>(info.shape[0]);
    const int columns = static_cast<int>(info.shape[1]);
    std::size_t value_count = 0;
    if (!scar::core::checked_size_mul(
            static_cast<std::size_t>(n_obs),
            static_cast<std::size_t>(columns),
            value_count)) {
        throw std::invalid_argument(
            "emissions shape is not representable by the native API");
    }
    const scar::DoubleView emission_view{
        static_cast<const double*>(info.ptr), value_count};
    scar::OuGridFilterResult result;
    {
        py::gil_scoped_release release;
        result = scar::filter_ou_grid_emissions(
            params,
            emission_view,
            n_obs,
            columns,
            config,
            backend,
            store_predictive,
            store_filtered,
            run_backward,
            run_smoothing);
    }
    return ou_grid_filter_result_to_dict(result);
}

}  // namespace

void bind_scar_ou(py::module_& m) {
    m.def(
        "copula_log_pdf_trajectory_grid",
        [](const scar::CopulaSpec& copula,
           Float64Array u,
           Float64Array latent_paths,
           int n_threads) {
            const scar::ObservationView observations =
                observation_view_from_array(
                    copula.model_descriptor().expected_dimension(), u);
            const py::buffer_info paths_info = latent_paths.request();
            if (paths_info.ndim != 2
                || paths_info.shape[0]
                    != static_cast<py::ssize_t>(observations.n_obs)
                || paths_info.shape[1] <= 0) {
                throw std::invalid_argument(
                    "latent_paths must have shape (n_obs, n_trajectories)");
            }
            scar::TrajectoryLogPdfResult result;
            {
                py::gil_scoped_release release;
                result = scar::copula_log_pdf_trajectory_grid(
                    copula,
                    observations,
                    static_cast<const double*>(paths_info.ptr),
                    static_cast<std::size_t>(paths_info.shape[1]),
                    n_threads);
            }
            return trajectory_log_pdf_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("latent_paths"),
        py::arg("n_threads") = 1);

    m.def(
        "_hermite_rule_cache_info",
        []() {
            return hermite_rule_cache_info_to_dict(
                scar_internal::hermite_rule_cache_info());
        });
    m.def(
        "_clear_hermite_rule_cache",
        &scar_internal::clear_hermite_rule_cache);
    m.def(
        "_set_hermite_rule_cache_limits_for_testing",
        &scar_internal::set_hermite_rule_cache_limits_for_testing,
        py::arg("max_entries"),
        py::arg("max_bytes"));
    m.def(
        "_reset_hermite_rule_cache_limits_for_testing",
        &scar_internal::reset_hermite_rule_cache_limits_for_testing);
    m.def(
        "_hermite_rule_for_testing",
        [](int quad_order, int basis_order) {
            std::vector<double> z;
            std::vector<double> weights;
            std::vector<double> basis;
            std::vector<double> weighted_basis;
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = scar_internal::standard_normal_hermite_rule_with_weighted_basis(
                    quad_order,
                    basis_order,
                    z,
                    weights,
                    basis,
                    weighted_basis);
            }
            if (!ok) {
                throw std::invalid_argument(
                    "invalid or numerically unstable Hermite rule");
            }
            return py::make_tuple(
                vector_to_array(z),
                vector_to_array(weights),
                vector_to_array(basis),
                vector_to_array(weighted_basis));
        },
        py::arg("quad_order"),
        py::arg("basis_order"));

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
                scar::ScarOuVectorResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.predictive_mean(params);
                }
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                scar::ScarOuVectorResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.mixture_h(params);
                }
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_pair",
            [](const scar::PreparedScarOuEvaluator& evaluator,
               const scar::OuParams& params) {
                scar::ScarOuVectorResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.mixture_h_pair(params);
                }
                return vector_result_to_dict(result);
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
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.predictive_mean_local_gh(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "predictive_mean_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.predictive_mean_matrix(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "predictive_mean_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.predictive_mean_auto(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "forward_rosenblatt_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.forward_rosenblatt_local_gh(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "forward_rosenblatt_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.forward_rosenblatt_matrix(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "forward_rosenblatt_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.forward_rosenblatt_auto(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "gaussian_rosenblatt_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.gaussian_rosenblatt_local_gh(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "gaussian_rosenblatt_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.gaussian_rosenblatt_matrix(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "gaussian_rosenblatt_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.gaussian_rosenblatt_auto(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "student_rosenblatt_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.student_rosenblatt_local_gh(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "student_rosenblatt_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.student_rosenblatt_matrix(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "student_rosenblatt_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.student_rosenblatt_auto(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_local_gh(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_matrix(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_auto(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_pair_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_pair_local_gh(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_pair_matrix",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_pair_matrix(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
            })
        .def(
            "mixture_h_pair_auto",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                const scar::ScarOuVectorResult result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.mixture_h_pair_auto(
                                params, copula, obs, config);
                        });
                return vector_result_to_dict(result);
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
