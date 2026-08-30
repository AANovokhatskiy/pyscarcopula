#include "array.hpp"
#include "module.hpp"

#include "scar/ou.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"
#include "scar/scar_ou/initialization.hpp"
#include "scar/scar_ou/parameterization.hpp"
#include "scar/scar_ou/policy.hpp"
#include "scar/scar_ou/quadrature.hpp"
#include "scar/scar_ou/sampling.hpp"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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

py::dict parameter_vector_to_dict(
    const scar::OuParameterVectorResult& result) {
    py::dict output;
    output["values"] = vector_to_array(result.value);
    output["status"] = static_cast<int>(result.status);
    return output;
}

py::dict initialization_to_dict(
    const scar::OuInitializationResult& result) {
    py::dict output;
    output["status"] = static_cast<int>(result.status);
    output["values"] = py::make_tuple(
        result.value.params.kappa,
        result.value.params.mu,
        result.value.params.nu);
    output["theta_mle"] = result.value.theta_mle;
    output["df_minus_two"] = result.value.theta_mle - 2.0;
    output["static_log_likelihood"] =
        result.value.static_log_likelihood;
    output["static_log_likelihood_per_observation"] =
        result.value.static_log_likelihood_per_observation;
    output["tau_abs"] = result.value.absolute_kendall_tau;
    output["strength"] = result.value.strength;
    output["sigma_x"] = result.value.stationary_scale;
    output["sigma_x_legacy"] = result.value.legacy_stationary_scale;
    output["rho_target"] = result.value.rho_target;
    switch (result.value.regime) {
    case scar::OuInitializationRegime::Weak:
        output["regime"] = "weak";
        break;
    case scar::OuInitializationRegime::Medium:
        output["regime"] = "medium";
        break;
    case scar::OuInitializationRegime::Strong:
        output["regime"] = "strong";
        break;
    default:
        output["regime"] = py::none();
        break;
    }
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

py::dict state_sample_to_dict(const scar::OuStateSampleResult& result) {
    py::dict output;
    output["values"] = vector_to_array(result.value.values);
    output["selection_draws_used"] = result.value.selection_draws_used;
    output["jitter_draws_used"] = result.value.jitter_draws_used;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
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
    py::object raw_u,
    const scar::OuNumericalConfig& config,
    const std::string& method) {
    auto u = real_float64_array_from_object(raw_u, "u");

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
    py::object raw_sum_z,
    py::object raw_sum_z2,
    const scar::OuNumericalConfig& config,
    const std::string& method) {
    auto sum_z = real_float64_array_from_object(raw_sum_z, "sum_z");
    auto sum_z2 = real_float64_array_from_object(raw_sum_z2, "sum_z2");

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
    m.def("model_ou_kappa_dt", [](double kappa, int n_obs) {
        const auto result = scar::ou_kappa_dt(kappa, n_obs);
        py::dict output;
        output["value"] = result.value;
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("kappa"), py::arg("n_obs"));
    m.def("model_ou_auto_backend", [](
            double kappa, int n_obs, double small_kdt) {
        const auto result = scar::ou_auto_backend(kappa, n_obs, small_kdt);
        py::dict output;
        output["backend"] = static_cast<int>(result.value);
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("kappa"), py::arg("n_obs"), py::arg("small_kdt"));
    m.def("model_ou_adaptive_spectral_basis_order", [](
            double kappa, int n_obs) {
        const auto result = scar::ou_adaptive_spectral_basis_order(
            kappa, n_obs);
        py::dict output;
        output["order"] = result.value;
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("kappa"), py::arg("n_obs"));
    m.def("model_ou_resolve_quad_order", [](
            int basis_order,
            const std::optional<int>& explicit_quad_order) {
        const auto result = scar::ou_resolve_quad_order(
            basis_order,
            explicit_quad_order.value_or(0),
            explicit_quad_order.has_value());
        py::dict output;
        output["order"] = result.value;
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("basis_order"), py::arg("explicit_quad_order"));
    py::enum_<scar::OuInitializationRegime>(
        m, "OuInitializationRegime")
        .value("NotApplicable", scar::OuInitializationRegime::NotApplicable)
        .value("Weak", scar::OuInitializationRegime::Weak)
        .value("Medium", scar::OuInitializationRegime::Medium)
        .value("Strong", scar::OuInitializationRegime::Strong);
    py::class_<scar::OuInitializationConfig>(
        m, "OuInitializationConfig")
        .def(py::init<>())
        .def_readwrite("rho_target",
            &scar::OuInitializationConfig::rho_target)
        .def_readwrite("sigma_fraction",
            &scar::OuInitializationConfig::sigma_fraction)
        .def_readwrite("weak_tau",
            &scar::OuInitializationConfig::weak_tau)
        .def_readwrite("strong_tau",
            &scar::OuInitializationConfig::strong_tau)
        .def_readwrite("weak_log_likelihood_per_observation",
            &scar::OuInitializationConfig::
                weak_log_likelihood_per_observation)
        .def_readwrite("strong_log_likelihood_per_observation",
            &scar::OuInitializationConfig::
                strong_log_likelihood_per_observation)
        .def_readwrite("weak_stationary_scale",
            &scar::OuInitializationConfig::weak_stationary_scale)
        .def_readwrite("maximum_stationary_scale",
            &scar::OuInitializationConfig::maximum_stationary_scale);
    m.def("ou_initial_kappa",
        [](std::size_t count, double rho_target,
           double kappa_min, double kappa_max) {
            const auto result = scar::ou_initial_kappa(
                count, rho_target, kappa_min, kappa_max);
            py::dict output;
            output["status"] = static_cast<int>(result.status);
            output["value"] = result.value;
            return output;
        }, py::arg("count"), py::arg("rho_target") = 0.96,
        py::arg("kappa_min") = 0.01, py::arg("kappa_max") = 100.0);
    m.def("ou_default_initial_point",
        [](double mu) {
            return initialization_to_dict(
                scar::ou_default_initial_point(mu));
        }, py::arg("mu"));
    m.def("ou_heuristic_initial_point",
        [](std::size_t count, double mu, double rho_target,
           double sigma_fraction) {
            return initialization_to_dict(
                scar::ou_heuristic_initial_point(
                    count, mu, rho_target, sigma_fraction));
        }, py::arg("count"), py::arg("mu"),
        py::arg("rho_target") = 0.95,
        py::arg("sigma_fraction") = 0.3);
    m.def("ou_stochastic_student_initial_point",
        [](std::size_t count, double theta_mle, double mu,
           double static_log_likelihood, double rho_target, double nu) {
            return initialization_to_dict(
                scar::ou_stochastic_student_initial_point(
                    count, theta_mle, mu, static_log_likelihood,
                    rho_target, nu));
        }, py::arg("count"), py::arg("theta_mle"), py::arg("mu"),
        py::arg("static_log_likelihood"),
        py::arg("rho_target") = 0.96, py::arg("nu") = 0.1);
    m.def("ou_strength_aware_initial_point",
        [](const Float64Array& observations, double theta_mle,
           double mu, double static_log_likelihood,
           const scar::OuInitializationConfig& config) {
            const py::buffer_info info = observations.request();
            if (info.ndim != 2 || info.shape[0] < 0 || info.shape[1] < 0) {
                throw std::invalid_argument(
                    "initialization observations must be 2D");
            }
            const scar::ObservationView view{
                static_cast<const double*>(info.ptr),
                static_cast<std::size_t>(info.shape[0]),
                static_cast<int>(info.shape[1])};
            scar::OuInitializationResult result;
            {
                py::gil_scoped_release release;
                result = scar::ou_strength_aware_initial_point(
                    view, theta_mle, mu, static_log_likelihood, config);
            }
            return initialization_to_dict(result);
        }, py::arg("observations"), py::arg("theta_mle"),
        py::arg("mu"), py::arg("static_log_likelihood"),
        py::arg("config"));
    m.def("ou_validate_trajectory_parameters",
        [](const scar::OuParams& params, std::size_t count) {
            return static_cast<int>(scar::validate_ou_trajectory_parameters(params, count));
        }, py::arg("params"), py::arg("count"));
    m.def("ou_sample_trajectory",
        [](const scar::OuParams& params, const Float64Array& normals) {
            const auto view = flat_view_from_array(normals, "standard_normals");
            scar::ScarOuVectorResult result;
            {
                py::gil_scoped_release release;
                result = scar::sample_ou_trajectory(params, view);
            }
            return vector_result_to_dict(result);
        }, py::arg("params"), py::arg("standard_normals"));
    m.def("ou_sample_stationary",
        [](const scar::OuParams& params, const Float64Array& normals) {
            const auto view = flat_view_from_array(normals, "standard_normals");
            scar::ScarOuVectorResult result;
            {
                py::gil_scoped_release release;
                result = scar::sample_ou_stationary(params, view);
            }
            return vector_result_to_dict(result);
        }, py::arg("params"), py::arg("standard_normals"));
    m.def("ou_sample_trajectory_block",
        [](const scar::OuParams& params, std::size_t total_count,
           double previous_state, bool initialize, py::object raw_normals) {
            const auto normals = real_float64_array_from_object(
                raw_normals, "standard_normals");
            const auto view = flat_view_from_array(normals, "standard_normals");
            scar::ScarOuVectorResult result;
            {
                py::gil_scoped_release release;
                result = scar::sample_ou_trajectory_block(
                    params, total_count, previous_state, initialize, view);
            }
            return vector_result_to_dict(result);
        }, py::arg("params"), py::arg("total_count"),
        py::arg("previous_state"), py::arg("initialize"),
        py::arg("standard_normals"));
    m.def("ou_sample_state_distribution",
        [](const Float64Array& z_grid,
           const Float64Array& probability,
           const Float64Array& selection_uniforms,
           const Float64Array& jitter_uniforms,
           bool histogram) {
            const auto states = flat_view_from_array(z_grid, "z_grid");
            const auto weights = flat_view_from_array(probability, "probability");
            const auto selection = flat_view_from_array(
                selection_uniforms, "selection_uniforms");
            const auto jitter = flat_view_from_array(
                jitter_uniforms, "jitter_uniforms");
            scar::OuStateSampleResult result;
            {
                py::gil_scoped_release release;
                result = scar::sample_ou_state_distribution(
                    states,
                    weights,
                    selection,
                    jitter,
                    histogram
                        ? scar::OuStateSamplingMode::Histogram
                        : scar::OuStateSamplingMode::Grid);
            }
            return state_sample_to_dict(result);
        },
        py::arg("z_grid"),
        py::arg("probability"),
        py::arg("selection_uniforms"),
        py::arg("jitter_uniforms"),
        py::arg("histogram"));
    m.def("ou_condition_state",
        [](const scar::CopulaSpec& copula,
           const Float64Array& z_grid,
           const Float64Array& probability,
           const Float64Array& observation) {
            const py::buffer_info info = observation.request();
            if (info.ndim != 2 || info.shape[0] != 1 || info.shape[1] < 1) {
                throw std::invalid_argument(
                    "observation must have shape (1, dimension)");
            }
            const auto states = flat_view_from_array(z_grid, "z_grid");
            const auto weights = flat_view_from_array(probability, "probability");
            const scar::ObservationView view{
                static_cast<const double*>(info.ptr),
                1,
                static_cast<int>(info.shape[1])};
            scar::StateDistribution result;
            {
                py::gil_scoped_release release;
                result = scar::condition_ou_state_distribution(
                    copula, states, weights, view);
            }
            return state_distribution_to_dict(result);
        },
        py::arg("copula"),
        py::arg("z_grid"),
        py::arg("probability"),
        py::arg("observation"));
    m.def("ou_trajectory_from_innovations",
        [](double x0, double mu, double rho, double sigma_cond,
           const Float64Array& innovations) {
            const auto view = flat_view_from_array(innovations, "innovations");
            scar::ScarOuVectorResult result;
            {
                py::gil_scoped_release release;
                result = scar::ou_trajectory_from_innovations(
                    x0, mu, rho, sigma_cond, view);
            }
            return vector_result_to_dict(result);
        }, py::arg("x0"), py::arg("mu"), py::arg("rho"),
        py::arg("sigma_cond"), py::arg("innovations"));
    m.def("ou_hermite_rule", [](int quad_order, int basis_order) {
        scar::Result<scar::OuHermiteRule> result;
        {
            py::gil_scoped_release release;
            result = scar::ou_hermite_rule(quad_order, basis_order);
        }
        py::dict output;
        output["status"] = static_cast<int>(result.status);
        output["nodes"] = vector_to_array(result.value.nodes);
        output["weights"] = vector_to_array(result.value.weights);
        output["basis"] = matrix_to_array(result.value.basis,
            result.value.quad_order, result.value.basis_order);
        return output;
    }, py::arg("quad_order"), py::arg("basis_order"));
    m.def("ou_default_quad_order", [](int basis_order) {
        const auto result = scar::ou_default_quad_order(basis_order);
        py::dict output;
        output["status"] = static_cast<int>(result.status);
        output["order"] = result.value;
        return output;
    }, py::arg("basis_order"));
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   corr_direction) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config,
               bool horizon_next) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
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
               py::object raw_u,
               const scar::OuNumericalConfig& config) {
                auto u = real_float64_array_from_object(raw_u, "u");
                const scar::SmoothedStateDistribution result =
                    with_observation_view_without_gil(
                        copula, u, [&](scar::ObservationView obs) {
                            return evaluator.smoothed_state_distribution_auto(
                                params, copula, obs, config);
                        });
                return smoothed_state_distribution_to_dict(result);
            });
    m.def("ou_to_log_stationary", [](const std::vector<double>& values) {
        scar::OuParameterVectorResult result;
        { py::gil_scoped_release release;
          result = scar::ou_to_log_stationary(values); }
        return parameter_vector_to_dict(result);
    }, py::arg("values"));
    m.def("ou_from_log_stationary", [](const std::vector<double>& values) {
        scar::OuParameterVectorResult result;
        { py::gil_scoped_release release;
          result = scar::ou_from_log_stationary(values); }
        return parameter_vector_to_dict(result);
    }, py::arg("values"));
    m.def("ou_gradient_to_log_stationary", [](
            const std::vector<double>& physical,
            const std::vector<double>& gradient) {
        scar::OuParameterVectorResult result;
        { py::gil_scoped_release release;
          result = scar::ou_gradient_to_log_stationary(
              physical, gradient); }
        return parameter_vector_to_dict(result);
    }, py::arg("physical"), py::arg("gradient"));
    m.def("ou_gradient_from_log_stationary", [](
            const std::vector<double>& physical,
            const std::vector<double>& gradient) {
        scar::OuParameterVectorResult result;
        { py::gil_scoped_release release;
          result = scar::ou_gradient_from_log_stationary(
              physical, gradient); }
        return parameter_vector_to_dict(result);
    }, py::arg("physical"), py::arg("gradient"));
    m.def("ou_project_optimizer_block", [](
            const std::vector<double>& values,
            const std::vector<double>& lower,
            const std::vector<double>& upper) {
        scar::OuParameterVectorResult result;
        { py::gil_scoped_release release;
          result = scar::ou_project_optimizer_block(
              values, lower, upper); }
        return parameter_vector_to_dict(result);
    }, py::arg("values"), py::arg("lower"), py::arg("upper"));
    m.def("optimizer_scaled_to_physical", [](
            const std::vector<double>& values,
            const std::vector<double>& scale) {
        return parameter_vector_to_dict(
            scar::optimizer_scaled_to_physical(values, scale));
    }, py::arg("values"), py::arg("scale"));
    m.def("physical_to_optimizer_scaled", [](
            const std::vector<double>& values,
            const std::vector<double>& scale) {
        return parameter_vector_to_dict(
            scar::physical_to_optimizer_scaled(values, scale));
    }, py::arg("values"), py::arg("scale"));
    m.def("gradient_to_optimizer_scaled", [](
            const std::vector<double>& gradient,
            const std::vector<double>& scale) {
        return parameter_vector_to_dict(
            scar::gradient_to_optimizer_scaled(gradient, scale));
    }, py::arg("gradient"), py::arg("scale"));
    m.def("gradient_from_optimizer_scaled", [](
            const std::vector<double>& gradient,
            const std::vector<double>& scale) {
        return parameter_vector_to_dict(
            scar::gradient_from_optimizer_scaled(gradient, scale));
    }, py::arg("gradient"), py::arg("scale"));
}

}  // namespace pyscarcopula::bindings
