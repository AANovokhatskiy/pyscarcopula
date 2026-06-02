#include "scar/copula.hpp"
#include "scar/ou.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

scar::Observations observations_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> u) {

    const py::buffer_info info = u.request();
    if (info.ndim != 2 || info.shape[1] != 2) {
        throw std::invalid_argument("u must be a 2D float64 array with shape (n, 2)");
    }
    const auto n_obs = static_cast<std::int64_t>(info.shape[0]);
    const double* data = static_cast<const double*>(info.ptr);

    scar::Observations out(static_cast<std::size_t>(n_obs));
    for (std::int64_t i = 0; i < n_obs; ++i) {
        const double first = data[2 * i];
        const double second = data[2 * i + 1];
        if (!std::isfinite(first) || !std::isfinite(second)) {
            throw std::invalid_argument("u must contain only finite values");
        }
        out[static_cast<std::size_t>(i)] = {first, second};
    }
    return out;
}

std::vector<double> vector_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> values) {

    const py::buffer_info info = values.request();
    const double* data = static_cast<const double*>(info.ptr);
    if (info.ndim == 0) {
        if (!std::isfinite(data[0])) {
            throw std::invalid_argument("values must contain only finite values");
        }
        return {data[0]};
    }
    if (info.ndim != 1) {
        throw std::invalid_argument("values must be a scalar or a 1D float64 array");
    }

    std::vector<double> out(static_cast<std::size_t>(info.shape[0]));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument("values must contain only finite values");
        }
        out[static_cast<std::size_t>(i)] = data[i];
    }
    return out;
}

py::array_t<double> vector_to_array(const std::vector<double>& values) {
    py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
    py::buffer_info info = out.request();
    double* data = static_cast<double*>(info.ptr);
    for (std::size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    return out;
}

py::list backend_chain_to_list(const std::vector<scar::OuBackend>& chain) {
    py::list out;
    for (const scar::OuBackend backend : chain) {
        out.append(static_cast<int>(backend));
    }
    return out;
}

py::dict loglik_result_to_dict(const scar::LogLikResult& result) {
    py::dict out;
    out["log_likelihood"] = result.log_likelihood;
    out["backend"] = static_cast<int>(result.backend);
    out["status"] = result.status;
    out["fallback_from"] = result.fallback_from;
    out["fallback_chain"] = backend_chain_to_list(result.fallback_chain);
    out["matrix_fallback_reason"] = result.matrix_fallback_reason;
    return out;
}

py::dict grad_loglik_result_to_dict(const scar::GradLogLikResult& result) {
    py::dict out;
    out["neg_log_likelihood"] = result.neg_log_likelihood;
    out["neg_gradient"] = vector_to_array(result.neg_gradient);
    out["backend"] = static_cast<int>(result.backend);
    out["status"] = result.status;
    out["fallback_from"] = result.fallback_from;
    out["fallback_chain"] = backend_chain_to_list(result.fallback_chain);
    out["matrix_fallback_reason"] = result.matrix_fallback_reason;
    return out;
}

py::dict vector_result_to_dict(
    const std::vector<double>& values,
    int status,
    int backend = -1) {

    py::dict out;
    out["values"] = vector_to_array(values);
    out["status"] = status;
    if (backend >= 0) {
        out["backend"] = backend;
    }
    return out;
}

py::dict state_distribution_to_dict(const scar::StateDistribution& result) {
    py::dict out;
    out["z_grid"] = vector_to_array(result.z_grid);
    out["prob"] = vector_to_array(result.prob);
    out["status"] = result.status;
    out["backend"] = static_cast<int>(result.backend);
    return out;
}

}  // namespace

PYBIND11_MODULE(_scar_cpp, m) {
    m.doc() = "pybind11 bindings for the pyscarcopula SCAR C++ kernels";

    py::enum_<scar::CopulaFamily>(m, "CopulaFamily")
        .value("Independent", scar::CopulaFamily::Independent)
        .value("Clayton", scar::CopulaFamily::Clayton)
        .value("Gumbel", scar::CopulaFamily::Gumbel)
        .value("Frank", scar::CopulaFamily::Frank)
        .value("Joe", scar::CopulaFamily::Joe)
        .value("Gaussian", scar::CopulaFamily::Gaussian);

    py::enum_<scar::Rotation>(m, "Rotation")
        .value("R0", scar::Rotation::R0)
        .value("R90", scar::Rotation::R90)
        .value("R180", scar::Rotation::R180)
        .value("R270", scar::Rotation::R270);

    py::enum_<scar::Transform>(m, "Transform")
        .value("Softplus", scar::Transform::Softplus)
        .value("XTanh", scar::Transform::XTanh)
        .value("GaussianTanh", scar::Transform::GaussianTanh);

    py::enum_<scar::OuBackend>(m, "OuBackend")
        .value("Spectral", scar::OuBackend::Spectral)
        .value("LocalGh", scar::OuBackend::LocalGh)
        .value("Matrix", scar::OuBackend::Matrix);

    py::class_<scar::CopulaSpec>(m, "CopulaSpec")
        .def(py::init<>())
        .def_readwrite("family", &scar::CopulaSpec::family)
        .def_readwrite("rotation", &scar::CopulaSpec::rotation)
        .def_readwrite("transform", &scar::CopulaSpec::transform)
        .def_readwrite("offset", &scar::CopulaSpec::offset);

    m.def(
        "copula_h",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(
                scar::copula_h(
                    copula, observations_from_array(u), vector_from_array(r)));
        });

    m.def(
        "copula_h_inverse",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> q_given,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(
                scar::copula_h_inverse(
                    copula,
                    observations_from_array(q_given),
                    vector_from_array(r)));
        });

    py::class_<scar::OuParams>(m, "OuParams")
        .def(py::init<>())
        .def_readwrite("kappa", &scar::OuParams::kappa)
        .def_readwrite("mu", &scar::OuParams::mu)
        .def_readwrite("nu", &scar::OuParams::nu);

    py::class_<scar::OuNumericalConfig>(m, "OuNumericalConfig")
        .def(py::init<>())
        .def_readwrite("K", &scar::OuNumericalConfig::K)
        .def_readwrite("grid_range", &scar::OuNumericalConfig::grid_range)
        .def_readwrite("adaptive", &scar::OuNumericalConfig::adaptive)
        .def_readwrite("pts_per_sigma", &scar::OuNumericalConfig::pts_per_sigma)
        .def_readwrite("max_K", &scar::OuNumericalConfig::max_K)
        .def_readwrite("r_gh", &scar::OuNumericalConfig::r_gh)
        .def_readwrite("gh_order", &scar::OuNumericalConfig::gh_order)
        .def_readwrite("auto_small_kdt", &scar::OuNumericalConfig::auto_small_kdt)
        .def_readwrite(
            "spectral_basis_order",
            &scar::OuNumericalConfig::spectral_basis_order)
        .def_readwrite(
            "spectral_quad_order",
            &scar::OuNumericalConfig::spectral_quad_order);

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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
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
                        params, copula, observations_from_array(u), config));
            })
        .def(
            "predictive_mean_local_gh",
            [](const scar::ScarOuEvaluator& evaluator,
               const scar::OuParams& params,
               const scar::CopulaSpec& copula,
               py::array_t<double, py::array::c_style | py::array::forcecast> u,
               const scar::OuNumericalConfig& config) {
                int status = 0;
                scar::Observations obs = observations_from_array(u);
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
                scar::Observations obs = observations_from_array(u);
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
                scar::Observations obs = observations_from_array(u);
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
                scar::Observations obs = observations_from_array(u);
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
                scar::Observations obs = observations_from_array(u);
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
                scar::Observations obs = observations_from_array(u);
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
                        params, copula, observations_from_array(u),
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
                        params, copula, observations_from_array(u),
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
                        params, copula, observations_from_array(u),
                        config, horizon_next));
            });
}
