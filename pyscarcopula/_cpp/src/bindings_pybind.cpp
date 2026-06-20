#include "scar/copula.hpp"
#include "scar/gas.hpp"
#include "scar/ou.hpp"
#include "scar_internal.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

scar::Observations observations_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> u) {

    const py::buffer_info info = u.request();
    if (info.ndim != 2 || info.shape[1] < 2) {
        throw std::invalid_argument(
            "u must be a 2D float64 array with shape (n, d), d >= 2");
    }
    const auto n_obs = static_cast<std::int64_t>(info.shape[0]);
    const auto dim = static_cast<std::int64_t>(info.shape[1]);
    std::size_t n_obs_size = 0;
    std::size_t dim_size = 0;
    std::size_t input_size = 0;
    if (!scar_internal::checked_nonnegative_size(n_obs, n_obs_size)
        || !scar_internal::checked_nonnegative_size(dim, dim_size)
        || !scar_internal::checked_size_mul(
            n_obs_size, dim_size, input_size)) {
        throw std::invalid_argument("u shape is not representable");
    }
    const double* data = static_cast<const double*>(info.ptr);

    scar::Observations out(n_obs_size);
    for (std::int64_t i = 0; i < n_obs; ++i) {
        std::vector<double> row(static_cast<std::size_t>(dim), 0.0);
        for (std::int64_t j = 0; j < dim; ++j) {
            const double value = data[i * dim + j];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("u must contain only finite values");
            }
            row[static_cast<std::size_t>(j)] = value;
        }
        out[static_cast<std::size_t>(i)] = std::move(row);
    }
    return out;
}

scar::ObservationView observation_view_from_array(
    const scar::CopulaSpec& copula,
    py::array_t<double, py::array::c_style | py::array::forcecast> u) {

    const py::buffer_info info = u.request();
    const int expected_dim =
        (copula.family == scar::CopulaFamily::Student
         || copula.family == scar::CopulaFamily::EquicorrGaussian)
        ? copula.dim
        : 2;
    if (info.ndim != 2 || info.shape[1] != expected_dim) {
        throw std::invalid_argument(
            "u must be a 2D float64 array with shape "
            "(n, expected copula dimension)");
    }

    std::size_t n_obs = 0;
    std::size_t dim = 0;
    std::size_t size = 0;
    if (!scar_internal::checked_nonnegative_size(info.shape[0], n_obs)
        || !scar_internal::checked_nonnegative_size(info.shape[1], dim)
        || !scar_internal::checked_size_mul(n_obs, dim, size)) {
        throw std::invalid_argument("u shape is not representable");
    }
    const double* data = static_cast<const double*>(info.ptr);
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument("u must contain only finite values");
        }
    }
    return {data, n_obs, expected_dim};
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

std::vector<double> flat_vector_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    const char* name) {

    const py::buffer_info info = values.request();
    const double* data = static_cast<const double*>(info.ptr);
    std::size_t size = 1;
    for (py::ssize_t extent : info.shape) {
        std::size_t extent_size = 0;
        if (!scar_internal::checked_nonnegative_size(
                static_cast<std::int64_t>(extent), extent_size)
            || !scar_internal::checked_size_mul(size, extent_size, size)) {
            throw std::invalid_argument(
                std::string(name) + " shape is not representable");
        }
    }
    std::vector<double> out(size);
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(
                std::string(name) + " must contain only finite values");
        }
        out[i] = data[i];
    }
    return out;
}

std::vector<int> int_vector_from_array(
    py::array_t<int, py::array::c_style | py::array::forcecast> values,
    const char* name) {

    const py::buffer_info info = values.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(
            std::string(name) + " must be a 1D integer array");
    }
    const int* data = static_cast<const int*>(info.ptr);
    return std::vector<int>(
        data, data + static_cast<std::size_t>(info.shape[0]));
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

py::array_t<double> grid_values_to_array(const scar::GridValues& values) {
    py::array_t<double> out({
        static_cast<py::ssize_t>(values.n_obs),
        static_cast<py::ssize_t>(values.n_grid),
    });
    py::buffer_info info = out.request();
    double* data = static_cast<double*>(info.ptr);
    std::copy(values.values.begin(), values.values.end(), data);
    return out;
}

void set_student_ppf_cache(
    scar::CopulaSpec& spec,
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> table) {

    const py::buffer_info nodes_info = nodes.request();
    const py::buffer_info table_info = table.request();
    if (nodes_info.ndim != 1) {
        throw std::invalid_argument("PPF nodes must be a 1D float64 array");
    }
    if (nodes_info.shape[0] < 2) {
        throw std::invalid_argument("PPF nodes must contain at least two values");
    }
    if (table_info.ndim != 3
        || table_info.shape[0] != nodes_info.shape[0]
        || table_info.shape[1] <= 0
        || table_info.shape[2] != spec.dim) {
        throw std::invalid_argument(
            "PPF table must have shape (n_nodes, n_obs, copula.dim)");
    }

    std::size_t n_nodes = 0;
    std::size_t n_obs_size = 0;
    std::size_t dim = 0;
    std::size_t rows = 0;
    std::size_t table_size = 0;
    if (!scar_internal::checked_nonnegative_size(
            nodes_info.shape[0], n_nodes)
        || !scar_internal::checked_nonnegative_size(
            table_info.shape[1], n_obs_size)
        || !scar_internal::checked_nonnegative_size(
            table_info.shape[2], dim)
        || !scar_internal::checked_size_mul(
            n_nodes, n_obs_size, rows)
        || !scar_internal::checked_size_mul(rows, dim, table_size)) {
        throw std::invalid_argument(
            "PPF table shape is not representable");
    }
    const auto n_obs = static_cast<std::int64_t>(n_obs_size);
    const double* nodes_data =
        static_cast<const double*>(nodes_info.ptr);
    const double* table_data =
        static_cast<const double*>(table_info.ptr);

    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (!std::isfinite(nodes_data[i])) {
            throw std::invalid_argument(
                "PPF nodes must contain only finite values");
        }
        if (i > 0 && nodes_data[i] <= nodes_data[i - 1]) {
            throw std::invalid_argument(
                "PPF nodes must be strictly increasing");
        }
    }
    for (std::size_t i = 0; i < table_size; ++i) {
        if (!std::isfinite(table_data[i])) {
            throw std::invalid_argument(
                "PPF table must contain only finite values");
        }
    }

    spec.ppf_nodes.assign(nodes_data, nodes_data + n_nodes);
    spec.ppf_table.assign(table_data, table_data + table_size);
    spec.ppf_n_obs = n_obs;
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
    out["neg_corr_gradient"] = vector_to_array(result.neg_corr_gradient);
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

py::dict gas_loglik_result_to_dict(const scar::GasLogLikResult& result) {
    py::dict out;
    out["log_likelihood"] = result.log_likelihood;
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict static_objective_result_to_dict(
    const scar::StaticObjectiveResult& result) {

    py::dict out;
    out["negative_log_likelihood"] = result.negative_log_likelihood;
    out["negative_gradient"] = result.negative_gradient;
    out["negative_correlation_gradient"] =
        vector_to_array(result.negative_correlation_gradient);
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict multivariate_rows_result_to_dict(
    const scar::MultivariateRowsResult& result) {

    py::dict out;
    out["log_pdf"] = vector_to_array(result.log_pdf);
    out["dlog_dr"] = vector_to_array(result.dlog_dr);
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict multivariate_grid_result_to_dict(
    const scar::MultivariateGridResult& result) {

    py::dict out;
    out["pdf"] = grid_values_to_array(result.pdf);
    out["d_pdf_dx"] = grid_values_to_array(result.d_pdf_dx);
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict conditional_sample_result_to_dict(
    const scar::ConditionalSampleResult& result) {

    py::array_t<double> values({
        static_cast<py::ssize_t>(result.n_rows),
        static_cast<py::ssize_t>(result.n_free),
    });
    py::buffer_info info = values.request();
    double* data = static_cast<double*>(info.ptr);
    std::copy(result.values.begin(), result.values.end(), data);
    py::dict out;
    out["values"] = std::move(values);
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict trajectory_log_pdf_result_to_dict(
    const scar::TrajectoryLogPdfResult& result) {

    py::dict out;
    out["log_pdf"] = grid_values_to_array(result.log_pdf);
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict gas_filter_result_to_dict(const scar::GasFilterResult& result) {
    py::dict out;
    out["g_path"] = vector_to_array(result.g_path);
    out["r_path"] = vector_to_array(result.r_path);
    out["score_path"] = vector_to_array(result.score_path);
    out["log_likelihood"] = result.log_likelihood;
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict gas_update_result_to_dict(const scar::GasUpdateResult& result) {
    py::dict out;
    out["g_next"] = result.g_next;
    out["r"] = result.r;
    out["r_next"] = result.r_next;
    out["log_likelihood"] = result.log_likelihood;
    out["score"] = result.score;
    out["status"] = result.status;
    return out;
}

py::dict gas_state_result_to_dict(const scar::GasStateResult& result) {
    py::dict out;
    out["g"] = result.g;
    out["parameter"] = result.parameter;
    out["status"] = result.status;
    return out;
}

py::dict gas_predict_result_to_dict(const scar::GasPredictResult& result) {
    py::dict out;
    out["parameter"] = result.parameter;
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict gas_path_result_to_dict(const scar::GasPathResult& result) {
    py::dict out;
    out["values"] = vector_to_array(result.values);
    out["status"] = result.status;
    out["failure_index"] = result.failure_index;
    return out;
}

py::dict hermite_rule_cache_info_to_dict(
    const scar_internal::HermiteRuleCacheInfo& info) {

    py::dict out;
    out["entries"] = info.entries;
    out["bytes"] = info.bytes;
    out["max_entries"] = info.max_entries;
    out["max_bytes"] = info.max_bytes;
    out["hits"] = info.hits;
    out["misses"] = info.misses;
    out["insertions"] = info.insertions;
    out["evictions"] = info.evictions;
    out["oversized_skips"] = info.oversized_skips;
    out["duplicate_builds"] = info.duplicate_builds;
    return out;
}

}  // namespace

PYBIND11_MODULE(_scar_cpp, m) {
    m.doc() = "pybind11 bindings for the pyscarcopula SCAR C++ kernels";
    py::register_exception_translator([](std::exception_ptr exception) {
        if (!exception) {
            return;
        }
        try {
            std::rethrow_exception(exception);
        } catch (const std::bad_alloc&) {
            PyErr_SetString(
                PyExc_MemoryError,
                "C++ SCAR kernel allocation failed after size validation");
        }
    });

    m.attr("MAX_GRID_SIZE") = py::int_(scar_internal::kMaxGridSize);
    m.attr("MAX_DENSE_GRID_SIZE") =
        py::int_(scar_internal::kMaxDenseGridSize);
    m.attr("MAX_SPECTRAL_ORDER") =
        py::int_(scar_internal::kMaxSpectralOrder);
    m.attr("PSEUDO_OBS_EPS") =
        py::float_(scar_internal::kPseudoObsEps);
    m.attr("H_FUNCTION_EPS") =
        py::float_(scar_internal::kHEps);
    m.attr("PDF_FLOOR") =
        py::float_(scar_internal::kPdfEps);
    m.attr("HERMITE_RULE_CACHE_MAX_ENTRIES") =
        py::int_(scar_internal::kHermiteRuleCacheMaxEntries);
    m.attr("HERMITE_RULE_CACHE_MAX_BYTES") =
        py::int_(scar_internal::kHermiteRuleCacheMaxBytes);
    m.attr("SCAR_OK") = py::int_(scar::SCAR_OK);
    m.attr("SCAR_NULL_POINTER") = py::int_(scar::SCAR_NULL_POINTER);
    m.attr("SCAR_INVALID_SIZE") = py::int_(scar::SCAR_INVALID_SIZE);
    m.attr("SCAR_INVALID_FAMILY") = py::int_(scar::SCAR_INVALID_FAMILY);
    m.attr("SCAR_INVALID_ROTATION") = py::int_(scar::SCAR_INVALID_ROTATION);
    m.attr("SCAR_INVALID_TRANSFORM") = py::int_(scar::SCAR_INVALID_TRANSFORM);
    m.attr("SCAR_INVALID_PARAMETER") = py::int_(scar::SCAR_INVALID_PARAMETER);
    m.attr("SCAR_NUMERICAL_FAILURE") =
        py::int_(scar::SCAR_NUMERICAL_FAILURE);

    m.def(
        "_student_quantile",
        &scar_internal::student_quantile_value,
        py::arg("p"),
        py::arg("df"));
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

    py::enum_<scar::CopulaFamily>(m, "CopulaFamily")
        .value("Independent", scar::CopulaFamily::Independent)
        .value("Clayton", scar::CopulaFamily::Clayton)
        .value("Gumbel", scar::CopulaFamily::Gumbel)
        .value("Frank", scar::CopulaFamily::Frank)
        .value("Joe", scar::CopulaFamily::Joe)
        .value("Gaussian", scar::CopulaFamily::Gaussian)
        .value("Student", scar::CopulaFamily::Student)
        .value(
            "EquicorrGaussian",
            scar::CopulaFamily::EquicorrGaussian)
        .value(
            "MultivariateGaussian",
            scar::CopulaFamily::MultivariateGaussian);

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

    py::enum_<scar::GasScaling>(m, "GasScaling")
        .value("Unit", scar::GasScaling::Unit)
        .value("Fisher", scar::GasScaling::Fisher);

    py::class_<scar::CopulaSpec>(m, "CopulaSpec")
        .def(py::init<>())
        .def_readwrite("family", &scar::CopulaSpec::family)
        .def_readwrite("rotation", &scar::CopulaSpec::rotation)
        .def_readwrite("transform", &scar::CopulaSpec::transform)
        .def_readwrite("offset", &scar::CopulaSpec::offset)
        .def_readwrite("dim", &scar::CopulaSpec::dim)
        .def_readwrite("l_inv", &scar::CopulaSpec::l_inv)
        .def_readwrite("log_det", &scar::CopulaSpec::log_det)
        .def_readwrite("ppf_n_obs", &scar::CopulaSpec::ppf_n_obs)
        .def_readwrite("ppf_nodes", &scar::CopulaSpec::ppf_nodes)
        .def_readwrite("ppf_table", &scar::CopulaSpec::ppf_table)
        .def(
            "set_student_ppf_cache",
            &set_student_ppf_cache,
            py::arg("nodes"),
            py::arg("table"));

    py::class_<scar::StaticCopulaEvaluator>(m, "StaticCopulaEvaluator")
        .def(
            py::init([](
                const scar::CopulaSpec& copula,
                py::array_t<
                    double,
                    py::array::c_style | py::array::forcecast> u) {
                return scar::StaticCopulaEvaluator(
                    copula, observations_from_array(u));
            }),
            py::arg("copula"),
            py::arg("u"))
        .def(
            "objective",
            [](const scar::StaticCopulaEvaluator& evaluator,
               double parameter) {
                scar::StaticObjectiveResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.objective(parameter);
                }
                return static_objective_result_to_dict(
                    result);
            },
            py::arg("parameter"))
        .def(
            "objective_with_correlation_gradient",
            [](const scar::StaticCopulaEvaluator& evaluator,
               double parameter) {
                scar::StaticObjectiveResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.objective(parameter, true);
                }
                return static_objective_result_to_dict(result);
            },
            py::arg("parameter"))
        .def(
            "log_pdf_rows",
            [](const scar::StaticCopulaEvaluator& evaluator,
               double parameter) {
                std::vector<double> result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.log_pdf_rows(parameter);
                }
                return vector_to_array(result);
            },
            py::arg("parameter"))
        .def_property_readonly(
            "status",
            &scar::StaticCopulaEvaluator::status);

    m.def(
        "copula_transform",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            return vector_to_array(
                scar::copula_transform(copula, vector_from_array(x)));
        },
        py::arg("copula"),
        py::arg("x"));

    m.def(
        "copula_inverse_transform",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(
                scar::copula_inverse_transform(copula, vector_from_array(r)));
        },
        py::arg("copula"),
        py::arg("r"));

    m.def(
        "copula_dtransform",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            return vector_to_array(
                scar::copula_dtransform(copula, vector_from_array(x)));
        },
        py::arg("copula"),
        py::arg("x"));

    m.def(
        "copula_tau_to_param",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               tau) {
            return vector_to_array(
                scar::copula_tau_to_param(
                    copula, vector_from_array(tau)));
        },
        py::arg("copula"),
        py::arg("tau"));

    m.def(
        "copula_param_to_tau",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(
                scar::copula_param_to_tau(
                    copula, vector_from_array(r)));
        },
        py::arg("copula"),
        py::arg("r"));

    m.def(
        "copula_log_pdf",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(scar::copula_log_pdf(
                copula, observations_from_array(u), vector_from_array(r)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"));

    m.def(
        "copula_pdf",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(scar::copula_pdf(
                copula, observations_from_array(u), vector_from_array(r)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"));

    m.def(
        "copula_dlog_pdf_dr",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(scar::copula_dlog_pdf_dr(
                copula, observations_from_array(u), vector_from_array(r)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"));

    m.def(
        "copula_h",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            return vector_to_array(
                scar::copula_h(
                    copula, observations_from_array(u), vector_from_array(r)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"));

    m.def(
        "copula_h_pair",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            const auto result = scar::copula_h_pair(
                copula, observations_from_array(u), vector_from_array(r));
            return py::make_tuple(
                vector_to_array(result.first),
                vector_to_array(result.second));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"));

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
        },
        py::arg("copula"),
        py::arg("q_given"),
        py::arg("r"));

    m.def(
        "copula_pdf_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               x_grid) {
            return grid_values_to_array(scar::copula_pdf_grid(
                copula,
                observations_from_array(u),
                vector_from_array(x_grid)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("x_grid"));

    m.def(
        "copula_pdf_and_grad_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               x_grid) {
            const auto result = scar::copula_pdf_and_grad_grid(
                copula,
                observations_from_array(u),
                vector_from_array(x_grid));
            return py::make_tuple(
                grid_values_to_array(result.pdf),
                grid_values_to_array(result.d_pdf_dx));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("x_grid"));

    m.def(
        "copula_pdf_parameter_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               r_grid) {
            return grid_values_to_array(scar::copula_pdf_parameter_grid(
                copula,
                observations_from_array(u),
                vector_from_array(r_grid)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r_grid"));

    m.def(
        "copula_h_parameter_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               r_grid) {
            return grid_values_to_array(scar::copula_h_parameter_grid(
                copula,
                observations_from_array(u),
                vector_from_array(r_grid)));
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r_grid"));

    m.def(
        "copula_log_pdf_trajectory_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               latent_paths) {
            const scar::ObservationView observations =
                observation_view_from_array(copula, u);
            const py::buffer_info paths_info = latent_paths.request();
            if (paths_info.ndim != 2
                || paths_info.shape[0]
                    != static_cast<py::ssize_t>(observations.n_obs)
                || paths_info.shape[1] <= 0) {
                throw std::invalid_argument(
                    "latent_paths must have shape (n_obs, n_trajectories)");
            }
            const double* paths =
                static_cast<const double*>(paths_info.ptr);
            scar::TrajectoryLogPdfResult result;
            {
                py::gil_scoped_release release;
                result = scar::copula_log_pdf_trajectory_grid(
                    copula,
                    observations,
                    paths,
                    static_cast<std::size_t>(paths_info.shape[1]));
            }
            return trajectory_log_pdf_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("latent_paths"));

    m.def(
        "multivariate_log_pdf_and_grad",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r,
           std::int64_t row_offset) {
            scar::MultivariateRowsResult result;
            const scar::Observations observations =
                observations_from_array(u);
            const std::vector<double> parameters = vector_from_array(r);
            {
                py::gil_scoped_release release;
                result = scar::multivariate_log_pdf_and_grad(
                    copula, observations, parameters, row_offset);
            }
            return multivariate_rows_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"),
        py::arg("row_offset") = 0);

    m.def(
        "multivariate_pdf_and_grad_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               x_grid,
           std::int64_t row_offset) {
            scar::MultivariateGridResult result;
            const scar::Observations observations =
                observations_from_array(u);
            const std::vector<double> grid = vector_from_array(x_grid);
            {
                py::gil_scoped_release release;
                result = scar::multivariate_pdf_and_grad_grid(
                    copula, observations, grid, row_offset);
            }
            return multivariate_grid_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("x_grid"),
        py::arg("row_offset") = 0);

    m.def(
        "multivariate_gaussian_conditional",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlations,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_latent,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws) {

            const py::buffer_info corr_info = correlations.request();
            const py::buffer_info draw_info = normal_draws.request();
            if ((corr_info.ndim != 2 && corr_info.ndim != 3)
                || corr_info.shape[corr_info.ndim - 1]
                    != corr_info.shape[corr_info.ndim - 2]
                || draw_info.ndim != 2) {
                throw std::invalid_argument(
                    "correlations must be (d,d) or (n,d,d), and "
                    "normal_draws must be (n,n_free)");
            }
            const int dimension = static_cast<int>(
                corr_info.shape[corr_info.ndim - 1]);
            const std::int64_t correlation_rows =
                corr_info.ndim == 2 ? 1 : corr_info.shape[0];
            const std::int64_t n_rows = draw_info.shape[0];
            scar::ConditionalSampleResult result;
            {
                const auto corr = flat_vector_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto latent = flat_vector_from_array(
                    given_latent, "given_latent");
                const auto draws = flat_vector_from_array(
                    normal_draws, "normal_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_conditional(
                    corr, correlation_rows, dimension, indices,
                    latent, draws, n_rows);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_latent"),
        py::arg("normal_draws"));

    m.def(
        "multivariate_student_conditional",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlations,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_latent,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_draws) {

            const py::buffer_info corr_info = correlations.request();
            const py::buffer_info draw_info = normal_draws.request();
            if ((corr_info.ndim != 2 && corr_info.ndim != 3)
                || corr_info.shape[corr_info.ndim - 1]
                    != corr_info.shape[corr_info.ndim - 2]
                || draw_info.ndim != 2) {
                throw std::invalid_argument(
                    "correlations must be (d,d) or (n,d,d), and "
                    "normal_draws must be (n,n_free)");
            }
            const int dimension = static_cast<int>(
                corr_info.shape[corr_info.ndim - 1]);
            const std::int64_t correlation_rows =
                corr_info.ndim == 2 ? 1 : corr_info.shape[0];
            const std::int64_t n_rows = draw_info.shape[0];
            scar::ConditionalSampleResult result;
            {
                const auto corr = flat_vector_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto latent = flat_vector_from_array(
                    given_latent, "given_latent");
                const auto degrees = flat_vector_from_array(df, "df");
                const auto draws = flat_vector_from_array(
                    normal_draws, "normal_draws");
                const auto chi = flat_vector_from_array(
                    chi_square_draws, "chi_square_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_student_conditional(
                    corr, correlation_rows, dimension, indices,
                    latent, degrees, draws, chi, n_rows);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_latent"),
        py::arg("df"),
        py::arg("normal_draws"),
        py::arg("chi_square_draws"));

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
