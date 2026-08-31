#include "array.hpp"
#include "module.hpp"

#include "scar/copula.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/model_policy.hpp"

#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

py::dict parameter_bounds_to_dict(
    const scar::ParameterBoundsResult& result) {
    py::dict output;
    output["lower"] = result.value.lower;
    output["upper"] = result.value.upper;
    output["status"] = static_cast<int>(result.status);
    return output;
}

py::dict fit_policy_to_dict(
    const scar::FitParameterPolicyResult& result) {
    py::dict output;
    output["initial"] = result.value.initial;
    output["lower"] = result.value.lower;
    output["upper"] = result.value.upper;
    output["has_lower"] = result.value.has_lower;
    output["has_upper"] = result.value.has_upper;
    output["status"] = static_cast<int>(result.status);
    return output;
}

py::dict optimizer_failure_to_dict(
    const scar::OptimizerFailureEvaluationResult& result) {
    py::dict output;
    output["objective"] = result.value.objective;
    output["gradient"] = vector_to_array(result.value.gradient);
    output["status"] = static_cast<int>(result.status);
    return output;
}

py::array_t<double> grid_values_to_array(const scar::GridValues& values) {
    return matrix_to_array(
        values.values,
        static_cast<std::size_t>(values.n_obs),
        static_cast<std::size_t>(values.n_grid));
}

py::array_t<double> pair_observations_to_array(
    const scar::Observations& observations) {

    std::vector<double> values;
    std::size_t size = 0;
    if (!scar::core::checked_size_mul(
            observations.size(), std::size_t{2}, size)) {
        throw std::invalid_argument(
            "pair sample output shape is not representable");
    }
    values.reserve(size);
    for (const std::vector<double>& row : observations) {
        if (row.size() != 2) {
            throw std::logic_error(
                "pair sampling kernel returned an invalid row width");
        }
        values.insert(values.end(), row.begin(), row.end());
    }
    return matrix_to_array(values, observations.size(), 2);
}

scar::Observations finite_rows_from_array(
    Float64Array values,
    const char* name) {

    const py::buffer_info info = values.request();
    if (info.ndim != 2 || info.shape[0] < 0 || info.shape[1] < 0) {
        throw std::invalid_argument(
            std::string(name) + " must be a finite 2D float64 array");
    }
    const std::size_t rows = static_cast<std::size_t>(info.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(info.shape[1]);
    const std::vector<double> flat = flat_vector_from_array(values, name);
    scar::Observations output(
        rows, std::vector<double>(columns, 0.0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            output[row][column] = flat[row * columns + column];
        }
    }
    return output;
}

void set_student_ppf_cache(
    scar::CopulaSpec& spec,
    Float64Array nodes,
    Float64Array table) {

    const py::buffer_info nodes_info = nodes.request();
    const py::buffer_info table_info = table.request();
    if (nodes_info.ndim != 1 || nodes_info.shape[0] < 2) {
        throw std::invalid_argument(
            "PPF nodes must be a 1D float64 array with at least two values");
    }
    if (table_info.ndim != 3
        || table_info.shape[0] != nodes_info.shape[0]
        || table_info.shape[1] <= 0
        || table_info.shape[2] != spec.dim) {
        throw std::invalid_argument(
            "PPF table must have shape (n_nodes, n_obs, copula.dim)");
    }

    const std::size_t node_count =
        static_cast<std::size_t>(nodes_info.shape[0]);
    const std::size_t observation_count =
        static_cast<std::size_t>(table_info.shape[1]);
    const std::size_t dimension =
        static_cast<std::size_t>(table_info.shape[2]);
    std::size_t rows = 0;
    std::size_t table_size = 0;
    if (!scar::core::checked_size_mul(
            node_count, observation_count, rows)
        || !scar::core::checked_size_mul(rows, dimension, table_size)) {
        throw std::invalid_argument("PPF table shape is not representable");
    }
    const auto* node_data = static_cast<const double*>(nodes_info.ptr);
    const auto* table_data = static_cast<const double*>(table_info.ptr);
    for (std::size_t index = 0; index < node_count; ++index) {
        if (!std::isfinite(node_data[index])) {
            throw std::invalid_argument(
                "PPF nodes must contain only finite values");
        }
        if (index > 0 && node_data[index] <= node_data[index - 1]) {
            throw std::invalid_argument(
                "PPF nodes must be strictly increasing");
        }
    }
    for (std::size_t index = 0; index < table_size; ++index) {
        if (!std::isfinite(table_data[index])) {
            throw std::invalid_argument(
                "PPF table must contain only finite values");
        }
    }

    spec.student_ppf_nodes().assign(node_data, node_data + node_count);
    spec.student_ppf_table().assign(table_data, table_data + table_size);
    spec.student_ppf_observation_count() =
        static_cast<std::int64_t>(observation_count);
}

py::dict static_objective_result_to_dict(
    const scar::StaticObjectiveResult& result) {

    py::dict output;
    output["negative_log_likelihood"] = result.negative_log_likelihood;
    output["negative_gradient"] = result.negative_gradient;
    output["negative_correlation_gradient"] =
        vector_to_array(result.negative_correlation_gradient);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_blocks"] = result.parallel_blocks;
    return output;
}

}  // namespace

void bind_copula(py::module_& m) {
    m.def("model_public_parameter_bounds",
        [](const scar::CopulaSpec& spec) {
            return parameter_bounds_to_dict(
                scar::model_public_parameter_bounds(spec));
        }, py::arg("spec"));
    m.def("model_ou_parameter_bounds", []() {
        return parameter_bounds_to_dict(scar::ou_parameter_bounds());
    });
    m.def("model_ou_scaled_optimizer_bounds",
        [](const std::vector<double>& scale) {
            return parameter_bounds_to_dict(
                scar::ou_scaled_optimizer_bounds(scale));
        }, py::arg("scale"));
    m.def("model_ou_log_stationary_parameter_bounds", []() {
        return parameter_bounds_to_dict(
            scar::ou_log_stationary_parameter_bounds());
    });
    m.def("model_ou_log_stationary_parameter_bounds_for_scale",
        [](const std::optional<double>& lower,
           const std::optional<double>& upper) {
            return parameter_bounds_to_dict(
                scar::ou_log_stationary_parameter_bounds_for_scale(
                    lower.value_or(0.0), lower.has_value(),
                    upper.value_or(0.0), upper.has_value()));
        }, py::arg("lower"), py::arg("upper"));
    m.def("model_ou_log_stationary_result_bounds", []() {
        return parameter_bounds_to_dict(
            scar::ou_log_stationary_result_bounds());
    });
    m.def("model_jacobi_parameter_bounds", []() {
        return parameter_bounds_to_dict(scar::jacobi_parameter_bounds());
    });
    m.def("model_gas_parameter_bounds",
        [](double gamma_bound, double beta_bound) {
            return parameter_bounds_to_dict(
                scar::gas_parameter_bounds(gamma_bound, beta_bound));
        }, py::arg("gamma_bound"), py::arg("beta_bound"));
    m.def("model_default_gas_parameter_bounds", []() {
        return parameter_bounds_to_dict(
            scar::default_gas_parameter_bounds());
    });
    m.def("model_normalize_positive_bounds",
        [](const std::optional<double>& lower,
           const std::optional<double>& upper) {
            return parameter_bounds_to_dict(
                scar::normalize_positive_bounds(
                    lower.value_or(0.0), lower.has_value(),
                    upper.value_or(0.0), upper.has_value()));
        }, py::arg("lower"), py::arg("upper"));
    m.def("model_stationary_scale_bounds", []() {
        return parameter_bounds_to_dict(scar::stationary_scale_bounds());
    });
    m.def("model_student_fit_policy",
        [](int dimension, bool stochastic) {
            return fit_policy_to_dict(
                scar::student_fit_parameter_policy(dimension, stochastic));
        }, py::arg("dimension"), py::arg("stochastic"));
    m.def("model_equicorr_fit_policy", []() {
        return fit_policy_to_dict(scar::equicorr_fit_parameter_policy());
    });
    m.def("model_default_pair_mle_parameter",
        [](const scar::CopulaSpec& spec) {
            const auto result = scar::default_pair_mle_parameter(spec);
            py::dict output;
            output["value"] = result.value;
            output["status"] = static_cast<int>(result.status);
            return output;
        }, py::arg("spec"));
    m.def("model_pair_mle_initial_parameter",
        [](const scar::CopulaSpec& spec, double tau) {
            const auto result = scar::pair_mle_initial_parameter(spec, tau);
            py::dict output;
            output["value"] = result.value;
            output["status"] = static_cast<int>(result.status);
            return output;
        }, py::arg("spec"), py::arg("tau"));
    m.def("model_gas_default_initial_point", [](double mu) {
        const auto result = scar::gas_default_initial_point(mu);
        py::dict output;
        output["values"] = vector_to_array(result.value);
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("mu"));
    m.def("model_optimizer_unit_scale",
        [](const std::vector<double>& parameters) {
            const auto result = scar::optimizer_unit_scale(parameters);
            py::dict output;
            output["values"] = vector_to_array(result.value);
            output["status"] = static_cast<int>(result.status);
            return output;
        }, py::arg("parameters"));
    m.def("model_project_optimizer_point", [](
            const std::vector<double>& parameters,
            const std::vector<double>& lower,
            const std::vector<double>& upper) {
        const auto result = scar::project_optimizer_point(
            parameters, lower, upper);
        py::dict output;
        output["values"] = vector_to_array(result.value);
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("parameters"), py::arg("lower"), py::arg("upper"));
    m.def("model_optimizer_failure_objective", []() {
        const auto result = scar::optimizer_failure_objective();
        py::dict output;
        output["objective"] = result.value;
        output["status"] = static_cast<int>(result.status);
        return output;
    });
    m.def("model_optimizer_failure_objective", [](double fail_value) {
        const auto result = scar::optimizer_failure_objective(fail_value);
        py::dict output;
        output["objective"] = result.value;
        output["status"] = static_cast<int>(result.status);
        return output;
    }, py::arg("fail_value"));
    m.def("model_optimizer_failure_evaluation", [](
            const std::vector<double>& parameters,
            const std::vector<double>& initial_parameters,
            double fail_value,
            bool directional_gradient) {
        return optimizer_failure_to_dict(
            scar::optimizer_failure_evaluation(
                parameters, initial_parameters, fail_value,
                directional_gradient));
    }, py::arg("parameters"), py::arg("initial_parameters"),
       py::arg("fail_value"), py::arg("directional_gradient"));
    m.def("model_optimizer_failure_evaluation_for_size", [](
            std::size_t gradient_size, double fail_value) {
        return optimizer_failure_to_dict(
            scar::optimizer_failure_evaluation_for_size(
                gradient_size, fail_value));
    }, py::arg("gradient_size"), py::arg("fail_value"));
    auto copula_family = py::enum_<scar::CopulaFamily>(
        m, "CopulaFamily", "Native copula-family dispatch identifier.");
#define SCAR_PAIR_FAMILY(                                                \
    enum_name, package_name, enum_value, transform_policy, rotation_policy, \
    default_transform, default_offset)                                  \
    copula_family.value(#enum_name, scar::CopulaFamily::enum_name);
#include "scar/copula/pair/families.def"
#undef SCAR_PAIR_FAMILY
    copula_family
        .value("Student", scar::CopulaFamily::Student)
        .value("EquicorrGaussian", scar::CopulaFamily::EquicorrGaussian)
        .value(
            "MultivariateGaussian",
            scar::CopulaFamily::MultivariateGaussian);

    py::enum_<scar::CorrelationKind>(
        m, "CorrelationKind", "Native correlation representation.")
        .value("DenseCholesky", scar::CorrelationKind::DenseCholesky)
        .value("Factor", scar::CorrelationKind::Factor)
        .value("Fixed", scar::CorrelationKind::Fixed)
        .value("Shrinkage", scar::CorrelationKind::Shrinkage)
        .value("Equicorrelation", scar::CorrelationKind::Equicorrelation)
        .value(
            "FactorJointDynamicEstimation",
            scar::CorrelationKind::FactorJointDynamicEstimation)
        .value("NotApplicable", scar::CorrelationKind::NotApplicable)
        .value("MixedPairEdges", scar::CorrelationKind::MixedPairEdges);
    py::enum_<scar::Rotation>(
        m, "Rotation", "Bivariate copula rotation in degrees.")
        .value("R0", scar::Rotation::R0)
        .value("R90", scar::Rotation::R90)
        .value("R180", scar::Rotation::R180)
        .value("R270", scar::Rotation::R270);
    py::enum_<scar::Transform>(
        m, "Transform", "Latent-state to copula-parameter transform.")
        .value("Softplus", scar::Transform::Softplus)
        .value("XTanh", scar::Transform::XTanh)
        .value("GaussianTanh", scar::Transform::GaussianTanh)
        .value("Exponential", scar::Transform::Exponential)
        .value("Logistic", scar::Transform::Logistic);

    py::class_<scar::CopulaSpec>(
        m,
        "CopulaSpec",
        "Native copula family, transform, dimension, and cached factors.")
        .def(py::init<>())
        .def_property(
            "family",
            [](const scar::CopulaSpec& spec) { return spec.family; },
            [](scar::CopulaSpec& spec, scar::CopulaFamily family) {
                if (spec.family != family) {
                    spec.family = family;
                    spec.reset_model_storage();
                }
            })
        .def_readwrite("rotation", &scar::CopulaSpec::rotation)
        .def_readwrite("transform", &scar::CopulaSpec::transform)
        .def_readwrite("offset", &scar::CopulaSpec::offset)
        .def_readwrite("dim", &scar::CopulaSpec::dim)
        .def_property(
            "correlation_kind",
            [](const scar::CopulaSpec& spec) {
                return spec.correlation_kind;
            },
            [](scar::CopulaSpec& spec, scar::CorrelationKind kind) {
                if (spec.correlation_kind != kind) {
                    spec.correlation_kind = kind;
                    spec.reset_model_storage();
                }
            })
        .def_property(
            "l_inv",
            [](const scar::CopulaSpec& spec) {
                if (spec.family != scar::CopulaFamily::Student
                    && spec.family
                        != scar::CopulaFamily::MultivariateGaussian) {
                    return std::vector<double>{};
                }
                return spec.dense_inverse_cholesky();
            },
            [](scar::CopulaSpec& spec, std::vector<double> value) {
                spec.dense_inverse_cholesky() = std::move(value);
            })
        .def_property(
            "factor_correlation",
            [](const scar::CopulaSpec& spec) {
                if ((spec.family != scar::CopulaFamily::Student
                     && spec.family
                         != scar::CopulaFamily::MultivariateGaussian)
                    || spec.correlation_kind
                        != scar::CorrelationKind::Factor) {
                    return std::shared_ptr<
                        const scar::FactorCorrelationOperator>{};
                }
                return spec.factor_operator();
            },
            [](scar::CopulaSpec& spec,
               std::shared_ptr<const scar::FactorCorrelationOperator> value) {
                spec.factor_operator() = std::move(value);
            })
        .def_property(
            "factor_dimension_tile",
            [](const scar::CopulaSpec& spec) {
                if ((spec.family != scar::CopulaFamily::Student
                     && spec.family
                         != scar::CopulaFamily::MultivariateGaussian)
                    || spec.correlation_kind
                        != scar::CorrelationKind::Factor) {
                    return std::size_t{16384};
                }
                return spec.factor_dimension_tile();
            },
            [](scar::CopulaSpec& spec, std::size_t value) {
                spec.factor_dimension_tile() = value;
            })
        .def_property(
            "log_det",
            [](const scar::CopulaSpec& spec) {
                if (spec.family != scar::CopulaFamily::Student
                    && spec.family
                        != scar::CopulaFamily::MultivariateGaussian) {
                    return 0.0;
                }
                return spec.dense_log_determinant();
            },
            [](scar::CopulaSpec& spec, double value) {
                spec.dense_log_determinant() = value;
            })
        .def_property(
            "ppf_n_obs",
            [](const scar::CopulaSpec& spec) {
                if (spec.family != scar::CopulaFamily::Student) {
                    return std::int64_t{0};
                }
                return spec.student_ppf_observation_count();
            },
            [](scar::CopulaSpec& spec, std::int64_t value) {
                spec.student_ppf_observation_count() = value;
            })
        .def_property(
            "ppf_nodes",
            [](const scar::CopulaSpec& spec) {
                if (spec.family != scar::CopulaFamily::Student) {
                    return std::vector<double>{};
                }
                return spec.student_ppf_nodes();
            },
            [](scar::CopulaSpec& spec, std::vector<double> value) {
                spec.student_ppf_nodes() = std::move(value);
            })
        .def_property(
            "ppf_table",
            [](const scar::CopulaSpec& spec) {
                if (spec.family != scar::CopulaFamily::Student) {
                    return std::vector<double>{};
                }
                return spec.student_ppf_table();
            },
            [](scar::CopulaSpec& spec, std::vector<double> value) {
                spec.student_ppf_table() = std::move(value);
            })
        .def(
            "set_student_ppf_cache",
            &set_student_ppf_cache,
            py::arg("nodes"),
            py::arg("table"));

    m.def(
        "_default_pair_copula_spec",
        &scar::default_pair_copula_spec,
        py::arg("family"));

    py::class_<scar::StaticCopulaEvaluator>(
        m,
        "StaticCopulaEvaluator",
        "Reusable static-likelihood evaluator for fixed observations.")
        .def(
            py::init([](
                const scar::CopulaSpec& copula,
                py::object u,
                int n_threads) {
                return scar::StaticCopulaEvaluator(
                    copula, observations_from_array(
                        real_float64_array_from_object(u, "u")), n_threads);
            }),
            py::arg("copula"),
            py::arg("u"),
            py::arg("n_threads") = 1)
        .def(
            py::init([](
                const scar::CopulaSpec& copula,
                py::object sum_z,
                py::object sum_z2,
                int n_threads) {
                return scar::StaticCopulaEvaluator(
                    copula,
                    vector_from_array(real_float64_array_from_object(sum_z, "sum_z")),
                    vector_from_array(real_float64_array_from_object(sum_z2, "sum_z2")),
                    n_threads);
            }),
            py::arg("copula"),
            py::arg("sum_z"),
            py::arg("sum_z2"),
            py::arg("n_threads") = 1)
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
            "objective_value",
            [](const scar::StaticCopulaEvaluator& evaluator,
               double parameter) {
                scar::StaticObjectiveResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.objective_value(parameter);
                }
                return static_objective_result_to_dict(result);
            },
            py::arg("parameter"))
        .def(
            "transformed_objective",
            [](const scar::StaticCopulaEvaluator& evaluator,
               double optimizer_parameter) {
                scar::StaticObjectiveResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.transformed_objective(
                        optimizer_parameter);
                }
                return static_objective_result_to_dict(result);
            },
            py::arg("optimizer_parameter"))
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
            "gaussian_objective_with_correlation_gradient",
            [](const scar::StaticCopulaEvaluator& evaluator,
               const scar::CopulaSpec& copula) {
                scar::StaticObjectiveResult result;
                {
                    py::gil_scoped_release release;
                    result = evaluator.gaussian_objective(copula, true);
                }
                return static_objective_result_to_dict(result);
            },
            py::arg("copula"))
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
            [](const scar::StaticCopulaEvaluator& evaluator) {
                return static_cast<int>(evaluator.status());
            });

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
        "copula_tau_to_param_capped",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> tau,
           const std::optional<double>& theta_cap) {
            return vector_to_array(
                scar::copula_tau_to_param_capped(
                    copula,
                    vector_from_array(tau),
                    theta_cap.value_or(0.0),
                    theta_cap.has_value()));
        },
        py::arg("copula"),
        py::arg("tau"),
        py::arg("theta_cap"));

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
        "copula_sample_from_uniforms",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               uniforms,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            const scar::Observations uniform_values =
                observations_from_array(uniforms);
            const std::vector<double> parameters = vector_from_array(r);
            scar::Observations result;
            {
                py::gil_scoped_release release;
                result = scar::copula_sample_from_uniforms(
                    copula, uniform_values, parameters);
            }
            return pair_observations_to_array(result);
        },
        py::arg("copula"),
        py::arg("uniforms"),
        py::arg("r"));

    m.def(
        "copula_sample_from_rng_draws",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               draws,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               auxiliary,
           py::array_t<double, py::array::c_style | py::array::forcecast> r) {
            const scar::Observations draw_values =
                finite_rows_from_array(draws, "draws");
            const scar::Observations auxiliary_values =
                finite_rows_from_array(auxiliary, "auxiliary");
            const std::vector<double> parameters = vector_from_array(r);
            scar::Observations result;
            {
                py::gil_scoped_release release;
                result = scar::copula_sample_from_rng_draws(
                    copula, draw_values, auxiliary_values, parameters);
            }
            return pair_observations_to_array(result);
        },
        py::arg("copula"),
        py::arg("draws"),
        py::arg("auxiliary"),
        py::arg("r"));

    m.def(
        "copula_conditional_sample_from_uniforms",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               uniforms,
           py::array_t<double, py::array::c_style | py::array::forcecast> r,
           int given_coordinate,
           double given_value,
           std::optional<double> bisection_tol,
           std::optional<int> bisection_maxiter) {
            scar::HInverseOptions options;
            if (bisection_tol) {
                options.tolerance = *bisection_tol;
            }
            if (bisection_maxiter) {
                options.max_iterations = *bisection_maxiter;
            }
            const std::vector<double> uniform_values =
                vector_from_array(uniforms);
            const std::vector<double> parameters = vector_from_array(r);
            scar::Observations result;
            {
                py::gil_scoped_release release;
                result = scar::copula_conditional_sample_from_uniforms(
                    copula,
                    uniform_values,
                    parameters,
                    given_coordinate,
                    given_value,
                    bisection_tol || bisection_maxiter ? &options : nullptr);
            }
            return pair_observations_to_array(result);
        },
        py::arg("copula"),
        py::arg("uniforms"),
        py::arg("r"),
        py::arg("given_coordinate"),
        py::arg("given_value"),
        py::kw_only(),
        py::arg("bisection_tol") = py::none(),
        py::arg("bisection_maxiter") = py::none());

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

}

}  // namespace pyscarcopula::bindings
