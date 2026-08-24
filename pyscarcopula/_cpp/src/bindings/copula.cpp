#include "common.hpp"

#include "scar/copula/prepared_pair_kernel.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_copula(py::module_& m) {
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
                py::array_t<
                    double,
                    py::array::c_style | py::array::forcecast> u,
                int n_threads) {
                return scar::StaticCopulaEvaluator(
                    copula, observations_from_array(u), n_threads);
            }),
            py::arg("copula"),
            py::arg("u"),
            py::arg("n_threads") = 1)
        .def(
            py::init([](
                const scar::CopulaSpec& copula,
                py::array_t<
                    double,
                    py::array::c_style | py::array::forcecast> sum_z,
                py::array_t<
                    double,
                    py::array::c_style | py::array::forcecast> sum_z2,
                int n_threads) {
                return scar::StaticCopulaEvaluator(
                    copula,
                    vector_from_array(sum_z),
                    vector_from_array(sum_z2),
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
               latent_paths,
           int n_threads) {
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
                    static_cast<std::size_t>(paths_info.shape[1]),
                    n_threads);
            }
            return trajectory_log_pdf_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("latent_paths"),
        py::arg("n_threads") = 1);
}

}  // namespace pyscarcopula::bindings
