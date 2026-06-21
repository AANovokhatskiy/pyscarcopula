#include "common.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_copula(py::module_& m) {
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
}

}  // namespace pyscarcopula::bindings
