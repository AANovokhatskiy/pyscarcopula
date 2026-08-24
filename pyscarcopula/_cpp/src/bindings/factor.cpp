#include "array.hpp"
#include "module.hpp"

#include "scar/factor.hpp"
#include "scar/core/checked_arithmetic.hpp"

#include <pybind11/stl.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

using ConstArray = py::array_t<
    double,
    py::array::c_style | py::array::forcecast>;
using MutableArray = py::array_t<double, py::array::c_style>;

std::size_t checked_rows(
    const py::buffer_info& info,
    std::size_t expected_columns,
    const char* name) {

    if (info.ndim != 2
        || info.shape[1] < 0
        || static_cast<std::size_t>(info.shape[1]) != expected_columns) {
        throw std::invalid_argument(
            std::string(name) + " must be a 2D array with expected width");
    }
    if (info.shape[0] < 0) {
        throw std::invalid_argument(
            std::string(name) + " shape is not representable");
    }
    const std::size_t rows = static_cast<std::size_t>(info.shape[0]);
    std::size_t values = 0;
    if (!scar::core::checked_size_mul(rows, expected_columns, values)
        || rows
            > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            std::string(name) + " shape is not representable");
    }
    return rows;
}

py::dict factor_student_rows_result_to_dict(
    const scar::FactorStudentRowsResult& result) {

    py::dict output;
    output["log_pdf"] = vector_to_array(result.log_pdf);
    output["dlog_ddf"] = vector_to_array(result.dlog_ddf);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["row_parallel_blocks"] = result.row_parallel_blocks;
    output["worker_workspace_peak_bytes"] =
        result.worker_workspace_peak_bytes;
    return output;
}

py::dict factor_student_joint_result_to_dict(
    const scar::FactorStudentJointResult& result,
    std::size_t dimension,
    std::size_t rank) {

    py::dict output;
    output["log_likelihood"] = result.log_likelihood;
    output["dlog_likelihood_ddf"] = result.dlog_likelihood_ddf;
    output["dlog_likelihood_dloadings"] = matrix_to_array(
        result.dlog_likelihood_dloadings, dimension, rank);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["reduction_blocks"] = result.reduction_blocks;
    output["parallel_blocks"] = result.parallel_blocks;
    output["worker_workspace_peak_bytes"] =
        result.worker_workspace_peak_bytes;
    output["reduction_workspace_bytes"] =
        result.reduction_workspace_bytes;
    return output;
}

py::dict factor_student_grid_result_to_dict(
    const scar::FactorStudentGridResult& result) {

    py::dict output;
    output["log_pdf"] = matrix_to_array(
        result.log_pdf, result.rows, result.grid_size);
    output["dlog_ddf"] = matrix_to_array(
        result.dlog_ddf, result.rows, result.grid_size);
    output["rows"] = result.rows;
    output["grid_size"] = result.grid_size;
    output["dimension_tiles"] = result.dimension_tiles;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_axis"] = result.parallel_axis;
    output["parallel_blocks"] = result.parallel_blocks;
    output["worker_workspace_peak_bytes"] =
        result.worker_workspace_peak_bytes;
    output["partial_workspace_peak_bytes"] =
        result.partial_workspace_peak_bytes;
    output["ppf_exact_values"] = result.ppf_exact_values;
    return output;
}

}  // namespace

void bind_factor(py::module_& m) {
    py::class_<
        scar::FactorCorrelationOperator,
        std::shared_ptr<scar::FactorCorrelationOperator>>(
        m, "_FactorCorrelationOperator")
        .def(
            py::init([](
                ConstArray loadings,
                double uniqueness_min) {
                const py::buffer_info info = loadings.request();
                if (info.ndim != 2
                    || info.shape[0] < 2
                    || info.shape[1] < 1
                    || info.shape[1] >= info.shape[0]) {
                    throw std::invalid_argument(
                        "loadings must have shape (d, k), 1 <= k < d");
                }
                const auto dimension =
                    static_cast<std::size_t>(info.shape[0]);
                const auto rank =
                    static_cast<std::size_t>(info.shape[1]);
                return std::make_shared<scar::FactorCorrelationOperator>(
                    flat_vector_from_array(loadings, "loadings"),
                    dimension,
                    rank,
                    uniqueness_min);
            }),
            py::arg("loadings"),
            py::arg("uniqueness_min") = 1e-8)
        .def_property_readonly(
            "dimension",
            &scar::FactorCorrelationOperator::dimension)
        .def_property_readonly(
            "rank",
            &scar::FactorCorrelationOperator::rank)
        .def_property_readonly(
            "uniqueness_min",
            &scar::FactorCorrelationOperator::uniqueness_min)
        .def_property_readonly(
            "logdet",
            &scar::FactorCorrelationOperator::logdet)
        .def_property_readonly(
            "condition_estimate",
            &scar::FactorCorrelationOperator::condition_estimate)
        .def_property_readonly(
            "loadings",
            [](const scar::FactorCorrelationOperator& self) {
                return matrix_to_array(
                    self.loadings(), self.dimension(), self.rank());
            })
        .def_property_readonly(
            "uniqueness",
            [](const scar::FactorCorrelationOperator& self) {
                return vector_to_array(self.uniqueness());
            })
        .def_property_readonly(
            "inverse_uniqueness",
            [](const scar::FactorCorrelationOperator& self) {
                return vector_to_array(self.inverse_uniqueness());
            })
        .def_property_readonly(
            "weighted_loadings",
            [](const scar::FactorCorrelationOperator& self) {
                return matrix_to_array(
                    self.weighted_loadings(),
                    self.dimension(),
                    self.rank());
            })
        .def_property_readonly(
            "cholesky_m",
            [](const scar::FactorCorrelationOperator& self) {
                return matrix_to_array(
                    self.cholesky_m(), self.rank(), self.rank());
            })
        .def(
            "matvec",
            [](const scar::FactorCorrelationOperator& self,
               ConstArray values,
               int n_threads) {
                const py::buffer_info info = values.request();
                const std::size_t rows = checked_rows(
                    info, self.dimension(), "values");
                py::array_t<double> output({
                    static_cast<py::ssize_t>(rows),
                    static_cast<py::ssize_t>(self.dimension()),
                });
                double* const output_data = output.mutable_data();
                {
                    py::gil_scoped_release release;
                    self.matvec_rows(
                        static_cast<const double*>(info.ptr),
                        rows,
                        output_data,
                        n_threads);
                }
                return output;
            },
            py::arg("values"),
            py::arg("n_threads") = 1)
        .def(
            "solve",
            [](const scar::FactorCorrelationOperator& self,
               ConstArray values,
               int n_threads) {
                const py::buffer_info info = values.request();
                const std::size_t rows = checked_rows(
                    info, self.dimension(), "values");
                py::array_t<double> output({
                    static_cast<py::ssize_t>(rows),
                    static_cast<py::ssize_t>(self.dimension()),
                });
                double* const output_data = output.mutable_data();
                {
                    py::gil_scoped_release release;
                    self.solve_rows(
                        static_cast<const double*>(info.ptr),
                        rows,
                        output_data,
                        n_threads);
                }
                return output;
            },
            py::arg("values"),
            py::arg("n_threads") = 1)
        .def(
            "quadratic_forms",
            [](const scar::FactorCorrelationOperator& self,
               ConstArray values,
               int n_threads) {
                const py::buffer_info info = values.request();
                const std::size_t rows = checked_rows(
                    info, self.dimension(), "values");
                py::array_t<double> output(
                    static_cast<py::ssize_t>(rows));
                double* const output_data = output.mutable_data();
                {
                    py::gil_scoped_release release;
                    self.quadratic_forms(
                        static_cast<const double*>(info.ptr),
                        rows,
                        output_data,
                        n_threads);
                }
                return output;
            },
            py::arg("values"),
            py::arg("n_threads") = 1)
        .def(
            "sample_normal_inplace",
            [](const scar::FactorCorrelationOperator& self,
               ConstArray factor_draws,
               MutableArray residual_draws,
               int n_threads) {
                const py::buffer_info factor_info = factor_draws.request();
                const py::buffer_info residual_info =
                    residual_draws.request();
                const std::size_t rows = checked_rows(
                    factor_info, self.rank(), "factor_draws");
                const std::size_t residual_rows = checked_rows(
                    residual_info,
                    self.dimension(),
                    "residual_draws");
                if (residual_rows != rows || !residual_draws.writeable()) {
                    throw std::invalid_argument(
                        "residual_draws must be writable and share row count");
                }
                {
                    py::gil_scoped_release release;
                    self.sample_normal_inplace(
                        static_cast<const double*>(factor_info.ptr),
                        static_cast<double*>(residual_info.ptr),
                        rows,
                        n_threads);
                }
            },
            py::arg("factor_draws"),
            py::arg("residual_draws"),
            py::arg("n_threads") = 1);

    m.def(
        "_factor_student_log_pdf_and_dlog_ddf",
        [](
            const scar::FactorCorrelationOperator& correlation,
            ConstArray observations,
            ConstArray df,
            int n_threads) {
            const py::buffer_info observation_info =
                observations.request();
            const std::size_t rows = checked_rows(
                observation_info,
                correlation.dimension(),
                "observations");
            const py::buffer_info df_info = df.request();
            if (df_info.ndim != 1
                || (df_info.shape[0] != 1
                    && df_info.shape[0]
                        != static_cast<py::ssize_t>(rows))) {
                throw std::invalid_argument(
                    "df must be a 1D array with one value or one per row");
            }
            scar::FactorStudentRowsResult result;
            {
                py::gil_scoped_release release;
                result = scar::factor_student_log_pdf_and_dlog_ddf(
                    correlation,
                    static_cast<const double*>(observation_info.ptr),
                    rows,
                    static_cast<const double*>(df_info.ptr),
                    static_cast<std::size_t>(df_info.shape[0]),
                    n_threads);
            }
            return factor_student_rows_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("observations"),
        py::arg("df"),
            py::arg("n_threads") = 1);

    m.def(
        "_factor_student_joint_likelihood_gradient",
        [](
            const scar::FactorCorrelationOperator& correlation,
            ConstArray observations,
            double df,
            int n_threads) {
            const py::buffer_info observation_info =
                observations.request();
            const std::size_t rows = checked_rows(
                observation_info,
                correlation.dimension(),
                "observations");
            scar::FactorStudentJointResult result;
            {
                py::gil_scoped_release release;
                result =
                    scar::factor_student_joint_likelihood_gradient(
                        correlation,
                        static_cast<const double*>(observation_info.ptr),
                        rows,
                        df,
                        n_threads);
            }
            return factor_student_joint_result_to_dict(
                result,
                correlation.dimension(),
                correlation.rank());
        },
        py::arg("correlation"),
        py::arg("observations"),
        py::arg("df"),
        py::arg("n_threads") = 1);

    m.def(
        "_factor_student_log_pdf_and_dlog_ddf_grid",
        [](
            const scar::FactorCorrelationOperator& correlation,
            ConstArray observations,
            ConstArray df_grid,
            std::size_t dimension_tile,
            int n_threads) {
            const py::buffer_info observation_info =
                observations.request();
            const std::size_t rows = checked_rows(
                observation_info,
                correlation.dimension(),
                "observations");
            const py::buffer_info grid_info = df_grid.request();
            if (grid_info.ndim != 1 || grid_info.shape[0] < 1) {
                throw std::invalid_argument(
                    "df_grid must be a non-empty 1D array");
            }
            const std::size_t grid_size =
                static_cast<std::size_t>(grid_info.shape[0]);
            scar::FactorStudentGridResult result;
            {
                py::gil_scoped_release release;
                result =
                    scar::factor_student_log_pdf_and_dlog_ddf_grid(
                        correlation,
                        static_cast<const double*>(observation_info.ptr),
                        rows,
                        static_cast<const double*>(grid_info.ptr),
                        grid_size,
                        dimension_tile,
                        n_threads);
            }
            return factor_student_grid_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("observations"),
        py::arg("df_grid"),
        py::arg("dimension_tile") = 16384,
        py::arg("n_threads") = 1);
}

}  // namespace pyscarcopula::bindings
