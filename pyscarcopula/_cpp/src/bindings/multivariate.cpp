#include "array.hpp"
#include "module.hpp"

#include "scar/copula.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/gaussian/conditional.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/sampling.hpp"
#include "scar/copula/multivariate/student/conditional.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"
#include "scar/core/checked_arithmetic.hpp"

#include <pybind11/stl.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

py::array_t<double> grid_values_to_array(const scar::GridValues& values) {
    return matrix_to_array(
        values.values,
        static_cast<std::size_t>(values.n_obs),
        static_cast<std::size_t>(values.n_grid));
}

py::dict multivariate_rows_result_to_dict(
    const scar::MultivariateRowsResult& result) {

    py::dict output;
    output["log_pdf"] = vector_to_array(result.log_pdf);
    output["dlog_dr"] = vector_to_array(result.dlog_dr);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["student_ppf_cache_values"] = result.student_ppf_cache_values;
    output["student_ppf_exact_values"] = result.student_ppf_exact_values;
    output["student_ppf_asymptotic_values"] =
        result.student_ppf_asymptotic_values;
    output["student_workspace_growth_events"] =
        result.student_workspace_growth_events;
    output["student_workspace_peak_bytes"] =
        result.student_workspace_peak_bytes;
    output["n_threads_requested"] = result.n_threads_requested;
    output["row_parallel_blocks"] = result.row_parallel_blocks;
    return output;
}

py::dict multivariate_grid_result_to_dict(
    const scar::MultivariateGridResult& result) {

    py::dict output;
    output["pdf"] = grid_values_to_array(result.pdf);
    output["d_pdf_dx"] = grid_values_to_array(result.d_pdf_dx);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["student_ppf_cache_values"] = result.student_ppf_cache_values;
    output["student_ppf_exact_values"] = result.student_ppf_exact_values;
    output["student_ppf_asymptotic_values"] =
        result.student_ppf_asymptotic_values;
    output["student_workspace_growth_events"] =
        result.student_workspace_growth_events;
    output["student_workspace_peak_bytes"] =
        result.student_workspace_peak_bytes;
    output["n_threads_requested"] = result.n_threads_requested;
    output["student_parallel_blocks"] = result.student_parallel_blocks;
    output["equicorr_parallel_blocks"] = result.equicorr_parallel_blocks;
    return output;
}

py::dict equicorr_preparation_result_to_dict(
    const scar::EquicorrPreparationResult& result) {

    py::dict output;
    output["sum_z"] = vector_to_array(result.sum_z);
    output["sum_z2"] = vector_to_array(result.sum_z2);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_blocks"] = result.parallel_blocks;
    output["parallel_axis"] = result.parallel_axis;
    output["dimension_tiles"] = result.dimension_tiles;
    output["temporary_values"] = result.temporary_values;
    output["clipping_events"] = result.clipping_events;
    output["nonfinite_values"] = result.nonfinite_values;
    return output;
}

py::dict conditional_sample_result_to_dict(
    const scar::ConditionalSampleResult& result) {

    py::dict output;
    output["values"] = result.is_ok()
        ? py::object(matrix_to_array(
            result.values,
            static_cast<std::size_t>(result.n_rows),
            static_cast<std::size_t>(result.n_free)))
        : py::object(py::array_t<double>(0));
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_blocks"] = result.parallel_blocks;
    output["correlation_factorizations"] =
        result.correlation_factorizations;
    return output;
}

py::dict multivariate_rosenblatt_result_to_dict(
    const scar::MultivariateRosenblattResult& result) {

    py::dict diagnostics;
    diagnostics["n_threads_requested"] = result.n_threads_requested;
    diagnostics["parallel_blocks"] = result.parallel_blocks;
    diagnostics["correlation_factorizations"] =
        result.correlation_factorizations;

    py::dict output;
    if (result.n_rows >= 0
        && result.dimension >= 0
        && result.residuals.size()
            == static_cast<std::size_t>(result.n_rows)
                * static_cast<std::size_t>(result.dimension)) {
        output["residuals"] = matrix_to_array(
            result.residuals,
            static_cast<std::size_t>(result.n_rows),
            static_cast<std::size_t>(result.dimension));
    } else {
        // A failed native operation intentionally owns no output matrix.  Its
        // structured status must still cross the binding boundary so Python
        // can translate it instead of seeing an unrelated shape exception.
        output["residuals"] = vector_to_array(result.residuals);
    }
    output["n_rows"] = result.n_rows;
    output["dimension"] = result.dimension;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_coordinate"] = result.failure.coordinate;
    output["diagnostics"] = std::move(diagnostics);
    return output;
}

py::dict radial_summary_result_to_dict(
    const scar::RadialSummaryResult& result) {

    py::dict output;
    output["values"] = vector_to_array(result.values);
    output["n_rows"] = result.n_rows;
    output["dimension"] = result.dimension;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_coordinate"] = result.failure.coordinate;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_blocks"] = result.parallel_blocks;
    return output;
}

py::dict gaussian_score_correlation_result_to_dict(
    const scar::GaussianScoreCorrelationResult& result) {

    py::dict output;
    if (result.dimension >= 0
        && result.correlation.size()
            == static_cast<std::size_t>(result.dimension)
                * static_cast<std::size_t>(result.dimension)) {
        output["correlation"] = matrix_to_array(
            result.correlation,
            static_cast<std::size_t>(result.dimension),
            static_cast<std::size_t>(result.dimension));
    } else {
        output["correlation"] = vector_to_array(result.correlation);
    }
    output["dimension"] = result.dimension;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_coordinate"] = result.failure.coordinate;
    return output;
}

py::dict vector_result_to_dict(
    const scar::Result<std::vector<double>>& result) {

    py::dict output;
    output["values"] = vector_to_array(result.value);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_coordinate"] = result.failure.coordinate;
    return output;
}

py::dict preprocessing_result_to_dict(
    const scar::CorrelationPreprocessingResult& result) {

    py::dict output;
    if (result.dimension >= 2
        && result.correlation.size()
            == result.dimension * result.dimension) {
        output["correlation"] = matrix_to_array(
            result.correlation, result.dimension, result.dimension);
        output["input_correlation"] = matrix_to_array(
            result.input_correlation, result.dimension, result.dimension);
    } else {
        output["correlation"] = vector_to_array(result.correlation);
        output["input_correlation"] = vector_to_array(
            result.input_correlation);
    }
    output["min_eigenvalue_before"] = result.min_eigenvalue_before;
    output["min_eigenvalue_after"] = result.min_eigenvalue_after;
    output["projection_applied"] = result.projection_applied;
    py::list nonfinite_pairs;
    for (std::size_t index = 0;
         index + 1 < result.nonfinite_kendall_pairs.size();
         index += 2) {
        nonfinite_pairs.append(py::make_tuple(
            result.nonfinite_kendall_pairs[index],
            result.nonfinite_kendall_pairs[index + 1]));
    }
    output["nonfinite_kendall_pairs"] = std::move(nonfinite_pairs);
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_coordinate"] = result.failure.coordinate;
    return output;
}

scar::DoubleView flat_view(const py::buffer_info& info) {
    return {
        static_cast<const double*>(info.ptr),
        static_cast<std::size_t>(info.size),
    };
}

}  // namespace

void bind_multivariate(py::module_& m) {
    using ConstArray = py::array_t<
        double,
        py::array::c_style | py::array::forcecast>;

    using PpfTableConfig = scar::copula::multivariate::student::PpfTableConfig;
    py::class_<PpfTableConfig>(m, "StudentPpfTableConfig")
        .def(py::init<>())
        .def_readwrite("df_lo", &PpfTableConfig::df_lo)
        .def_readwrite("df_hi", &PpfTableConfig::df_hi)
        .def_readwrite("n_boundary", &PpfTableConfig::n_boundary)
        .def_readwrite("n_lo", &PpfTableConfig::n_lo)
        .def_readwrite("n_hi", &PpfTableConfig::n_hi)
        .def_readwrite("max_table_bytes", &PpfTableConfig::max_table_bytes);
    m.def("student_prepare_ppf_table",
        [](const Float64Array& observations, const PpfTableConfig& config) {
            const auto view = flat_view_from_array(observations, "observations");
            scar::Result<scar::copula::multivariate::student::PreparedPpfTable> result;
            {
                py::gil_scoped_release release;
                result = scar::copula::multivariate::student::prepare_ppf_table(view, config);
            }
            py::dict output;
            output["status"] = static_cast<int>(result.status);
            output["failure_index"] = result.failure.index;
            output["observations"] = vector_to_array(std::move(result.value.observations));
            output["nodes"] = vector_to_array(std::move(result.value.nodes));
            output["table"] = vector_to_array(std::move(result.value.table));
            output["has_table"] = result.value.has_table;
            return output;
        }, py::arg("observations"), py::arg("config"));
    m.def("student_evaluate_ppf_table",
        [](const Float64Array& observations, const Float64Array& nodes,
           const Float64Array& table, double df, std::size_t offset, std::size_t count) {
            const auto obs_view = flat_view_from_array(observations, "observations");
            const auto node_view = flat_view_from_array(nodes, "nodes");
            const auto table_info = table.request();
            const auto table_view = flat_view(table_info);
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::copula::multivariate::student::evaluate_ppf_table(
                    obs_view, node_view, table_view, df, offset, count);
            }
            return vector_result_to_dict(result);
        }, py::arg("observations"), py::arg("nodes"), py::arg("table"),
        py::arg("df"), py::arg("offset"), py::arg("count"));
    m.def("student_interpolate_ppf_table",
        [](const Float64Array& nodes, const Float64Array& table, double df, std::size_t width) {
            const auto node_view = flat_view_from_array(nodes, "nodes");
            const auto table_info = table.request();
            const auto table_view = flat_view(table_info);
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::copula::multivariate::student::interpolate_ppf_table(
                    node_view, table_view, df, width);
            }
            return vector_result_to_dict(result);
        }, py::arg("nodes"), py::arg("table"), py::arg("df"), py::arg("width"));

    m.def(
        "static_correlation_logistic",
        [](ConstArray values) {
            const py::buffer_info info = values.request();
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::logistic_transform(flat_view(info));
            }
            return vector_result_to_dict(result);
        },
        py::arg("values"));
    m.def(
        "static_correlation_logit",
        [](ConstArray values) {
            const py::buffer_info info = values.request();
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::logit_transform(flat_view(info));
            }
            return vector_result_to_dict(result);
        },
        py::arg("values"));
    m.def(
        "static_preprocess_correlation",
        [](ConstArray correlation, double eigenvalue_floor) {
            const py::buffer_info info = correlation.request();
            if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
                throw std::invalid_argument(
                    "correlation must be a square matrix");
            }
            scar::CorrelationPreprocessingResult result;
            {
                py::gil_scoped_release release;
                result = scar::preprocess_correlation(
                    flat_view(info),
                    static_cast<std::size_t>(info.shape[0]),
                    eigenvalue_floor);
            }
            return preprocessing_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("eigenvalue_floor") = 1e-8);
    m.def(
        "static_prepare_dense_correlation",
        [](ConstArray correlation) {
            const py::buffer_info info = correlation.request();
            if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
                throw std::invalid_argument(
                    "correlation must be a square matrix");
            }
            scar::DenseCorrelationPreparationResult result;
            {
                py::gil_scoped_release release;
                result = scar::prepare_dense_correlation(
                    flat_view(info),
                    static_cast<std::size_t>(info.shape[0]));
            }
            py::dict output;
            output["inverse_cholesky"] = vector_to_array(
                result.inverse_cholesky);
            output["log_determinant"] = result.log_determinant;
            output["dimension"] = result.dimension;
            output["status"] = static_cast<int>(result.status);
            output["failure_index"] = result.failure.index;
            output["failure_coordinate"] = result.failure.coordinate;
            return output;
        },
        py::arg("correlation"));
    m.def(
        "static_validate_correlation",
        [](ConstArray correlation, double tolerance) {
            const py::buffer_info info = correlation.request();
            if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
                throw std::invalid_argument(
                    "correlation must be a square matrix");
            }
            scar::Result<bool> result;
            {
                py::gil_scoped_release release;
                result = scar::validate_correlation(
                    flat_view(info),
                    static_cast<std::size_t>(info.shape[0]),
                    tolerance);
            }
            py::dict output;
            output["valid"] = result.value;
            output["status"] = static_cast<int>(result.status);
            output["failure_index"] = result.failure.index;
            output["failure_coordinate"] = result.failure.coordinate;
            return output;
        },
        py::arg("correlation"),
        py::arg("tolerance") = 1e-8);
    m.def(
        "static_estimate_kendall_correlation",
        [](ConstArray observations, double eigenvalue_floor) {
            const py::buffer_info info = observations.request();
            if (info.ndim != 2) {
                throw std::invalid_argument(
                    "observations must have shape (n,d)");
            }
            scar::CorrelationPreprocessingResult result;
            {
                const scar::ObservationView view = observation_view_from_array(
                    static_cast<int>(info.shape[1]), observations);
                py::gil_scoped_release release;
                result = scar::estimate_kendall_correlation(
                    view, eigenvalue_floor);
            }
            return preprocessing_result_to_dict(result);
        },
        py::arg("observations"),
        py::arg("eigenvalue_floor") = 1e-8);
    m.def(
        "static_make_shrinkage_correlation",
        [](double raw_parameter, ConstArray base) {
            const py::buffer_info info = base.request();
            if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
                throw std::invalid_argument("base must be a square matrix");
            }
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::make_shrinkage_correlation(
                    raw_parameter,
                    flat_view(info),
                    static_cast<std::size_t>(info.shape[0]));
            }
            py::dict output = vector_result_to_dict(result);
            output["dimension"] = static_cast<std::size_t>(
                info.shape[0]);
            return output;
        },
        py::arg("raw_parameter"),
        py::arg("base"));
    m.def(
        "static_pack_cholesky_correlation",
        [](ConstArray correlation, double eigenvalue_floor) {
            const py::buffer_info info = correlation.request();
            if (info.ndim != 2 || info.shape[0] != info.shape[1]) {
                throw std::invalid_argument(
                    "correlation must be a square matrix");
            }
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::pack_cholesky_correlation(
                    flat_view(info),
                    static_cast<std::size_t>(info.shape[0]),
                    eigenvalue_floor);
            }
            return vector_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("eigenvalue_floor") = 1e-8);
    m.def(
        "static_unpack_cholesky_correlation",
        [](ConstArray parameters, std::size_t dimension) {
            const py::buffer_info info = parameters.request();
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::unpack_cholesky_correlation(
                    flat_view(info), dimension);
            }
            py::dict output = vector_result_to_dict(result);
            output["dimension"] = dimension;
            return output;
        },
        py::arg("parameters"),
        py::arg("dimension"));
    m.def(
        "static_correlation_gradient_to_raw",
        [](int mode,
           ConstArray parameters,
           ConstArray correlation,
           ConstArray correlation_gradient,
           ConstArray base) {
            const py::buffer_info parameter_info = parameters.request();
            const py::buffer_info correlation_info = correlation.request();
            const py::buffer_info gradient_info =
                correlation_gradient.request();
            const py::buffer_info base_info = base.request();
            if (correlation_info.ndim != 2
                || correlation_info.shape[0] != correlation_info.shape[1]) {
                throw std::invalid_argument(
                    "correlation must be a square matrix");
            }
            const std::size_t dimension =
                static_cast<std::size_t>(correlation_info.shape[0]);
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::correlation_gradient_to_raw(
                    static_cast<scar::DenseCorrelationMode>(mode),
                    flat_view(parameter_info),
                    flat_view(correlation_info),
                    flat_view(gradient_info),
                    flat_view(base_info),
                    dimension);
            }
            return vector_result_to_dict(result);
        },
        py::arg("mode"),
        py::arg("parameters"),
        py::arg("correlation"),
        py::arg("correlation_gradient"),
        py::arg("base"));
    m.def(
        "static_shrinkage_raw_direction",
        [](ConstArray parameters, ConstArray base) {
            const py::buffer_info parameter_info = parameters.request();
            const py::buffer_info base_info = base.request();
            if (base_info.ndim != 2
                || base_info.shape[0] != base_info.shape[1]) {
                throw std::invalid_argument("base must be a square matrix");
            }
            scar::Result<std::vector<double>> result;
            {
                py::gil_scoped_release release;
                result = scar::shrinkage_raw_correlation_direction(
                    flat_view(parameter_info),
                    flat_view(base_info),
                    static_cast<std::size_t>(base_info.shape[0]));
            }
            return vector_result_to_dict(result);
        },
        py::arg("parameters"),
        py::arg("base"));

    m.def(
        "gaussian_score_correlation",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> u) {
            const py::buffer_info info = u.request();
            if (info.ndim != 2) {
                throw std::invalid_argument("u must have shape (n,d)");
            }
            scar::GaussianScoreCorrelationResult result;
            {
                const auto observations = observation_view_from_array(
                    static_cast<int>(info.shape[1]), u);
                py::gil_scoped_release release;
                result = scar::gaussian_score_correlation(observations);
            }
            return gaussian_score_correlation_result_to_dict(result);
        },
        py::arg("u"));

    m.def(
        "_student_quantile",
        &scar_internal::student_quantile_value,
        py::arg("p"),
        py::arg("df"));
    m.def(
        "_student_quantile_large_df",
        [](double p, double df) {
            double value = 0.0;
            double derivative = 0.0;
            scar_internal::student_quantile_large_df_value_and_derivative(
                p, df, value, derivative);
            return py::make_tuple(value, derivative);
        },
        py::arg("p"),
        py::arg("df"));
    m.def(
        "_student_quantile_with_df_derivative",
        [](double p, double df) {
            double value = 0.0;
            double derivative = 0.0;
            scar_internal::student_quantile_value_and_derivative(
                p, df, value, derivative);
            return py::make_tuple(value, derivative);
        },
        py::arg("p"),
        py::arg("df"));

    m.def(
        "prepare_equicorr_sufficient_statistics",
        [](
            py::object u_input,
            std::size_t dimension_tile,
            int n_threads) {
            const auto u = real_float64_array_from_object(u_input, "u");
            const py::buffer_info info = u.request();
            if (info.ndim != 2 || info.shape[1] < 2) {
                throw std::invalid_argument(
                    "u must be a 2D float64 array with shape (n, d), d >= 2");
            }
            if (info.shape[0] < 0 || info.shape[1] < 0) {
                throw std::invalid_argument("u shape is not representable");
            }
            const std::size_t n_obs =
                static_cast<std::size_t>(info.shape[0]);
            const std::size_t dimension =
                static_cast<std::size_t>(info.shape[1]);
            std::size_t values = 0;
            if (dimension
                    > static_cast<std::size_t>(
                        std::numeric_limits<int>::max())
                || !scar::core::checked_size_mul(
                    n_obs, dimension, values)) {
                throw std::invalid_argument("u shape is not representable");
            }
            scar::EquicorrPreparationResult result;
            {
                const scar::ObservationView observations{
                    static_cast<const double*>(info.ptr),
                    n_obs,
                    static_cast<int>(dimension),
                };
                py::gil_scoped_release release;
                result = scar::prepare_equicorr_sufficient_statistics(
                    observations, dimension_tile, n_threads);
            }
            return equicorr_preparation_result_to_dict(result);
        },
        py::arg("u"),
        py::arg("dimension_tile") = 16384,
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_log_pdf_and_grad",
        [](const scar::CopulaSpec& copula,
           py::object u,
           py::object r,
           std::int64_t row_offset,
           int n_threads) {
            scar::MultivariateRowsResult result;
            const scar::Observations observations =
                observations_from_array(real_float64_array_from_object(u, "u"));
            const std::vector<double> parameters = vector_from_array(
                real_float64_array_from_object(r, "r"));
            {
                py::gil_scoped_release release;
                result = scar::multivariate_log_pdf_and_grad(
                    copula, observations, parameters, row_offset, n_threads);
            }
            return multivariate_rows_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("r"),
        py::arg("row_offset") = 0,
        py::arg("n_threads") = 1);

    m.def(
        "equicorr_log_pdf_and_grad_from_stats",
        [](
            const scar::CopulaSpec& copula,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                sum_z,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                sum_z2,
            py::array_t<double, py::array::c_style | py::array::forcecast> r,
            int n_threads) {
            scar::MultivariateRowsResult result;
            const auto sums = flat_view_from_array(sum_z, "sum_z");
            const auto sums2 = flat_view_from_array(sum_z2, "sum_z2");
            const auto parameters = vector_from_array(r);
            {
                py::gil_scoped_release release;
                result = scar::equicorr_log_pdf_and_grad_from_stats(
                    copula, sums, sums2, parameters, n_threads);
            }
            return multivariate_rows_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("sum_z"),
        py::arg("sum_z2"),
        py::arg("r"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_pdf_and_grad_grid",
        [](const scar::CopulaSpec& copula,
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               x_grid,
           std::int64_t row_offset,
           int n_threads) {
            scar::MultivariateGridResult result;
            const scar::Observations observations =
                observations_from_array(u);
            const std::vector<double> grid = vector_from_array(x_grid);
            {
                py::gil_scoped_release release;
                result = scar::multivariate_pdf_and_grad_grid(
                    copula, observations, grid, row_offset, n_threads);
            }
            return multivariate_grid_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("u"),
        py::arg("x_grid"),
        py::arg("row_offset") = 0,
        py::arg("n_threads") = 1);

    m.def(
        "equicorr_pdf_and_grad_grid_from_stats",
        [](
            const scar::CopulaSpec& copula,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                sum_z,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                sum_z2,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                x_grid,
            int n_threads) {
            scar::MultivariateGridResult result;
            const auto sums = flat_view_from_array(sum_z, "sum_z");
            const auto sums2 = flat_view_from_array(sum_z2, "sum_z2");
            const auto grid = vector_from_array(x_grid);
            {
                py::gil_scoped_release release;
                result = scar::equicorr_pdf_and_grad_grid_from_stats(
                    copula, sums, sums2, grid, n_threads);
            }
            return multivariate_grid_result_to_dict(result);
        },
        py::arg("copula"),
        py::arg("sum_z"),
        py::arg("sum_z2"),
        py::arg("x_grid"),
        py::arg("n_threads") = 1);

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
                normal_draws,
            int n_threads) {

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
                const auto corr = flat_view_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto latent = flat_view_from_array(
                    given_latent, "given_latent");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_conditional(
                    corr, correlation_rows, dimension, indices,
                    latent, draws, n_rows, n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_latent"),
        py::arg("normal_draws"),
        py::arg("n_threads") = 1);

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
                chi_square_draws,
            int n_threads) {

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
                const auto corr = flat_view_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto latent = flat_view_from_array(
                    given_latent, "given_latent");
                const auto degrees = flat_view_from_array(df, "df");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                const auto chi = flat_view_from_array(
                    chi_square_draws, "chi_square_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_student_conditional(
                    corr, correlation_rows, dimension, indices,
                    latent, degrees, draws, chi, n_rows, n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_latent"),
        py::arg("df"),
        py::arg("normal_draws"),
        py::arg("chi_square_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_student_sample_from_normal_uniforms",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast> df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_uniforms,
            int n_threads) {
            const py::buffer_info correlation_info = correlation.request();
            const py::buffer_info draw_info = normal_draws.request();
            if (correlation_info.ndim != 2
                || correlation_info.shape[0] != correlation_info.shape[1]
                || correlation_info.shape[0] < 2
                || draw_info.ndim != 2
                || draw_info.shape[1] != correlation_info.shape[0]) {
                throw std::invalid_argument(
                    "correlation must have shape (d,d) and normal_draws "
                    "must have shape (n,d)");
            }
            scar::ConditionalSampleResult result;
            {
                const auto correlation_view = flat_view_from_array(
                    correlation, "correlation");
                const auto degrees = flat_view_from_array(df, "df");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                const auto uniforms = flat_view_from_array(
                    chi_square_uniforms, "chi_square_uniforms");
                py::gil_scoped_release release;
                result = scar::multivariate_student_sample_dense_from_uniforms(
                    correlation_view,
                    static_cast<int>(correlation_info.shape[0]),
                    degrees,
                    draws,
                    uniforms,
                    static_cast<std::int64_t>(draw_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("df"),
        py::arg("normal_draws"),
        py::arg("chi_square_uniforms"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_gaussian_sample_from_normals",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            int n_threads) {
            const py::buffer_info correlation_info = correlation.request();
            const py::buffer_info draw_info = normal_draws.request();
            if (correlation_info.ndim != 2
                || correlation_info.shape[0] != correlation_info.shape[1]
                || correlation_info.shape[0] < 2
                || draw_info.ndim != 2
                || draw_info.shape[1] != correlation_info.shape[0]) {
                throw std::invalid_argument(
                    "correlation must have shape (d,d) and normal_draws "
                    "must have shape (n,d)");
            }
            scar::ConditionalSampleResult result;
            {
                const auto correlation_view = flat_view_from_array(
                    correlation, "correlation");
                const auto draw_view = flat_view_from_array(
                    normal_draws, "normal_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_sample_dense(
                    correlation_view,
                    static_cast<int>(correlation_info.shape[0]),
                    draw_view,
                    static_cast<std::int64_t>(draw_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("normal_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "equicorr_gaussian_common_draw_count",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                rho,
            int dimension,
            std::int64_t n_rows) {
            scar::Result<std::int64_t> result;
            {
                const auto parameters = flat_view_from_array(rho, "rho");
                py::gil_scoped_release release;
                result = scar::equicorr_gaussian_common_draw_count(
                    parameters, dimension, n_rows);
            }
            py::dict output;
            output["status"] = static_cast<int>(result.status);
            output["failure_index"] = result.failure.index;
            output["count"] = result.value;
            return output;
        },
        py::arg("rho"),
        py::arg("dimension"),
        py::arg("n_rows"));

    m.def(
        "validate_equicorrelation_path",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                rho,
            int dimension,
            std::int64_t n_rows) {
            const auto parameters = flat_view_from_array(rho, "rho");
            return static_cast<int>(scar::validate_equicorrelation_path(
                parameters, dimension, n_rows));
        },
        py::arg("rho"),
        py::arg("dimension"),
        py::arg("n_rows"));

    m.def(
        "equicorr_gaussian_sample_from_normals",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                rho,
            int dimension,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                common_draws,
            int n_threads) {
            const py::buffer_info draw_info = normal_draws.request();
            const py::buffer_info common_info = common_draws.request();
            if (draw_info.ndim != 2
                || draw_info.shape[1] != dimension
                || common_info.ndim != 1) {
                throw py::value_error(
                    "normal_draws must have shape (n, dimension) and "
                    "common_draws must be one-dimensional");
            }
            scar::ConditionalSampleResult result;
            {
                const auto parameters = flat_view_from_array(rho, "rho");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                const auto common = flat_view_from_array(
                    common_draws, "common_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_sample_equicorrelation(
                    parameters,
                    dimension,
                    draws,
                    common,
                    static_cast<std::int64_t>(draw_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("rho"),
        py::arg("dimension"),
        py::arg("normal_draws"),
        py::arg("common_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_student_sample_from_draws",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_draws,
            int n_threads) {
            const py::buffer_info correlation_info = correlation.request();
            const py::buffer_info draw_info = normal_draws.request();
            if (correlation_info.ndim != 2
                || correlation_info.shape[0] != correlation_info.shape[1]
                || correlation_info.shape[0] < 2
                || draw_info.ndim != 2
                || draw_info.shape[1] != correlation_info.shape[0]) {
                throw std::invalid_argument(
                    "correlation must have shape (d,d) and normal_draws "
                    "must have shape (n,d)");
            }
            scar::ConditionalSampleResult result;
            {
                const auto correlation_view = flat_view_from_array(
                    correlation, "correlation");
                const auto degrees = flat_view_from_array(df, "df");
                const auto draw_view = flat_view_from_array(
                    normal_draws, "normal_draws");
                const auto chi = flat_view_from_array(
                    chi_square_draws, "chi_square_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_student_sample_dense(
                    correlation_view,
                    static_cast<int>(correlation_info.shape[0]),
                    degrees,
                    draw_view,
                    chi,
                    static_cast<std::int64_t>(draw_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("df"),
        py::arg("normal_draws"),
        py::arg("chi_square_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_student_sample_from_normal_uniforms",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast> df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                factor_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residual_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_uniforms,
            int n_threads) {
            const py::buffer_info factor_info = factor_draws.request();
            const py::buffer_info residual_info = residual_draws.request();
            if (factor_info.ndim != 2
                || residual_info.ndim != 2
                || factor_info.shape[0] != residual_info.shape[0]
                || factor_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.rank())
                || residual_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "factor_draws and residual_draws have invalid shapes");
            }
            scar::ConditionalSampleResult result;
            {
                const auto degrees = flat_view_from_array(df, "df");
                const auto factors = flat_view_from_array(
                    factor_draws, "factor_draws");
                const auto residuals = flat_view_from_array(
                    residual_draws, "residual_draws");
                const auto uniforms = flat_view_from_array(
                    chi_square_uniforms, "chi_square_uniforms");
                py::gil_scoped_release release;
                result = scar::multivariate_student_sample_factor_from_uniforms(
                    correlation,
                    degrees,
                    factors,
                    residuals,
                    uniforms,
                    static_cast<std::int64_t>(factor_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("df"),
        py::arg("factor_draws"),
        py::arg("residual_draws"),
        py::arg("chi_square_uniforms"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_gaussian_sample_from_normals",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                factor_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residual_draws,
            int n_threads) {
            const py::buffer_info factor_info = factor_draws.request();
            const py::buffer_info residual_info = residual_draws.request();
            if (factor_info.ndim != 2
                || residual_info.ndim != 2
                || factor_info.shape[0] != residual_info.shape[0]
                || factor_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.rank())
                || residual_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "factor_draws and residual_draws have invalid shapes");
            }
            scar::ConditionalSampleResult result;
            {
                const auto factors = flat_view_from_array(
                    factor_draws, "factor_draws");
                const auto residuals = flat_view_from_array(
                    residual_draws, "residual_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_sample_factor(
                    correlation,
                    factors,
                    residuals,
                    static_cast<std::int64_t>(factor_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("factor_draws"),
        py::arg("residual_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_student_sample_from_draws",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                factor_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residual_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_draws,
            int n_threads) {
            const py::buffer_info factor_info = factor_draws.request();
            const py::buffer_info residual_info = residual_draws.request();
            if (factor_info.ndim != 2
                || residual_info.ndim != 2
                || factor_info.shape[0] != residual_info.shape[0]
                || factor_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.rank())
                || residual_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "factor_draws and residual_draws have invalid shapes");
            }
            scar::ConditionalSampleResult result;
            {
                const auto degrees = flat_view_from_array(df, "df");
                const auto factors = flat_view_from_array(
                    factor_draws, "factor_draws");
                const auto residuals = flat_view_from_array(
                    residual_draws, "residual_draws");
                const auto chi = flat_view_from_array(
                    chi_square_draws, "chi_square_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_student_sample_factor(
                    correlation,
                    degrees,
                    factors,
                    residuals,
                    chi,
                    static_cast<std::int64_t>(factor_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("df"),
        py::arg("factor_draws"),
        py::arg("residual_draws"),
        py::arg("chi_square_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_student_conditional_from_normal_uniforms",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlations,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast> df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_uniforms,
            int n_threads) {
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
            scar::ConditionalSampleResult result;
            {
                const auto corr = flat_view_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto given = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto degrees = flat_view_from_array(df, "df");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                const auto uniforms = flat_view_from_array(
                    chi_square_uniforms, "chi_square_uniforms");
                py::gil_scoped_release release;
                result = scar::
                    multivariate_student_conditional_from_normal_uniforms(
                        corr,
                        correlation_rows,
                        dimension,
                        indices,
                        given,
                        degrees,
                        draws,
                        uniforms,
                        static_cast<std::int64_t>(draw_info.shape[0]),
                        n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("df"),
        py::arg("normal_draws"),
        py::arg("chi_square_uniforms"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_gaussian_conditional_from_uniforms",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlations,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            int n_threads) {
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
            scar::ConditionalSampleResult result;
            {
                const auto corr = flat_view_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto uniforms = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_conditional_from_uniforms(
                    corr,
                    correlation_rows,
                    dimension,
                    indices,
                    uniforms,
                    draws,
                    static_cast<std::int64_t>(draw_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("normal_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "equicorr_gaussian_conditional_from_uniforms",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                rho,
            int dimension,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            int n_threads) {
            const py::buffer_info draw_info = normal_draws.request();
            if (draw_info.ndim != 2) {
                throw py::value_error(
                    "normal_draws must be a two-dimensional array");
            }
            scar::ConditionalSampleResult result;
            {
                const auto parameters = flat_view_from_array(rho, "rho");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto uniforms = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                py::gil_scoped_release release;
                result = scar::
                    multivariate_gaussian_conditional_equicorrelation_from_uniforms(
                        parameters,
                        dimension,
                        indices,
                        uniforms,
                        draws,
                        static_cast<std::int64_t>(draw_info.shape[0]),
                        n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("rho"),
        py::arg("dimension"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("normal_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "multivariate_student_conditional_from_uniforms",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlations,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                normal_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_draws,
            int n_threads) {
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
            scar::ConditionalSampleResult result;
            {
                const auto corr = flat_view_from_array(
                    correlations, "correlations");
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto uniforms = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto degrees = flat_view_from_array(df, "df");
                const auto draws = flat_view_from_array(
                    normal_draws, "normal_draws");
                const auto chi = flat_view_from_array(
                    chi_square_draws, "chi_square_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_student_conditional_from_uniforms(
                    corr,
                    correlation_rows,
                    dimension,
                    indices,
                    uniforms,
                    degrees,
                    draws,
                    chi,
                    static_cast<std::int64_t>(draw_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlations"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("df"),
        py::arg("normal_draws"),
        py::arg("chi_square_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_student_conditional_from_normal_uniforms",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast> df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                factor_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residual_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_uniforms,
            int n_threads) {
            const py::buffer_info factor_info = factor_draws.request();
            const py::buffer_info residual_info = residual_draws.request();
            if (factor_info.ndim != 2
                || residual_info.ndim != 2
                || factor_info.shape[0] != residual_info.shape[0]
                || factor_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.rank())
                || residual_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "factor_draws and residual_draws have invalid shapes");
            }
            scar::ConditionalSampleResult result;
            {
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto given = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto degrees = flat_view_from_array(df, "df");
                const auto factors = flat_view_from_array(
                    factor_draws, "factor_draws");
                const auto residuals = flat_view_from_array(
                    residual_draws, "residual_draws");
                const auto uniforms = flat_view_from_array(
                    chi_square_uniforms, "chi_square_uniforms");
                py::gil_scoped_release release;
                result = scar::
                    multivariate_student_conditional_factor_from_normal_uniforms(
                        correlation,
                        indices,
                        given,
                        degrees,
                        factors,
                        residuals,
                        uniforms,
                        static_cast<std::int64_t>(factor_info.shape[0]),
                        n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("df"),
        py::arg("factor_draws"),
        py::arg("residual_draws"),
        py::arg("chi_square_uniforms"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_gaussian_conditional_from_uniforms",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                factor_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residual_draws,
            int n_threads) {
            const py::buffer_info factor_info = factor_draws.request();
            const py::buffer_info residual_info = residual_draws.request();
            if (factor_info.ndim != 2
                || residual_info.ndim != 2
                || factor_info.shape[0] != residual_info.shape[0]
                || factor_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.rank())
                || residual_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "factor_draws and residual_draws have invalid shapes");
            }
            scar::ConditionalSampleResult result;
            {
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto uniforms = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto factors = flat_view_from_array(
                    factor_draws, "factor_draws");
                const auto residuals = flat_view_from_array(
                    residual_draws, "residual_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_gaussian_conditional_factor(
                    correlation,
                    indices,
                    uniforms,
                    factors,
                    residuals,
                    static_cast<std::int64_t>(factor_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("factor_draws"),
        py::arg("residual_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_student_conditional_from_uniforms",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                given_indices,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                given_uniforms,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                df,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                factor_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residual_draws,
            py::array_t<double, py::array::c_style | py::array::forcecast>
                chi_square_draws,
            int n_threads) {
            const py::buffer_info factor_info = factor_draws.request();
            const py::buffer_info residual_info = residual_draws.request();
            if (factor_info.ndim != 2
                || residual_info.ndim != 2
                || factor_info.shape[0] != residual_info.shape[0]
                || factor_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.rank())
                || residual_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "factor_draws and residual_draws have invalid shapes");
            }
            scar::ConditionalSampleResult result;
            {
                const auto indices = int_vector_from_array(
                    given_indices, "given_indices");
                const auto uniforms = flat_view_from_array(
                    given_uniforms, "given_uniforms");
                const auto degrees = flat_view_from_array(df, "df");
                const auto factors = flat_view_from_array(
                    factor_draws, "factor_draws");
                const auto residuals = flat_view_from_array(
                    residual_draws, "residual_draws");
                const auto chi = flat_view_from_array(
                    chi_square_draws, "chi_square_draws");
                py::gil_scoped_release release;
                result = scar::multivariate_student_conditional_factor(
                    correlation,
                    indices,
                    uniforms,
                    degrees,
                    factors,
                    residuals,
                    chi,
                    static_cast<std::int64_t>(factor_info.shape[0]),
                    n_threads);
            }
            return conditional_sample_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("given_indices"),
        py::arg("given_uniforms"),
        py::arg("df"),
        py::arg("factor_draws"),
        py::arg("residual_draws"),
        py::arg("chi_square_draws"),
        py::arg("n_threads") = 1);

    m.def(
        "dense_gaussian_rosenblatt_transform",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast> u,
            int n_threads) {
            const py::buffer_info correlation_info = correlation.request();
            const py::buffer_info observation_info = u.request();
            if (correlation_info.ndim != 2
                || correlation_info.shape[0] != correlation_info.shape[1]
                || correlation_info.shape[0] < 1
                || observation_info.ndim != 2
                || observation_info.shape[1] != correlation_info.shape[0]) {
                throw std::invalid_argument(
                    "correlation must have shape (d,d) and u must have "
                    "shape (n,d)");
            }
            scar::MultivariateRosenblattResult result;
            {
                const auto correlation_view = flat_view_from_array(
                    correlation, "correlation");
                const scar::ObservationView observations{
                    static_cast<const double*>(observation_info.ptr),
                    static_cast<std::size_t>(observation_info.shape[0]),
                    static_cast<int>(observation_info.shape[1]),
                };
                py::gil_scoped_release release;
                result = scar::gaussian_rosenblatt_dense(
                    correlation_view,
                    static_cast<int>(correlation_info.shape[0]),
                    observations,
                    n_threads);
            }
            return multivariate_rosenblatt_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("u"),
        py::arg("n_threads") = 1);

    m.def(
        "equicorr_gaussian_rosenblatt_transform",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                rho,
            py::array_t<double, py::array::c_style | py::array::forcecast> u,
            int n_threads) {
            const py::buffer_info observation_info = u.request();
            if (observation_info.ndim != 2) {
                throw py::value_error("u must be a two-dimensional array");
            }
            scar::MultivariateRosenblattResult result;
            {
                const auto parameters = flat_view_from_array(rho, "rho");
                const scar::ObservationView observations{
                    static_cast<const double*>(observation_info.ptr),
                    static_cast<std::size_t>(observation_info.shape[0]),
                    static_cast<int>(observation_info.shape[1]),
                };
                py::gil_scoped_release release;
                result = scar::gaussian_rosenblatt_equicorrelation(
                    parameters, observations, n_threads);
            }
            return multivariate_rosenblatt_result_to_dict(result);
        },
        py::arg("rho"),
        py::arg("u"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_gaussian_rosenblatt_transform",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast> u,
            int n_threads) {
            const py::buffer_info observation_info = u.request();
            if (observation_info.ndim != 2
                || observation_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "u must have shape (n, correlation.dimension)");
            }
            scar::MultivariateRosenblattResult result;
            {
                const scar::ObservationView observations{
                    static_cast<const double*>(observation_info.ptr),
                    static_cast<std::size_t>(observation_info.shape[0]),
                    static_cast<int>(observation_info.shape[1]),
                };
                py::gil_scoped_release release;
                result = scar::gaussian_rosenblatt_factor(
                    correlation, observations, n_threads);
            }
            return multivariate_rosenblatt_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("u"),
        py::arg("n_threads") = 1);

    m.def(
        "factor_student_rosenblatt_transform",
        [](
            const scar::FactorCorrelationOperator& correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast> u,
            py::array_t<double, py::array::c_style | py::array::forcecast> df,
            int n_threads) {
            const py::buffer_info observation_info = u.request();
            if (observation_info.ndim != 2
                || observation_info.shape[1]
                    != static_cast<py::ssize_t>(correlation.dimension())) {
                throw std::invalid_argument(
                    "u must have shape (n, correlation.dimension)");
            }
            scar::MultivariateRosenblattResult result;
            {
                const scar::ObservationView observations{
                    static_cast<const double*>(observation_info.ptr),
                    static_cast<std::size_t>(observation_info.shape[0]),
                    static_cast<int>(observation_info.shape[1]),
                };
                const auto degrees = flat_view_from_array(df, "df");
                py::gil_scoped_release release;
                result = scar::student_rosenblatt_factor(
                    correlation, observations, degrees, n_threads);
            }
            return multivariate_rosenblatt_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("u"),
        py::arg("df"),
        py::arg("n_threads") = 1);

    m.def(
        "radial_uniform_summary",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                residuals,
            int n_threads) {
            const py::buffer_info info = residuals.request();
            if (info.ndim != 2 || info.shape[1] < 1) {
                throw std::invalid_argument(
                    "residuals must have shape (n,d), d >= 1");
            }
            scar::RadialSummaryResult result;
            {
                const scar::ObservationView values{
                    static_cast<const double*>(info.ptr),
                    static_cast<std::size_t>(info.shape[0]),
                    static_cast<int>(info.shape[1]),
                };
                py::gil_scoped_release release;
                result = scar::radial_uniform_summary(
                    values, n_threads);
            }
            return radial_summary_result_to_dict(result);
        },
        py::arg("residuals"),
        py::arg("n_threads") = 1);

    m.def(
        "dense_student_rosenblatt_transform",
        [](
            py::array_t<double, py::array::c_style | py::array::forcecast>
                correlation,
            py::array_t<double, py::array::c_style | py::array::forcecast> u,
            py::array_t<double, py::array::c_style | py::array::forcecast> df,
            int n_threads) {

            const py::buffer_info correlation_info = correlation.request();
            const py::buffer_info observation_info = u.request();
            if (correlation_info.ndim != 2
                || correlation_info.shape[0] != correlation_info.shape[1]
                || observation_info.ndim != 2
                || observation_info.shape[1] != correlation_info.shape[0]
                || correlation_info.shape[0] <= 0
                || correlation_info.shape[0]
                    > static_cast<py::ssize_t>(
                        std::numeric_limits<int>::max())) {
                throw std::invalid_argument(
                    "correlation must have shape (d, d) and u must have "
                    "shape (n, d), with d >= 1");
            }

            scar::DenseStudentRosenblattResult result;
            {
                const auto correlation_view = flat_view_from_array(
                    correlation, "correlation");
                const auto df_view = flat_view_from_array(df, "df");
                const scar::ObservationView observations{
                    static_cast<const double*>(observation_info.ptr),
                    static_cast<std::size_t>(observation_info.shape[0]),
                    static_cast<int>(observation_info.shape[1]),
                };
                py::gil_scoped_release release;
                result = scar::student_rosenblatt_dense(
                    correlation_view,
                    static_cast<int>(correlation_info.shape[0]),
                    observations,
                    df_view,
                    n_threads);
            }

            return multivariate_rosenblatt_result_to_dict(result);
        },
        py::arg("correlation"),
        py::arg("u"),
        py::arg("df"),
        py::arg("n_threads") = 1);
}

}  // namespace pyscarcopula::bindings
