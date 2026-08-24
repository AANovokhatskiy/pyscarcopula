#include "array.hpp"
#include "module.hpp"

#include "scar/copula.hpp"
#include "scar/copula/multivariate/gaussian/conditional.hpp"
#include "scar/copula/multivariate/student/conditional.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"
#include "scar/core/checked_arithmetic.hpp"

#include <pybind11/stl.h>

#include <limits>
#include <stdexcept>

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
    output["values"] = matrix_to_array(
        result.values,
        static_cast<std::size_t>(result.n_rows),
        static_cast<std::size_t>(result.n_free));
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["n_threads_requested"] = result.n_threads_requested;
    output["parallel_blocks"] = result.parallel_blocks;
    output["correlation_factorizations"] =
        result.correlation_factorizations;
    return output;
}

}  // namespace

void bind_multivariate(py::module_& m) {
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
            py::array_t<double, py::array::c_style | py::array::forcecast> u,
            std::size_t dimension_tile,
            int n_threads) {
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
           py::array_t<double, py::array::c_style | py::array::forcecast> u,
           py::array_t<double, py::array::c_style | py::array::forcecast> r,
           std::int64_t row_offset,
           int n_threads) {
            scar::MultivariateRowsResult result;
            const scar::Observations observations =
                observations_from_array(u);
            const std::vector<double> parameters = vector_from_array(r);
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

            py::dict diagnostics;
            diagnostics["n_threads_requested"] =
                result.n_threads_requested;
            diagnostics["parallel_blocks"] = result.parallel_blocks;
            diagnostics["correlation_factorizations"] =
                result.correlation_factorizations;

            py::dict out;
            out["residuals"] = vector_to_array(result.residuals);
            out["n_rows"] = result.n_rows;
            out["dimension"] = result.dimension;
            out["status"] = static_cast<int>(result.status);
            out["failure_index"] = result.failure.index;
            out["failure_coordinate"] = result.failure.coordinate;
            out["diagnostics"] = std::move(diagnostics);
            return out;
        },
        py::arg("correlation"),
        py::arg("u"),
        py::arg("df"),
        py::arg("n_threads") = 1);
}

}  // namespace pyscarcopula::bindings
