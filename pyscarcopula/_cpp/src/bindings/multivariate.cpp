#include "common.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_multivariate(py::module_& m) {
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
            std::size_t n_obs = 0;
            std::size_t dimension = 0;
            std::size_t values = 0;
            if (!scar_internal::checked_nonnegative_size(
                    info.shape[0], n_obs)
                || !scar_internal::checked_nonnegative_size(
                    info.shape[1], dimension)
                || dimension
                    > static_cast<std::size_t>(
                        std::numeric_limits<int>::max())
                || !scar_internal::checked_size_mul(
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
}

}  // namespace pyscarcopula::bindings
