#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/copula/multivariate/gaussian/density.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/sampling.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/factor_grid.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/copula/spec.hpp"
#include "scar/math/gamma.hpp"

#include <cmath>
#include <vector>

namespace {

bool close(double first, double second, double tolerance = 2e-13) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

}  // namespace

int run_multivariate_model_tests() {
    namespace student = scar::copula::multivariate::student;
    const double probabilities[] = {0.25, 0.5, 0.75};
    student::PpfTableConfig config;
    config.max_table_bytes = 1;
    const auto ppf = student::prepare_ppf_table(
        scar::DoubleView{probabilities, 3}, config);
    if (!ppf.is_ok() || ppf.value.has_table || !ppf.value.table.empty()
        || ppf.value.nodes.front() != config.df_lo
        || ppf.value.nodes.back() != config.df_hi) {
        return 20;
    }
    const auto quantiles = student::evaluate_ppf_table(
        scar::DoubleView{ppf.value.observations.data(), 3},
        scar::DoubleView{ppf.value.nodes.data(), ppf.value.nodes.size()},
        scar::DoubleView{}, 1.0, 0, 3);
    if (!quantiles.is_ok() || quantiles.value.size() != 3
        || !close(quantiles.value[0], -1.0)
        || !close(quantiles.value[1], 0.0)
        || !close(quantiles.value[2], 1.0)) {
        return 21;
    }
    const auto bad_slice = student::evaluate_ppf_table(
        scar::DoubleView{probabilities, 3},
        scar::DoubleView{ppf.value.nodes.data(), ppf.value.nodes.size()},
        scar::DoubleView{}, 5.0, 3, 1);
    if (bad_slice.status != scar::Status::InvalidSize) {
        return 22;
    }
    const std::vector<double> identity{
        1.0, 0.0,
        0.0, 1.0,
    };
    const scar::DenseCorrelationPreparationResult dense =
        scar::prepare_dense_correlation(
            scar::DoubleView{identity.data(), identity.size()}, 2);
    if (!dense.is_ok()
        || dense.inverse_cholesky != identity
        || dense.log_determinant != 0.0) {
        return 1;
    }

    const scar::copula::multivariate::correlation::DenseCorrelation
        dense_model{identity, 0.0};
    const double scores[] = {-0.5, 0.8};
    if (!close(
            scar::copula::multivariate::gaussian::log_pdf(
                dense_model, 2, scores),
            0.0)) {
        return 2;
    }

    const scar::FactorCorrelationOperator factor(
        std::vector<double>{0.0, 0.0, 0.0}, 3, 1, 1e-6);
    const double factor_values[] = {0.25, -0.5, 0.75};
    double quadratic_form = 0.0;
    factor.quadratic_forms(factor_values, 1, &quadratic_form);
    if (!close(factor.logdet(), 0.0)
        || !close(quadratic_form, 0.875)) {
        return 3;
    }
    std::vector<double> projection;
    std::vector<double> solved;
    if (!close(
            scar::copula::multivariate::gaussian::log_pdf(
                factor, factor_values, projection, solved),
            0.0)) {
        return 4;
    }

    scar::CopulaSpec equicorrelation;
    equicorrelation.family = scar::CopulaFamily::EquicorrGaussian;
    equicorrelation.dim = 3;
    scar_internal::EquicorrStats stats;
    if (!scar_internal::equicorr_sufficient_statistics(
            equicorrelation, factor_values, stats)
        || !close(scar_internal::equicorr_log_pdf_from_stats(
            equicorrelation, stats, 0.0, nullptr), 0.0)) {
        return 5;
    }

    if (!close(scar_internal::student_cdf_value(0.0, 5.0), 0.5)
        || !(scar_internal::student_pdf_value(0.0, 5.0) > 0.0)) {
        return 6;
    }

    const std::vector<double> zero_normals(4, 0.0);
    const std::vector<double> chi_square{5.0, 5.0};
    const std::vector<double> degrees_of_freedom{5.0, 5.0};
    const scar::ConditionalSampleResult student_sample =
        scar::multivariate_student_sample_dense(
            scar::DoubleView{identity.data(), identity.size()},
            2,
            scar::DoubleView{
                degrees_of_freedom.data(), degrees_of_freedom.size()},
            scar::DoubleView{zero_normals.data(), zero_normals.size()},
            scar::DoubleView{chi_square.data(), chi_square.size()},
            2,
            2);
    if (!student_sample.is_ok() || student_sample.values.size() != 4) {
        return 7;
    }
    for (double value : student_sample.values) {
        if (!close(value, 0.5)) {
            return 8;
        }
    }

    const std::vector<double> chi_square_uniforms{0.5, 0.5};
    const scar::ConditionalSampleResult uniform_student_sample =
        scar::multivariate_student_sample_dense_from_uniforms(
            scar::DoubleView{identity.data(), identity.size()},
            2,
            scar::DoubleView{
                degrees_of_freedom.data(), degrees_of_freedom.size()},
            scar::DoubleView{zero_normals.data(), zero_normals.size()},
            scar::DoubleView{
                chi_square_uniforms.data(), chi_square_uniforms.size()},
            2,
            2);
    if (!uniform_student_sample.is_ok()
        || uniform_student_sample.values.size() != 4) {
        return 25;
    }
    for (double value : uniform_student_sample.values) {
        if (!close(value, 0.5)) {
            return 26;
        }
    }
    const double chi_square_median = scar::math::chi_square_quantile(0.5, 8.0);
    if (!close(chi_square_median, 7.344121497701794, 2e-11)
        || !close(
            scar::math::regularized_gamma_p(4.0, chi_square_median / 2.0),
            0.5,
            2e-13)) {
        return 27;
    }
    const std::vector<double> invalid_chi_square_uniforms{1.0, 0.5};
    const scar::ConditionalSampleResult invalid_uniform_student_sample =
        scar::multivariate_student_sample_dense_from_uniforms(
            scar::DoubleView{identity.data(), identity.size()},
            2,
            scar::DoubleView{
                degrees_of_freedom.data(), degrees_of_freedom.size()},
            scar::DoubleView{zero_normals.data(), zero_normals.size()},
            scar::DoubleView{
                invalid_chi_square_uniforms.data(),
                invalid_chi_square_uniforms.size()},
            2,
            1);
    if (invalid_uniform_student_sample.status != scar::Status::InvalidParameter) {
        return 28;
    }
    const std::vector<double> positive_rho{0.1, 0.2};
    const std::vector<double> mixed_rho{0.1, -0.2};
    const auto positive_common = scar::equicorr_gaussian_common_draw_count(
        {positive_rho.data(), positive_rho.size()}, 3, 2);
    const auto mixed_common = scar::equicorr_gaussian_common_draw_count(
        {mixed_rho.data(), mixed_rho.size()}, 3, 2);
    const auto invalid_common = scar::equicorr_gaussian_common_draw_count(
        {mixed_rho.data(), mixed_rho.size()}, 1, 2);
    if (!positive_common.is_ok() || positive_common.value != 2) {
        return 29;
    }
    if (!mixed_common.is_ok() || mixed_common.value != 0) {
        return 30;
    }
    if (invalid_common.status != scar::Status::InvalidSize) {
        return 31;
    }

    const scar::FactorCorrelationOperator student_factor(
        std::vector<double>{0.0, 0.0}, 2, 1, 1e-6);
    const double student_observations[] = {
        0.25, 0.75,
        0.40, 0.60,
    };
    const double student_raw_grid[] = {-1.0, 0.5};
    const double student_df_grid[] = {
        2.01 + std::log1p(std::exp(student_raw_grid[0])),
        2.01 + std::log1p(std::exp(student_raw_grid[1])),
    };
    const auto student_stochastic =
        scar::factor_student_stochastic_pdf_and_grad_grid(
            student_factor, student_observations, 2,
            student_raw_grid, 2, 2.01, 8, 2);
    const auto student_grid = scar::factor_student_log_pdf_and_dlog_ddf_grid(
        student_factor, student_observations, 2,
        student_df_grid, 2, 8, 2);
    const auto student_density = scar::factor_student_density_from_log_grid(
        student_grid.log_pdf.data(), student_grid.dlog_ddf.data(),
        student_grid.log_pdf.size());
    if (!student_stochastic.is_ok() || !student_grid.is_ok()
        || !student_density.is_ok()
        || student_stochastic.pdf.size() != 4
        || student_stochastic.d_pdf_dx.size() != 4) {
        return 23;
    }
    for (std::size_t cell = 0; cell < student_stochastic.pdf.size(); ++cell) {
        const double derivative =
            1.0 / (1.0 + std::exp(-student_raw_grid[cell % 2]));
        if (!close(student_stochastic.pdf[cell], student_density.pdf[cell])
            || !close(
                student_stochastic.d_pdf_dx[cell],
                student_density.d_pdf_ddf[cell] * derivative)) {
            return 24;
        }
    }

    const std::vector<double> observations{
        0.25, 0.75,
        0.60, 0.40,
    };
    const scar::ObservationView observation_view{
        observations.data(), 2, 2};
    const scar::MultivariateRosenblattResult dense_rosenblatt =
        scar::gaussian_rosenblatt_dense(
            scar::DoubleView{identity.data(), identity.size()},
            2,
            observation_view,
            2);
    const std::vector<double> zero_rho{0.0};
    const scar::MultivariateRosenblattResult equicorr_rosenblatt =
        scar::gaussian_rosenblatt_equicorrelation(
            scar::DoubleView{zero_rho.data(), zero_rho.size()},
            observation_view,
            2);
    if (!dense_rosenblatt.is_ok()
        || !equicorr_rosenblatt.is_ok()
        || dense_rosenblatt.residuals.size() != observations.size()
        || equicorr_rosenblatt.residuals.size() != observations.size()) {
        return 9;
    }
    for (std::size_t index = 0; index < observations.size(); ++index) {
        if (!close(dense_rosenblatt.residuals[index], observations[index])
            || !close(
                equicorr_rosenblatt.residuals[index], observations[index])) {
            return 10;
        }
    }
    return 0;
}
