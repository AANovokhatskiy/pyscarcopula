#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/copula/multivariate/gaussian/density.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/sampling.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/spec.hpp"

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
