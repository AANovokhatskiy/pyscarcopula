#include "scar/copula.hpp"
#include "scar/copula/capability.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/sampling.hpp"
#include "scar/copula/multivariate/student/factor_grid.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"
#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/math/normal.hpp"
#include "scar/model_policy.hpp"
#include "scar/numerical_validation.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kDimension = 3;
constexpr double kUniquenessMin = 1e-6;

scar::DoubleView view(const std::vector<double>& values) {
    return {values.data(), values.size()};
}

bool close(double first, double second, double tolerance = 3e-11) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

bool close_vectors(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double tolerance = 3e-11) {

    if (first.size() != second.size()) {
        return false;
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (!close(first[index], second[index], tolerance)) {
            return false;
        }
    }
    return true;
}

bool finite_values(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

std::vector<double> flatten(const scar::Observations& observations) {
    std::vector<double> output;
    for (const auto& row : observations) {
        output.insert(output.end(), row.begin(), row.end());
    }
    return output;
}

scar::CopulaSpec equicorr_spec() {
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::EquicorrGaussian;
    spec.dim = kDimension;
    spec.transform = scar::Transform::GaussianTanh;
    spec.correlation_kind = scar::CorrelationKind::Equicorrelation;
    return spec;
}

scar::CopulaSpec dense_stochastic_student_spec(
    const std::vector<double>& correlation,
    scar::CorrelationKind kind) {

    const auto prepared = scar::prepare_dense_correlation(
        view(correlation), kDimension);
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = kDimension;
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0 + 1e-6;
    spec.correlation_kind = kind;
    spec.dense_inverse_cholesky() = prepared.inverse_cholesky;
    spec.dense_log_determinant() = prepared.log_determinant;
    return spec;
}

scar::CopulaSpec factor_stochastic_student_spec(
    const std::shared_ptr<const scar::FactorCorrelationOperator>& factor) {

    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = kDimension;
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0 + 1e-6;
    spec.correlation_kind = scar::CorrelationKind::Factor;
    spec.factor_operator() = factor;
    spec.dense_log_determinant() = factor->logdet();
    return spec;
}

}  // namespace

int run_equicorr_stochastic_student_tests() {
    constexpr std::array<scar::NativeOperation, 7> common_operations{
        scar::NativeOperation::ParameterTransformBoundsInitialization,
        scar::NativeOperation::RowGridDensityGradient,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::NativeOperation::RosenblattResidual,
        scar::NativeOperation::RadialGofSummary,
        scar::NativeOperation::UnconditionalSamplingTransform,
        scar::NativeOperation::ConditionalSamplingTransform,
    };
    constexpr std::array<scar::DynamicsKind, 3> supported_dynamics{
        scar::DynamicsKind::Mle,
        scar::DynamicsKind::Gas,
        scar::DynamicsKind::ScarTmOu,
    };

    // The product identities advertise the exact same common operation set
    // for MLE/GAS/OU. State filtering is dynamic-only; Jacobi is unsupported.
    const auto equicorr_descriptor = scar::make_typed_model_descriptor(
        scar::NativeModelId::EquicorrGaussian,
        kDimension,
        scar::CorrelationKind::Equicorrelation);
    for (scar::DynamicsKind dynamics : supported_dynamics) {
        for (scar::NativeOperation operation : common_operations) {
            const auto capability = scar::query_capability(
                equicorr_descriptor, operation, dynamics);
            if (!capability.supported || !capability.reason.empty()) {
                return 1;
            }
        }
        const auto state = scar::query_capability(
            equicorr_descriptor,
            scar::NativeOperation::StateFilterSmoother,
            dynamics);
        if (state.supported != (dynamics != scar::DynamicsKind::Mle)) {
            return 2;
        }
    }
    const auto equicorr_jacobi = scar::query_capability(
        equicorr_descriptor,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::DynamicsKind::ScarTmJacobi);
    if (equicorr_jacobi.supported || equicorr_jacobi.reason.empty()) {
        return 3;
    }

    constexpr std::array<scar::CorrelationKind, 5> student_modes{
        scar::CorrelationKind::Fixed,
        scar::CorrelationKind::Shrinkage,
        scar::CorrelationKind::DenseCholesky,
        scar::CorrelationKind::Factor,
        scar::CorrelationKind::FactorJointDynamicEstimation,
    };
    for (scar::CorrelationKind mode : student_modes) {
        const auto descriptor = scar::make_typed_model_descriptor(
            scar::NativeModelId::StochasticStudent,
            kDimension,
            mode,
            0,
            mode == scar::CorrelationKind::FactorJointDynamicEstimation
                ? scar::FactorEstimationKind::Joint
                : scar::FactorEstimationKind::TwoStage);
        for (scar::DynamicsKind dynamics : supported_dynamics) {
            for (scar::NativeOperation operation : common_operations) {
                const auto capability = scar::query_capability(
                    descriptor, operation, dynamics);
                const bool dynamic_joint =
                    mode == scar::CorrelationKind::FactorJointDynamicEstimation
                    && dynamics != scar::DynamicsKind::Mle;
                if (capability.supported == dynamic_joint
                    || (!capability.supported && capability.reason.empty())) {
                    return 4;
                }
            }
            const auto state = scar::query_capability(
                descriptor,
                scar::NativeOperation::StateFilterSmoother,
                dynamics);
            const bool expected_state = dynamics != scar::DynamicsKind::Mle
                && mode
                    != scar::CorrelationKind::FactorJointDynamicEstimation;
            if (state.supported != expected_state
                || (!state.supported && state.reason.empty())) {
                return 5;
            }
        }
        const auto jacobi = scar::query_capability(
            descriptor,
            scar::NativeOperation::LikelihoodObjectiveGradient,
            scar::DynamicsKind::ScarTmJacobi);
        if (jacobi.supported || jacobi.reason.empty()) {
            return 6;
        }
    }

    const scar::Observations observations{
        {0.20, 0.40, 0.60},
        {0.75, 0.35, 0.55},
        {0.10, 0.80, 0.45},
        {0.90, 0.25, 0.70},
    };
    const std::vector<double> flat = flatten(observations);
    const scar::ObservationView observation_view{
        flat.data(), observations.size(), kDimension};

    // Equicorrelation sufficient statistics are created once natively and
    // reused by row, grid, objective and dynamic-emission entry points.
    scar::CopulaSpec equicorr = equicorr_spec();
    const auto preparation_one = scar::prepare_equicorr_sufficient_statistics(
        observation_view, 2, 1);
    const auto preparation_four = scar::prepare_equicorr_sufficient_statistics(
        observation_view, 2, 4);
    if (!preparation_one.is_ok() || !preparation_four.is_ok()
        || !close_vectors(preparation_one.sum_z, preparation_four.sum_z)
        || !close_vectors(preparation_one.sum_z2, preparation_four.sum_z2)
        || preparation_one.sum_z.size() != observations.size()) {
        return 7;
    }
    const auto valid_statistics = scar::validate_equicorr_prepared_statistics(
        view(preparation_one.sum_z),
        view(preparation_one.sum_z2),
        kDimension,
        1e-10);
    if (!valid_statistics.is_ok()) {
        return 8;
    }
    equicorr.equicorr_sum_scores() = preparation_one.sum_z;
    equicorr.equicorr_sum_squares() = preparation_one.sum_z2;

    const std::vector<double> correlations{0.0, 0.10, -0.10, 0.20};
    const auto direct_rows = scar::multivariate_log_pdf_and_grad(
        equicorr, observations, correlations, 0, 4);
    const auto prepared_rows = scar::equicorr_log_pdf_and_grad_from_stats(
        equicorr,
        view(preparation_one.sum_z),
        view(preparation_one.sum_z2),
        correlations,
        4);
    if (!direct_rows.is_ok() || !prepared_rows.is_ok()
        || !close_vectors(direct_rows.log_pdf, prepared_rows.log_pdf)
        || !close_vectors(direct_rows.dlog_dr, prepared_rows.dlog_dr)) {
        return 9;
    }
    const std::vector<double> state_grid{-0.75, 0.0, 0.80};
    const auto direct_grid = scar::multivariate_pdf_and_grad_grid(
        equicorr, observations, state_grid, 0, 4);
    const auto prepared_grid = scar::equicorr_pdf_and_grad_grid_from_stats(
        equicorr,
        view(preparation_one.sum_z),
        view(preparation_one.sum_z2),
        state_grid,
        4);
    if (!direct_grid.is_ok() || !prepared_grid.is_ok()
        || !close_vectors(direct_grid.pdf.values, prepared_grid.pdf.values)
        || !close_vectors(
            direct_grid.d_pdf_dx.values,
            prepared_grid.d_pdf_dx.values)) {
        return 10;
    }

    scar::StaticCopulaEvaluator equicorr_objective(
        equicorr,
        preparation_one.sum_z,
        preparation_one.sum_z2,
        4);
    const double raw_rho = 0.25;
    const double physical_rho = scar::copula_transform(
        equicorr, {raw_rho})[0];
    const auto transformed_objective =
        equicorr_objective.transformed_objective(raw_rho);
    const auto physical_objective = equicorr_objective.objective(physical_rho);
    if (!transformed_objective.is_ok() || !physical_objective.is_ok()
        || !close(
            transformed_objective.negative_log_likelihood,
            physical_objective.negative_log_likelihood)
        || !std::isfinite(transformed_objective.negative_gradient)) {
        return 11;
    }
    const auto equicorr_policy = scar::equicorr_fit_parameter_policy();
    if (!equicorr_policy.is_ok() || equicorr_policy.value.initial != 0.5
        || !equicorr_policy.value.has_lower
        || !equicorr_policy.value.has_upper) {
        return 12;
    }

    scar::PreparedDynamicEmission equicorr_emission(equicorr);
    auto equicorr_workspace = equicorr_emission.make_workspace(true);
    const auto equicorr_row = equicorr_emission.evaluate_state(
        observations[0].data(), 0, raw_rho, true, equicorr_workspace);
    std::vector<double> grid_parameters;
    std::vector<double> grid_derivatives;
    equicorr_emission.prepare_grid_transform(
        state_grid, grid_parameters, grid_derivatives);
    std::vector<double> emission_density;
    std::vector<double> emission_gradient;
    equicorr_emission.fill_density_and_gradient_grid(
        flat.data(),
        static_cast<std::int64_t>(observations.size()),
        grid_parameters,
        grid_derivatives,
        emission_density,
        emission_gradient,
        4);
    if (equicorr_emission.kind()
            != scar::DynamicEmissionKind::Equicorrelation
        || !equicorr_emission.is_supported_for_ou()
        || !equicorr_row.is_ok()
        || !close(equicorr_row.parameter, physical_rho)
        || !close_vectors(emission_density, direct_grid.pdf.values)
        || !close_vectors(emission_gradient, direct_grid.d_pdf_dx.values)) {
        return 13;
    }

    const std::vector<double> zero_rho{0.0, 0.0};
    const std::vector<double> normal_draws{
        0.0, 1.0, -1.0,
        0.5, -0.5, 0.25,
    };
    const std::vector<double> common_draws{2.0, -2.0};
    const auto common_count = scar::equicorr_gaussian_common_draw_count(
        view(zero_rho), kDimension, 2);
    const auto equicorr_sample =
        scar::multivariate_gaussian_sample_equicorrelation(
            view(zero_rho),
            kDimension,
            view(normal_draws),
            view(common_draws),
            2,
            4);
    const std::vector<int> given_indices{0};
    const std::vector<double> given_uniforms{0.25, 0.75};
    const std::vector<double> conditional_normals{-0.3, 0.8, 0.4, -0.6};
    const auto equicorr_conditional =
        scar::multivariate_gaussian_conditional_equicorrelation_from_uniforms(
            view(zero_rho),
            kDimension,
            given_indices,
            view(given_uniforms),
            view(conditional_normals),
            2,
            4);
    const std::vector<double> shared_zero_rho{0.0};
    const auto equicorr_residuals = scar::gaussian_rosenblatt_equicorrelation(
        view(shared_zero_rho), observation_view, 4);
    const auto equicorr_radial = scar::radial_uniform_summary(
        {equicorr_residuals.residuals.data(), observations.size(), kDimension},
        4);
    if (!common_count.is_ok() || common_count.value != 2) {
        return 22;
    }
    if (!equicorr_sample.is_ok()) {
        return 23;
    }
    if (!equicorr_conditional.is_ok()
        || equicorr_conditional.n_free != 2) {
        return 24;
    }
    if (!equicorr_residuals.is_ok()
        || !close_vectors(equicorr_residuals.residuals, flat, 2e-15)) {
        return 25;
    }
    if (!equicorr_radial.is_ok()) {
        return 26;
    }
    for (std::size_t index = 0; index < normal_draws.size(); ++index) {
        if (!close(
                equicorr_sample.values[index],
                scar::math::normal_cdf(normal_draws[index]),
                2e-15)) {
            return 15;
        }
    }

    // Stochastic Student uses the same prepared Student owner for dense and
    // factor correlations, with raw state -> df handled entirely in C++.
    const std::vector<double> identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const auto zero_factor = std::make_shared<scar::FactorCorrelationOperator>(
        std::vector<double>(kDimension, 0.0),
        kDimension,
        1,
        kUniquenessMin);
    scar::CopulaSpec dense_student = dense_stochastic_student_spec(
        identity, scar::CorrelationKind::Fixed);
    scar::CopulaSpec factor_student = factor_stochastic_student_spec(
        zero_factor);
    const auto stochastic_policy = scar::student_fit_parameter_policy(
        kDimension, true);
    if (!stochastic_policy.is_ok()
        || stochastic_policy.value.initial != 5.0
        || !stochastic_policy.value.has_lower
        || stochastic_policy.value.has_upper
        || !(stochastic_policy.value.lower > 2.0)) {
        return 16;
    }

    const std::vector<double> raw_df_rows{-0.4, 0.1, 0.6, 1.0};
    const std::vector<double> physical_df_rows = scar::copula_transform(
        dense_student, raw_df_rows);
    const auto dense_student_rows = scar::multivariate_log_pdf_and_grad(
        dense_student, observations, physical_df_rows, 0, 4);
    const auto factor_student_rows = scar::multivariate_log_pdf_and_grad(
        factor_student, observations, physical_df_rows, 0, 4);
    const auto dense_student_grid = scar::multivariate_pdf_and_grad_grid(
        dense_student, observations, state_grid, 0, 4);
    const auto factor_student_grid =
        scar::factor_student_stochastic_pdf_and_grad_grid(
            *zero_factor,
            flat.data(),
            observations.size(),
            state_grid.data(),
            state_grid.size(),
            dense_student.offset,
            2,
            4);
    if (!dense_student_rows.is_ok() || !factor_student_rows.is_ok()) {
        return 27;
    }
    if (!dense_student_grid.is_ok() || !factor_student_grid.is_ok()) {
        return 28;
    }
    if (!close_vectors(
            dense_student_rows.log_pdf, factor_student_rows.log_pdf)
        || !close_vectors(
            dense_student_rows.dlog_dr, factor_student_rows.dlog_dr)) {
        return 29;
    }
    if (!close_vectors(
            dense_student_grid.pdf.values, factor_student_grid.pdf)
        || !close_vectors(
            dense_student_grid.d_pdf_dx.values,
            factor_student_grid.d_pdf_dx)) {
        return 30;
    }

    scar::StaticCopulaEvaluator dense_student_objective(
        dense_student, observations, 4);
    scar::StaticCopulaEvaluator factor_student_objective(
        factor_student, observations, 4);
    const double objective_df = scar::copula_transform(
        dense_student, {0.25})[0];
    const auto dense_objective = dense_student_objective.objective(
        objective_df, true);
    const auto factor_objective = factor_student_objective.objective(
        objective_df, false);
    if (!dense_objective.is_ok()) {
        return 31;
    }
    if (!factor_objective.is_ok()) {
        return 33;
    }
    if (!close(
            dense_objective.negative_log_likelihood,
            factor_objective.negative_log_likelihood)
        || !close(
            dense_objective.negative_gradient,
            factor_objective.negative_gradient)) {
        return 32;
    }

    scar::PreparedDynamicEmission dense_student_emission(dense_student);
    scar::PreparedDynamicEmission factor_student_emission(factor_student);
    auto dense_student_workspace =
        dense_student_emission.make_workspace(true);
    auto factor_student_workspace =
        factor_student_emission.make_workspace(true);
    const auto dense_student_row = dense_student_emission.evaluate_state(
        observations[0].data(), 0, 0.25, true, dense_student_workspace);
    const auto factor_student_row = factor_student_emission.evaluate_state(
        observations[0].data(), 0, 0.25, true, factor_student_workspace);
    if (dense_student_emission.kind() != scar::DynamicEmissionKind::Student
        || factor_student_emission.kind()
            != scar::DynamicEmissionKind::Student
        || !dense_student_emission.is_supported_for_ou()
        || !factor_student_emission.is_supported_for_ou()
        || !dense_student_row.is_ok() || !factor_student_row.is_ok()
        || !close(dense_student_row.parameter, factor_student_row.parameter)
        || !close(dense_student_row.log_pdf, factor_student_row.log_pdf)
        || !close(
            dense_student_row.dlog_dparameter,
            factor_student_row.dlog_dparameter)) {
        return 19;
    }

    const std::vector<double> df{5.0};
    const auto dense_student_residuals = scar::student_rosenblatt_dense(
        view(identity), kDimension, observation_view, view(df), 4);
    const auto factor_student_residuals = scar::student_rosenblatt_factor(
        *zero_factor, observation_view, view(df), 4);
    const auto student_radial = scar::radial_uniform_summary(
        {dense_student_residuals.residuals.data(),
         observations.size(), kDimension},
        4);
    const std::vector<double> sample_df{5.0, 7.0};
    const std::vector<double> chi_square{5.0, 7.0};
    const std::vector<double> factor_draws{0.75, -0.25};
    const auto dense_student_sample = scar::multivariate_student_sample_dense(
        view(identity), kDimension, view(sample_df), view(normal_draws),
        view(chi_square), 2, 4);
    const auto factor_student_sample = scar::multivariate_student_sample_factor(
        *zero_factor, view(sample_df), view(factor_draws), view(normal_draws),
        view(chi_square), 2, 4);
    const std::vector<double> full_residuals{
        0.0, -0.3, 0.8,
        0.0, 0.4, -0.6,
    };
    const std::vector<double> conditional_chi_square{6.0, 8.0};
    const auto dense_student_conditional =
        scar::multivariate_student_conditional_from_uniforms(
            view(identity), 1, kDimension, given_indices,
            view(given_uniforms), view(sample_df), view(conditional_normals),
            view(conditional_chi_square), 2, 4);
    const auto factor_student_conditional =
        scar::multivariate_student_conditional_factor(
            *zero_factor, given_indices, view(given_uniforms), view(sample_df),
            view(factor_draws), view(full_residuals),
            view(conditional_chi_square), 2, 4);
    if (!dense_student_residuals.is_ok()
        || !factor_student_residuals.is_ok() || !student_radial.is_ok()
        || !dense_student_sample.is_ok() || !factor_student_sample.is_ok()
        || !dense_student_conditional.is_ok()
        || !factor_student_conditional.is_ok()
        || !close_vectors(
            dense_student_residuals.residuals,
            factor_student_residuals.residuals)
        || !close_vectors(
            dense_student_sample.values, factor_student_sample.values)
        || !close_vectors(
            dense_student_conditional.values,
            factor_student_conditional.values)) {
        return 20;
    }

    // Invalid prepared statistics and correlation paths are typed failures.
    const std::vector<double> impossible_sum{10.0};
    const std::vector<double> impossible_sum2{1.0};
    const auto invalid_statistics = scar::validate_equicorr_prepared_statistics(
        view(impossible_sum), view(impossible_sum2), kDimension, 1e-10);
    const std::vector<double> invalid_rho{-0.75};
    const auto invalid_path = scar::validate_equicorrelation_path(
        view(invalid_rho), kDimension, 1);
    if (invalid_statistics.code
            != scar::NumericalValidationCode::CauchyBound
        || invalid_path != scar::Status::InvalidParameter
        || !finite_values(equicorr_radial.values)
        || !finite_values(student_radial.values)) {
        return 21;
    }

    return 0;
}
