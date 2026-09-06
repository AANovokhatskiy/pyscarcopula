#include "scar/copula.hpp"
#include "scar/copula/capability.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/factor_parameterization.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/sampling.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/factor_density.hpp"
#include "scar/copula/multivariate/student/factor_grid.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kDimension = 3;
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

std::vector<double> as_double(
    const std::vector<std::int64_t>& values) {

    std::vector<double> result;
    result.reserve(values.size());
    for (std::int64_t value : values) {
        result.push_back(static_cast<double>(value));
    }
    return result;
}

std::vector<double> as_double(
    const std::vector<std::uint8_t>& values) {

    std::vector<double> result;
    result.reserve(values.size());
    for (std::uint8_t value : values) {
        result.push_back(static_cast<double>(value));
    }
    return result;
}

scar::CopulaSpec dense_spec(
    const std::vector<double>& correlation,
    scar::CorrelationKind kind) {

    const auto prepared = scar::prepare_dense_correlation(
        view(correlation), kDimension);
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = static_cast<int>(kDimension);
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0;
    spec.correlation_kind = kind;
    spec.dense_inverse_cholesky() = prepared.inverse_cholesky;
    spec.dense_log_determinant() = prepared.log_determinant;
    return spec;
}

scar::CopulaSpec factor_spec(
    const std::shared_ptr<const scar::FactorCorrelationOperator>& factor) {

    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = static_cast<int>(kDimension);
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0;
    spec.correlation_kind = scar::CorrelationKind::Factor;
    spec.factor_operator() = factor;
    spec.dense_log_determinant() = factor->logdet();
    return spec;
}

std::vector<double> flatten(const scar::Observations& observations) {
    std::vector<double> result;
    for (const auto& row : observations) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

}  // namespace

int run_multivariate_student_tests() {
    namespace student = scar::copula::multivariate::student;
    const std::vector<double> identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const scar::Observations observations{
        {0.20, 0.40, 0.60},
        {0.75, 0.35, 0.55},
        {0.10, 0.80, 0.45},
        {0.90, 0.25, 0.70},
    };
    const std::vector<double> flat_observations = flatten(observations);

    // Distribution, tails, exact quantiles, df derivatives and the large-df
    // expansion are direct numerical owners.
    const auto distribution = scar_internal::student_distribution_parameters(5.0);
    if (!close(scar_internal::student_cdf_value(0.0, 5.0), 0.5, 2e-15)
        || !close(
            scar_internal::student_pdf_value(0.0, distribution),
            scar_internal::student_pdf_value(0.0, 5.0),
            2e-15)
        || !close(
            scar_internal::student_survival_positive_value(1.25, 5.0),
            1.0 - scar_internal::student_cdf_refined_value(1.25, 5.0),
            2e-15)) {
        return 1;
    }
    for (double probability : {1e-10, 0.25, 0.5, 0.75, 1.0 - 1e-10}) {
        const double quantile = scar_internal::student_quantile_refined_value(
            probability, 5.0);
        const double recovered = scar_internal::student_cdf_refined_value(
            quantile, 5.0);
        if (!close(recovered, probability, 3e-12)) {
            return 2;
        }
    }
    double quantile = 0.0;
    double derivative = 0.0;
    scar_internal::student_quantile_value_and_derivative(
        0.82, 6.0, quantile, derivative);
    const double finite_difference = (
        scar_internal::student_quantile_refined_value(0.82, 6.0001)
        - scar_internal::student_quantile_refined_value(0.82, 5.9999))
        / 0.0002;
    double large_quantile = 0.0;
    double large_derivative = 0.0;
    scar_internal::student_quantile_large_df_value_and_derivative(
        0.82, 2000.0, large_quantile, large_derivative);
    if (!close(derivative, finite_difference, 2e-7)
        || !std::isfinite(large_derivative)
        || !close(
            large_quantile,
            scar_internal::student_quantile_refined_value(0.82, 2000.0),
            2e-12)) {
        return 3;
    }
    double survival = 0.0;
    double survival_derivative = 0.0;
    scar_internal::student_survival_positive_df_value_and_derivative(
        1.25, 5.0, survival, survival_derivative);
    const double survival_difference = (
        scar_internal::student_survival_positive_value(1.25, 5.0001)
        - scar_internal::student_survival_positive_value(1.25, 4.9999))
        / 0.0002;
    if (!close(
            survival,
            scar_internal::student_survival_positive_value(1.25, 5.0),
            2e-15)
        || !close(survival_derivative, survival_difference, 2e-8)) {
        return 4;
    }

    // PPF construction covers a real table, exact fallback under the memory
    // gate, synthetic interpolation, invalid configuration and tail policy.
    student::PpfTableConfig table_config;
    table_config.df_hi = 50.0;
    table_config.n_boundary = 1;
    table_config.n_lo = 2;
    table_config.n_hi = 2;
    table_config.max_table_bytes = 1024 * 1024;
    const auto ppf_table = student::prepare_ppf_table(
        view(flat_observations), table_config);
    student::PpfTableConfig memory_config = table_config;
    memory_config.max_table_bytes = 1;
    const auto exact_fallback = student::prepare_ppf_table(
        view(flat_observations), memory_config);
    student::PpfTableConfig bad_config = table_config;
    bad_config.df_lo = 2.1;
    const auto bad_table = student::prepare_ppf_table(
        view(flat_observations), bad_config);
    if (!ppf_table.is_ok() || !ppf_table.value.has_table
        || ppf_table.value.table.size()
            != ppf_table.value.nodes.size() * flat_observations.size()
        || !exact_fallback.is_ok() || exact_fallback.value.has_table
        || !exact_fallback.value.table.empty()
        || bad_table.status != scar::Status::InvalidParameter) {
        return 5;
    }
    const auto exact_evaluation = student::evaluate_ppf_table(
        view(exact_fallback.value.observations),
        view(exact_fallback.value.nodes),
        {},
        5.0,
        0,
        flat_observations.size());
    if (!exact_evaluation.is_ok()
        || exact_evaluation.value.size() != flat_observations.size()
        || !finite_values(exact_evaluation.value)) {
        return 6;
    }
    const std::vector<double> interpolation_nodes{3.0, 5.0, 10.0, 30.0};
    std::vector<double> interpolation_table;
    for (double node : interpolation_nodes) {
        for (std::size_t column = 0; column < kDimension; ++column) {
            interpolation_table.push_back(
                0.1 * node + static_cast<double>(column));
        }
    }
    const auto interpolated = student::interpolate_ppf_table(
        view(interpolation_nodes), view(interpolation_table), 7.0, kDimension);
    if (!interpolated.is_ok() || interpolated.value.size() != kDimension) {
        return 7;
    }
    for (std::size_t column = 0; column < kDimension; ++column) {
        if (!close(
                interpolated.value[column],
                0.7 + static_cast<double>(column),
                2e-14)) {
            return 8;
        }
    }

    // A prepared density reuses the same PPF cache and workspace. The second
    // evaluation must not grow its buffers; a df beyond the cache switches to
    // the native asymptotic policy.
    scar::CopulaSpec cached_spec = dense_spec(
        identity, scar::CorrelationKind::Fixed);
    auto& cache = student::ppf_cache(cached_spec);
    cache.observation_count = static_cast<std::int64_t>(observations.size());
    cache.nodes = ppf_table.value.nodes;
    cache.table = ppf_table.value.table;
    const auto prepared_density = scar_internal::prepare_student_density(
        cached_spec);
    scar_internal::StudentWorkspace workspace;
    double cached_log_pdf = 0.0;
    double cached_derivative = 0.0;
    if (!scar_internal::student_log_pdf_and_dlog_ddf(
            prepared_density,
            observations[0].data(),
            6.0,
            0,
            cached_log_pdf,
            cached_derivative,
            workspace)
        || workspace.diagnostics.ppf_cache_values != kDimension) {
        return 9;
    }
    const std::uint64_t growth_events = workspace.diagnostics.growth_events;
    if (!scar_internal::student_log_pdf_and_dlog_ddf(
            prepared_density,
            observations[0].data(),
            6.0,
            0,
            cached_log_pdf,
            cached_derivative,
            workspace)
        || workspace.diagnostics.growth_events != growth_events
        || workspace.diagnostics.ppf_cache_values != 2 * kDimension) {
        return 10;
    }
    if (!scar_internal::student_log_pdf_and_dlog_ddf(
            prepared_density,
            observations[0].data(),
            2000.0,
            0,
            cached_log_pdf,
            cached_derivative,
            workspace)
        || workspace.diagnostics.ppf_asymptotic_values != kDimension) {
        return 11;
    }

    // Every registered static policy cell is explicit. Factor two-stage and
    // joint policies share density/sampling owners; joint likelihood has the
    // additional loading-gradient owner below.
    struct PolicyCase {
        scar::CorrelationKind correlation;
        scar::FactorEstimationKind estimation;
    };
    constexpr std::array<PolicyCase, 5> policies{
        PolicyCase{scar::CorrelationKind::Fixed,
                   scar::FactorEstimationKind::TwoStage},
        PolicyCase{scar::CorrelationKind::Shrinkage,
                   scar::FactorEstimationKind::TwoStage},
        PolicyCase{scar::CorrelationKind::DenseCholesky,
                   scar::FactorEstimationKind::TwoStage},
        PolicyCase{scar::CorrelationKind::Factor,
                   scar::FactorEstimationKind::TwoStage},
        PolicyCase{scar::CorrelationKind::Factor,
                   scar::FactorEstimationKind::Joint},
    };
    constexpr std::array<scar::NativeOperation, 7> supported_operations{
        scar::NativeOperation::ParameterTransformBoundsInitialization,
        scar::NativeOperation::RowGridDensityGradient,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::NativeOperation::RosenblattResidual,
        scar::NativeOperation::RadialGofSummary,
        scar::NativeOperation::UnconditionalSamplingTransform,
        scar::NativeOperation::ConditionalSamplingTransform,
    };
    for (const PolicyCase& policy : policies) {
        const auto descriptor = scar::make_typed_model_descriptor(
            scar::NativeModelId::Student,
            static_cast<int>(kDimension),
            policy.correlation,
            0,
            policy.estimation);
        for (scar::NativeOperation operation : supported_operations) {
            const auto capability = scar::query_capability(
                descriptor, operation, scar::DynamicsKind::Mle);
            if (!capability.supported || !capability.reason.empty()) {
                return 12;
            }
        }
        for (scar::NativeOperation operation : {
                 scar::NativeOperation::PointDensityDerivatives,
                 scar::NativeOperation::StateFilterSmoother,
                 scar::NativeOperation::ArbitraryConditionalMcmc,
                 scar::NativeOperation::EdgeStructureSelectionScore}) {
            const auto capability = scar::query_capability(
                descriptor, operation, scar::DynamicsKind::Mle);
            if (capability.supported || capability.reason.empty()) {
                return 13;
            }
        }
        const auto dynamic = scar::query_capability(
            descriptor,
            scar::NativeOperation::LikelihoodObjectiveGradient,
            scar::DynamicsKind::Gas);
        if (dynamic.supported || dynamic.reason.empty()) {
            return 14;
        }
    }

    scar::CopulaSpec transform_spec = dense_spec(
        identity, scar::CorrelationKind::Fixed);
    const std::vector<double> raw_parameters{-2.0, 0.0, 2.0};
    const std::vector<double> physical_parameters = scar::copula_transform(
        transform_spec, raw_parameters);
    const std::vector<double> recovered_parameters =
        scar::copula_inverse_transform(transform_spec, physical_parameters);
    const std::vector<double> transform_derivative = scar::copula_dtransform(
        transform_spec, raw_parameters);
    if (!close_vectors(raw_parameters, recovered_parameters, 3e-14)
        || !finite_values(transform_derivative)) {
        return 15;
    }
    for (std::size_t index = 0; index < physical_parameters.size(); ++index) {
        if (!(physical_parameters[index] > 2.0)
            || !(transform_derivative[index] > 0.0)) {
            return 16;
        }
    }

    // Dense fixed/shrinkage/Cholesky and factor paths agree at identity.
    const auto zero_factor = std::make_shared<scar::FactorCorrelationOperator>(
        std::vector<double>(kDimension, 0.0),
        kDimension,
        1,
        kUniquenessMin);
    std::vector<scar::CopulaSpec> specifications;
    specifications.push_back(dense_spec(
        identity, scar::CorrelationKind::Fixed));
    specifications.push_back(dense_spec(
        identity, scar::CorrelationKind::Shrinkage));
    specifications.push_back(dense_spec(
        identity, scar::CorrelationKind::DenseCholesky));
    specifications.push_back(factor_spec(zero_factor));
    const std::vector<double> df{5.0};
    const std::vector<double> state_grid{-1.0, 0.0, 0.75};
    std::vector<double> reference_rows;
    std::vector<double> reference_grid;
    for (std::size_t specification_index = 0;
         specification_index < specifications.size();
         ++specification_index) {
        const scar::CopulaSpec& spec = specifications[specification_index];
        const auto serial_rows = scar::multivariate_log_pdf_and_grad(
            spec, observations, df, 0, 1);
        const auto parallel_rows = scar::multivariate_log_pdf_and_grad(
            spec, observations, df, 0, 4);
        const auto serial_grid = scar::multivariate_pdf_and_grad_grid(
            spec, observations, state_grid, 0, 1);
        const auto parallel_grid = scar::multivariate_pdf_and_grad_grid(
            spec, observations, state_grid, 0, 4);
        const int diagnostic_base =
            30 + static_cast<int>(specification_index) * 10;
        if (!serial_rows.is_ok()) {
            return diagnostic_base;
        }
        if (!parallel_rows.is_ok()) {
            return diagnostic_base + 1;
        }
        if (!serial_grid.is_ok()) {
            return diagnostic_base + 2;
        }
        if (!parallel_grid.is_ok()) {
            return diagnostic_base + 3;
        }
        if (!close_vectors(serial_rows.log_pdf, parallel_rows.log_pdf)) {
            return diagnostic_base + 4;
        }
        if (!close_vectors(serial_rows.dlog_dr, parallel_rows.dlog_dr)) {
            return diagnostic_base + 5;
        }
        if (!close_vectors(
                serial_grid.pdf.values, parallel_grid.pdf.values)) {
            return diagnostic_base + 6;
        }
        if (!close_vectors(
                serial_grid.d_pdf_dx.values,
                parallel_grid.d_pdf_dx.values)) {
            return diagnostic_base + 7;
        }
        scar::StaticCopulaEvaluator serial_objective(spec, observations, 1);
        scar::StaticCopulaEvaluator parallel_objective(spec, observations, 4);
        const bool correlation_gradient =
            spec.correlation_kind != scar::CorrelationKind::Factor;
        const auto objective_one = serial_objective.objective(
            5.0, correlation_gradient);
        const auto objective_four = parallel_objective.objective(
            5.0, correlation_gradient);
        if (!objective_one.is_ok() || !objective_four.is_ok()
            || !close(
                objective_one.negative_log_likelihood,
                objective_four.negative_log_likelihood)
            || !close(
                objective_one.negative_gradient,
                objective_four.negative_gradient)
            || (correlation_gradient
                && objective_one.negative_correlation_gradient.size() != 3)) {
            return 18;
        }
        if (reference_rows.empty()) {
            reference_rows = serial_rows.log_pdf;
            reference_grid = serial_grid.pdf.values;
        } else if (!close_vectors(reference_rows, serial_rows.log_pdf)
                   || !close_vectors(reference_grid, serial_grid.pdf.values)) {
            return 19;
        }
    }

    // Factor-specific rows/grid, df score, loading score and penalized joint
    // objective use a non-trivial operator. Gradients are checked by finite
    // differences of the concrete C++ objective.
    const std::vector<double> loadings{0.20, -0.25, 0.15};
    const scar::FactorCorrelationOperator factor(
        loadings, kDimension, 1, kUniquenessMin);
    const auto factor_rows_one = scar::factor_student_log_pdf_and_dlog_ddf(
        factor, flat_observations.data(), observations.size(),
        df.data(), df.size(), 1);
    const auto factor_rows_four = scar::factor_student_log_pdf_and_dlog_ddf(
        factor, flat_observations.data(), observations.size(),
        df.data(), df.size(), 4);
    const std::vector<double> df_grid{3.0, 5.0, 12.0};
    const auto factor_grid = scar::factor_student_log_pdf_and_dlog_ddf_grid(
        factor, flat_observations.data(), observations.size(),
        df_grid.data(), df_grid.size(), 2, 4);
    const auto factor_density = scar::factor_student_density_from_log_grid(
        factor_grid.log_pdf.data(), factor_grid.dlog_ddf.data(),
        factor_grid.log_pdf.size());
    const std::vector<double> factor_raw_grid{-1.0, 0.0, 0.75};
    const auto factor_stochastic =
        scar::factor_student_stochastic_pdf_and_grad_grid(
            factor, flat_observations.data(), observations.size(),
            factor_raw_grid.data(), factor_raw_grid.size(), 2.01, 2, 4);
    if (!factor_rows_one.is_ok()) {
        return 70;
    }
    if (!factor_rows_four.is_ok()) {
        return 71;
    }
    if (!factor_grid.is_ok()) {
        return 72;
    }
    if (!factor_density.is_ok()) {
        return 73;
    }
    if (!close_vectors(
            factor_rows_one.log_pdf, factor_rows_four.log_pdf)) {
        return 74;
    }
    if (!close_vectors(
            factor_rows_one.dlog_ddf, factor_rows_four.dlog_ddf)) {
        return 75;
    }
    if (!factor_stochastic.is_ok()
        || factor_stochastic.pdf.size() != factor_grid.log_pdf.size()
        || !finite_values(factor_stochastic.pdf)
        || !finite_values(factor_stochastic.d_pdf_dx)) {
        return 76;
    }
    for (std::size_t cell = 0; cell < factor_grid.log_pdf.size(); ++cell) {
        if (!close(
                factor_density.pdf[cell],
                std::exp(factor_grid.log_pdf[cell]),
                2e-14)
            || !close(
                factor_density.d_pdf_ddf[cell],
                factor_density.pdf[cell] * factor_grid.dlog_ddf[cell],
                2e-14)) {
            return 77;
        }
    }
    if (!finite_values(factor_grid.log_pdf)
        || !finite_values(factor_grid.dlog_ddf)) {
        return 77;
    }
    const auto joint = scar::factor_student_joint_likelihood_gradient(
        factor, flat_observations.data(), observations.size(), 5.0, 4);
    if (!joint.is_ok()
        || joint.dlog_likelihood_dloadings.size() != loadings.size()
        || !finite_values(joint.dlog_likelihood_dloadings)) {
        return 21;
    }
    for (std::size_t index = 0; index < loadings.size(); ++index) {
        std::vector<double> plus = loadings;
        std::vector<double> minus = loadings;
        plus[index] += 1e-6;
        minus[index] -= 1e-6;
        const scar::FactorCorrelationOperator plus_factor(
            plus, kDimension, 1, kUniquenessMin);
        const scar::FactorCorrelationOperator minus_factor(
            minus, kDimension, 1, kUniquenessMin);
        const double loading_difference = (
            scar::factor_student_joint_likelihood_gradient(
                plus_factor, flat_observations.data(), observations.size(),
                5.0, 1).log_likelihood
            - scar::factor_student_joint_likelihood_gradient(
                minus_factor, flat_observations.data(), observations.size(),
                5.0, 1).log_likelihood)
            / 2e-6;
        if (!close(
                loading_difference,
                joint.dlog_likelihood_dloadings[index],
                3e-6)) {
            return 22;
        }
    }
    const double df_difference = (
        scar::factor_student_joint_likelihood_gradient(
            factor, flat_observations.data(), observations.size(),
            5.0001, 1).log_likelihood
        - scar::factor_student_joint_likelihood_gradient(
            factor, flat_observations.data(), observations.size(),
            4.9999, 1).log_likelihood)
        / 0.0002;
    if (!close(df_difference, joint.dlog_likelihood_ddf, 3e-6)) {
        return 23;
    }

    const auto parameterization = scar::factor_parameterization_from_loadings(
        view(loadings), kDimension, 1, kUniquenessMin);
    const std::vector<double> free_rows = as_double(parameterization.free_rows);
    const std::vector<double> free_columns = as_double(
        parameterization.free_columns);
    const std::vector<double> diagonal_entries = as_double(
        parameterization.diagonal_entries);
    const auto penalized =
        scar::factor_student_penalized_parameterized_objective_gradient(
            flat_observations.data(), observations.size(), 5.0,
            parameterization.parameters.data(),
            parameterization.parameters.size(),
            free_rows.data(), free_columns.data(), diagonal_entries.data(),
            kDimension, 1, parameterization.max_norm, kUniquenessMin,
            1e6, 0.01, 4);
    if (!parameterization.is_ok() || !penalized.is_ok()
        || penalized.gradient.size()
            != parameterization.parameters.size() + 1
        || !finite_values(penalized.gradient)
        || !std::isfinite(penalized.objective)) {
        return 24;
    }

    // Dense/factor residuals and fixed-draw unconditional/conditional samples
    // agree under the zero-loading representation of identity correlation.
    const scar::ObservationView observation_view{
        flat_observations.data(), observations.size(),
        static_cast<int>(kDimension)};
    const auto dense_residuals = scar::student_rosenblatt_dense(
        view(identity), static_cast<int>(kDimension), observation_view,
        view(df), 4);
    const auto factor_residuals = scar::student_rosenblatt_factor(
        *zero_factor, observation_view, view(df), 4);
    if (!dense_residuals.is_ok() || !factor_residuals.is_ok()
        || !close_vectors(
            dense_residuals.residuals,
            factor_residuals.residuals,
            3e-12)) {
        return 25;
    }
    const auto radial = scar::radial_uniform_summary(
        {dense_residuals.residuals.data(), observations.size(),
         static_cast<int>(kDimension)},
        4);
    if (!radial.is_ok() || !finite_values(radial.values)) {
        return 26;
    }

    const std::vector<double> normal_draws{
        0.0, 1.0, -1.0,
        0.5, -0.5, 0.25,
    };
    const std::vector<double> sample_df{5.0, 7.0};
    const std::vector<double> chi_square{5.0, 7.0};
    const std::vector<double> factor_draws{0.75, -0.25};
    const auto dense_sample = scar::multivariate_student_sample_dense(
        view(identity), static_cast<int>(kDimension), view(sample_df),
        view(normal_draws), view(chi_square), 2, 4);
    const auto factor_sample = scar::multivariate_student_sample_factor(
        *zero_factor, view(sample_df), view(factor_draws),
        view(normal_draws), view(chi_square), 2, 4);
    if (!dense_sample.is_ok() || !factor_sample.is_ok()
        || !close_vectors(dense_sample.values, factor_sample.values, 2e-14)) {
        return 27;
    }
    const std::vector<int> given_indices{0};
    const std::vector<double> given_uniforms{0.25, 0.75};
    const std::vector<double> conditional_normals{-0.3, 0.8, 0.4, -0.6};
    const std::vector<double> full_residuals{
        0.0, -0.3, 0.8,
        0.0, 0.4, -0.6,
    };
    const std::vector<double> conditional_chi_square{6.0, 8.0};
    const auto dense_conditional =
        scar::multivariate_student_conditional_from_uniforms(
            view(identity), 1, static_cast<int>(kDimension), given_indices,
            view(given_uniforms), view(sample_df), view(conditional_normals),
            view(conditional_chi_square), 2, 4);
    const auto factor_conditional = scar::multivariate_student_conditional_factor(
        *zero_factor, given_indices, view(given_uniforms), view(sample_df),
        view(factor_draws), view(full_residuals),
        view(conditional_chi_square), 2, 4);
    if (!dense_conditional.is_ok() || !factor_conditional.is_ok()
        || dense_conditional.n_free != 2 || factor_conditional.n_free != 2
        || !close_vectors(
            dense_conditional.values, factor_conditional.values, 3e-12)) {
        return 28;
    }

    // Typed failures cover invalid df, non-finite observations, cache slicing,
    // and fixed-draw parameter validation. Throwing low-level factor APIs are
    // frozen as exceptions at their direct C++ boundary.
    const std::vector<double> invalid_df{2.0};
    const auto invalid_rows = scar::multivariate_log_pdf_and_grad(
        specifications.front(), observations, invalid_df, 0, 1);
    const auto bad_slice = student::evaluate_ppf_table(
        view(flat_observations), view(ppf_table.value.nodes),
        view(ppf_table.value.table), 5.0,
        flat_observations.size(), 1);
    std::vector<double> nonfinite_observations = flat_observations;
    nonfinite_observations[4] = std::numeric_limits<double>::quiet_NaN();
    const auto nonfinite_joint = scar::factor_student_joint_likelihood_gradient(
        factor, nonfinite_observations.data(), observations.size(), 5.0, 1);
    const std::vector<double> bad_chi_square{5.0, 0.0};
    const auto invalid_sample = scar::multivariate_student_sample_dense(
        view(identity), static_cast<int>(kDimension), view(sample_df),
        view(normal_draws), view(bad_chi_square), 2, 1);
    bool invalid_factor_df_threw = false;
    try {
        static_cast<void>(scar::factor_student_log_pdf_and_dlog_ddf(
            factor, flat_observations.data(), observations.size(),
            invalid_df.data(), invalid_df.size(), 1));
    } catch (const std::invalid_argument&) {
        invalid_factor_df_threw = true;
    }
    if (invalid_rows.status != scar::Status::NumericalFailure
        || invalid_rows.failure.index != 0
        || bad_slice.status != scar::Status::InvalidSize
        || nonfinite_joint.status != scar::Status::NumericalFailure
        || nonfinite_joint.failure.index != 1
        || invalid_sample.status != scar::Status::InvalidParameter
        || !invalid_factor_df_threw) {
        return 29;
    }

    return 0;
}
