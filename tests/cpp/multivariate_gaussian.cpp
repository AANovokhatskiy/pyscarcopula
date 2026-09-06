#include "scar/copula.hpp"
#include "scar/copula/capability.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/factor_parameterization.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/gaussian/density.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/multivariate/sampling.hpp"
#include "scar/math/normal.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDimension = 3;
constexpr double kUniquenessMin = 1e-6;

scar::DoubleView view(const std::vector<double>& values) {
    return {values.data(), values.size()};
}

bool close(double first, double second, double tolerance = 2e-12) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

bool close_vectors(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double tolerance = 2e-12) {

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
    spec.family = scar::CopulaFamily::MultivariateGaussian;
    spec.dim = static_cast<int>(kDimension);
    spec.correlation_kind = kind;
    spec.dense_inverse_cholesky() = prepared.inverse_cholesky;
    spec.dense_log_determinant() = prepared.log_determinant;
    return spec;
}

scar::CopulaSpec factor_spec(
    const std::shared_ptr<const scar::FactorCorrelationOperator>& factor) {

    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::MultivariateGaussian;
    spec.dim = static_cast<int>(kDimension);
    spec.correlation_kind = scar::CorrelationKind::Factor;
    spec.factor_operator() = factor;
    spec.dense_log_determinant() = factor->logdet();
    return spec;
}

std::vector<int> free_coordinates(const std::vector<int>& given) {
    std::vector<int> result;
    for (int coordinate = 0;
         coordinate < static_cast<int>(kDimension);
         ++coordinate) {
        bool is_given = false;
        for (int value : given) {
            is_given = is_given || value == coordinate;
        }
        if (!is_given) {
            result.push_back(coordinate);
        }
    }
    return result;
}

}  // namespace

int run_multivariate_gaussian_tests() {
    const std::vector<double> identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::vector<double> correlation{
        1.0, 0.25, -0.10,
        0.25, 1.0, 0.20,
        -0.10, 0.20, 1.0,
    };

    // correlation: validation, projection, dense preparation and both raw
    // parameterizations are direct native contracts.
    const auto valid = scar::validate_correlation(
        view(correlation), kDimension, 1e-12);
    const auto preprocessed = scar::preprocess_correlation(
        view(correlation), kDimension, 1e-8);
    const auto prepared = scar::prepare_dense_correlation(
        view(correlation), kDimension);
    if (!valid.is_ok() || !valid.value
        || !preprocessed.is_ok() || preprocessed.projection_applied
        || !close_vectors(preprocessed.correlation, correlation, 2e-12)
        || !prepared.is_ok()
        || prepared.inverse_cholesky.size() != identity.size()
        || !std::isfinite(prepared.log_determinant)) {
        return 1;
    }
    const std::vector<double> indefinite{
        1.0, 1.2, 0.0,
        1.2, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const auto invalid = scar::validate_correlation(
        view(indefinite), kDimension, 1e-12);
    const auto projected = scar::preprocess_correlation(
        view(indefinite), kDimension, 1e-6);
    if (invalid.is_ok() || invalid.failure.coordinate < 0
        || !projected.is_ok() || !projected.projection_applied
        || projected.min_eigenvalue_before >= 0.0
        || projected.min_eigenvalue_after <= 0.0) {
        return 2;
    }

    const auto packed = scar::pack_cholesky_correlation(
        view(correlation), kDimension, 1e-10);
    if (!packed.is_ok() || packed.value.size() != 3) {
        return 3;
    }
    const auto unpacked = scar::unpack_cholesky_correlation(
        view(packed.value), kDimension);
    if (!unpacked.is_ok()
        || !close_vectors(unpacked.value, correlation, 3e-12)) {
        return 4;
    }
    const std::vector<double> correlation_score{0.4, -0.25, 0.7};
    const auto cholesky_pullback = scar::correlation_gradient_to_raw(
        scar::DenseCorrelationMode::Cholesky,
        view(packed.value),
        view(unpacked.value),
        view(correlation_score),
        {},
        kDimension);
    if (!cholesky_pullback.is_ok()
        || cholesky_pullback.value.size() != packed.value.size()
        || !finite_values(cholesky_pullback.value)) {
        return 5;
    }
    for (std::size_t parameter = 0;
         parameter < packed.value.size();
         ++parameter) {
        std::vector<double> plus = packed.value;
        std::vector<double> minus = packed.value;
        plus[parameter] += 1e-6;
        minus[parameter] -= 1e-6;
        const auto plus_correlation = scar::unpack_cholesky_correlation(
            view(plus), kDimension);
        const auto minus_correlation = scar::unpack_cholesky_correlation(
            view(minus), kDimension);
        if (!plus_correlation.is_ok() || !minus_correlation.is_ok()) {
            return 6;
        }
        double finite_difference = 0.0;
        std::size_t score_index = 0;
        for (std::size_t row = 1; row < kDimension; ++row) {
            for (std::size_t column = 0; column < row; ++column) {
                const std::size_t index = row * kDimension + column;
                finite_difference += correlation_score[score_index++]
                    * (plus_correlation.value[index]
                       - minus_correlation.value[index])
                    / 2e-6;
            }
        }
        if (!close(
                finite_difference,
                cholesky_pullback.value[parameter],
                2e-8)) {
            return 7;
        }
    }

    const std::vector<double> shrinkage_raw{0.0};
    const auto shrunk = scar::make_shrinkage_correlation(
        shrinkage_raw[0], view(correlation), kDimension);
    const auto shrinkage_direction =
        scar::shrinkage_raw_correlation_direction(
            view(shrinkage_raw), view(correlation), kDimension);
    const auto shrinkage_pullback = scar::correlation_gradient_to_raw(
        scar::DenseCorrelationMode::Shrinkage,
        view(shrinkage_raw),
        view(shrunk.value),
        view(correlation_score),
        view(correlation),
        kDimension);
    double direction_product = 0.0;
    if (shrinkage_direction.is_ok()) {
        for (std::size_t index = 0;
             index < correlation_score.size();
             ++index) {
            direction_product += correlation_score[index]
                * shrinkage_direction.value[index];
        }
    }
    if (!shrunk.is_ok() || !shrinkage_direction.is_ok()
        || !shrinkage_pullback.is_ok()
        || shrinkage_pullback.value.size() != 1
        || !close(shrinkage_pullback.value[0], direction_product, 2e-14)) {
        return 8;
    }

    // Factor reconstruction and pullback use the factor package directly.
    const std::vector<double> loadings{0.20, -0.30, 0.10};
    const auto factor_transform = scar::factor_correlation_from_loadings(
        view(loadings), kDimension, 1, kUniquenessMin);
    const auto factor_dense = scar::factor_correlation_to_dense(
        view(factor_transform.loadings),
        view(factor_transform.uniqueness),
        kDimension,
        1);
    const auto factor_parameters = scar::factor_parameterization_from_loadings(
        view(loadings), kDimension, 1, kUniquenessMin);
    if (!factor_transform.is_ok() || !factor_dense.is_ok()
        || !factor_parameters.is_ok()
        || factor_parameters.parameters.size() != kDimension) {
        return 9;
    }
    const std::vector<double> free_rows = as_double(
        factor_parameters.free_rows);
    const std::vector<double> free_columns = as_double(
        factor_parameters.free_columns);
    const std::vector<double> diagonal_entries = as_double(
        factor_parameters.diagonal_entries);
    const auto reconstructed_loadings = scar::factor_parameterization_loadings(
        view(factor_parameters.parameters),
        view(free_rows),
        view(free_columns),
        view(diagonal_entries),
        kDimension,
        1,
        factor_parameters.max_norm);
    const auto reconstructed_factor = scar::factor_correlation_from_loadings(
        view(reconstructed_loadings.value),
        kDimension,
        1,
        kUniquenessMin);
    const auto reconstructed_dense = scar::factor_correlation_to_dense(
        view(reconstructed_factor.loadings),
        view(reconstructed_factor.uniqueness),
        kDimension,
        1);
    if (!reconstructed_loadings.is_ok() || !reconstructed_factor.is_ok()
        || !reconstructed_dense.is_ok()
        || !close_vectors(reconstructed_dense.value, factor_dense.value, 2e-12)) {
        return 10;
    }
    const std::vector<double> loading_score{0.7, -0.4, 0.2};
    const auto factor_pullback = scar::factor_parameterization_pullback(
        view(factor_parameters.parameters),
        view(loading_score),
        view(free_rows),
        view(free_columns),
        view(diagonal_entries),
        kDimension,
        1,
        factor_parameters.max_norm);
    if (!factor_pullback.is_ok()
        || factor_pullback.value.size()
            != factor_parameters.parameters.size()
        || !finite_values(factor_pullback.value)) {
        return 11;
    }
    for (std::size_t parameter = 0;
         parameter < factor_parameters.parameters.size();
         ++parameter) {
        std::vector<double> plus = factor_parameters.parameters;
        std::vector<double> minus = factor_parameters.parameters;
        plus[parameter] += 1e-6;
        minus[parameter] -= 1e-6;
        const auto plus_loadings = scar::factor_parameterization_loadings(
            view(plus), view(free_rows), view(free_columns),
            view(diagonal_entries), kDimension, 1,
            factor_parameters.max_norm);
        const auto minus_loadings = scar::factor_parameterization_loadings(
            view(minus), view(free_rows), view(free_columns),
            view(diagonal_entries), kDimension, 1,
            factor_parameters.max_norm);
        if (!plus_loadings.is_ok() || !minus_loadings.is_ok()) {
            return 12;
        }
        double finite_difference = 0.0;
        for (std::size_t index = 0; index < loading_score.size(); ++index) {
            finite_difference += loading_score[index]
                * (plus_loadings.value[index] - minus_loadings.value[index])
                / 2e-6;
        }
        if (!close(
                finite_difference,
                factor_pullback.value[parameter],
                2e-8)) {
            return 13;
        }
    }

    // The capability matrix is exact: all four concrete correlation modes
    // expose six static operation cells. The absent grid-gradient owner is
    // frozen as unsupported rather than advertised with a hidden fallback.
    constexpr std::array<scar::NativeOperation, 6> supported_operations{
        scar::NativeOperation::ParameterTransformBoundsInitialization,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::NativeOperation::RosenblattResidual,
        scar::NativeOperation::RadialGofSummary,
        scar::NativeOperation::UnconditionalSamplingTransform,
        scar::NativeOperation::ConditionalSamplingTransform,
    };
    constexpr std::array<scar::CorrelationKind, 4> modes{
        scar::CorrelationKind::Fixed,
        scar::CorrelationKind::Shrinkage,
        scar::CorrelationKind::DenseCholesky,
        scar::CorrelationKind::Factor,
    };
    for (scar::CorrelationKind mode : modes) {
        const auto descriptor = scar::make_typed_model_descriptor(
            scar::NativeModelId::Gaussian,
            static_cast<int>(kDimension),
            mode,
            0,
            scar::FactorEstimationKind::TwoStage);
        for (scar::NativeOperation operation : supported_operations) {
            const auto capability = scar::query_capability(
                descriptor, operation, scar::DynamicsKind::Mle);
            if (!capability.supported || !capability.reason.empty()) {
                return 14;
            }
        }
        const auto row_grid = scar::query_capability(
            descriptor,
            scar::NativeOperation::RowGridDensityGradient,
            scar::DynamicsKind::Mle);
        if (row_grid.supported
            || row_grid.reason.find("no row/grid gradient")
                == std::string::npos) {
            return 15;
        }
        for (scar::NativeOperation operation : {
                 scar::NativeOperation::PointDensityDerivatives,
                 scar::NativeOperation::StateFilterSmoother,
                 scar::NativeOperation::ArbitraryConditionalMcmc,
                 scar::NativeOperation::EdgeStructureSelectionScore}) {
            const auto capability = scar::query_capability(
                descriptor, operation, scar::DynamicsKind::Mle);
            if (capability.supported || capability.reason.empty()) {
                return 16;
            }
        }
    }
    const auto joint_factor = scar::make_typed_model_descriptor(
        scar::NativeModelId::Gaussian,
        static_cast<int>(kDimension),
        scar::CorrelationKind::Factor,
        0,
        scar::FactorEstimationKind::Joint);
    const auto joint_objective = scar::query_capability(
        joint_factor,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::DynamicsKind::Mle);
    if (joint_objective.supported
        || joint_objective.reason.find("factor-loading score")
            == std::string::npos) {
        return 17;
    }

    // Row log-density and the full objective/correlation-gradient contract are
    // checked for fixed, shrinkage, Cholesky, and two-stage factor modes.
    const scar::Observations observations{
        {0.20, 0.40, 0.60},
        {0.75, 0.35, 0.55},
        {0.10, 0.80, 0.45},
        {0.90, 0.25, 0.70},
    };
    std::vector<scar::CopulaSpec> specifications;
    specifications.push_back(dense_spec(
        correlation, scar::CorrelationKind::Fixed));
    specifications.push_back(dense_spec(
        shrunk.value, scar::CorrelationKind::Shrinkage));
    specifications.push_back(dense_spec(
        unpacked.value, scar::CorrelationKind::DenseCholesky));
    const auto factor = std::make_shared<scar::FactorCorrelationOperator>(
        factor_transform.loadings, kDimension, 1, kUniquenessMin);
    specifications.push_back(factor_spec(factor));
    for (const scar::CopulaSpec& spec : specifications) {
        scar::StaticCopulaEvaluator serial(spec, observations, 1);
        scar::StaticCopulaEvaluator parallel(spec, observations, 4);
        if (serial.status() != scar::Status::Ok
            || parallel.status() != scar::Status::Ok) {
            return 18;
        }
        const bool request_correlation_gradient =
            spec.correlation_kind != scar::CorrelationKind::Factor;
        const auto serial_objective = serial.gaussian_objective(
            spec, request_correlation_gradient);
        const auto parallel_objective = parallel.gaussian_objective(
            spec, request_correlation_gradient);
        const std::vector<double> serial_rows = serial.log_pdf_rows(0.0);
        const std::vector<double> parallel_rows = parallel.log_pdf_rows(0.0);
        if (!serial_objective.is_ok() || !parallel_objective.is_ok()
            || !close(
                serial_objective.negative_log_likelihood,
                parallel_objective.negative_log_likelihood,
                2e-13)
            || !close_vectors(serial_rows, parallel_rows, 2e-13)
            || !finite_values(serial_rows)) {
            return 19;
        }
        double row_sum = 0.0;
        for (double value : serial_rows) {
            row_sum += value;
        }
        if (!close(
                serial_objective.negative_log_likelihood,
                -row_sum,
                2e-13)
            || (request_correlation_gradient
                && (serial_objective.negative_correlation_gradient.size() != 3
                    || !close_vectors(
                        serial_objective.negative_correlation_gradient,
                        parallel_objective.negative_correlation_gradient,
                        2e-13)))
            || (!request_correlation_gradient
                && !serial_objective.negative_correlation_gradient.empty())) {
            return 20;
        }
    }
    const auto bad_factor_gradient =
        scar::StaticCopulaEvaluator(specifications.back(), observations, 1)
            .gaussian_objective(specifications.back(), true);
    if (bad_factor_gradient.status != scar::Status::InvalidParameter
        || std::isfinite(bad_factor_gradient.negative_log_likelihood)) {
        return 21;
    }

    const double scores[]{0.3, -0.7, 1.1};
    const scar::copula::multivariate::correlation::DenseCorrelation
        factor_as_dense{
            scar::prepare_dense_correlation(
                view(factor_dense.value), kDimension).inverse_cholesky,
            scar::prepare_dense_correlation(
                view(factor_dense.value), kDimension).log_determinant,
        };
    std::vector<double> projection;
    std::vector<double> solved;
    if (!close(
            scar::copula::multivariate::gaussian::log_pdf(
                *factor, scores, projection, solved),
            scar::copula::multivariate::gaussian::log_pdf(
                factor_as_dense,
                static_cast<int>(kDimension),
                scores),
            3e-13)) {
        return 22;
    }

    // Identity correlation makes fixed normal transcripts exact for both
    // dense and factor unconditional transforms.
    const std::vector<double> normal_draws{
        0.0, 1.0, -1.0,
        0.5, -0.5, 0.25,
    };
    const auto zero_factor = std::make_shared<scar::FactorCorrelationOperator>(
        std::vector<double>(kDimension, 0.0),
        kDimension,
        1,
        kUniquenessMin);
    const std::vector<double> factor_draws{0.75, -0.25};
    const auto dense_sample = scar::multivariate_gaussian_sample_dense(
        view(identity),
        static_cast<int>(kDimension),
        view(normal_draws),
        2,
        4);
    const auto factor_sample = scar::multivariate_gaussian_sample_factor(
        *zero_factor,
        view(factor_draws),
        view(normal_draws),
        2,
        4);
    if (!dense_sample.is_ok() || !factor_sample.is_ok()
        || dense_sample.correlation_factorizations != 1
        || dense_sample.values.size() != normal_draws.size()
        || factor_sample.values.size() != normal_draws.size()) {
        return 23;
    }
    for (std::size_t index = 0; index < normal_draws.size(); ++index) {
        const double expected = scar::math::normal_cdf(normal_draws[index]);
        if (!close(dense_sample.values[index], expected, 2e-15)
            || !close(factor_sample.values[index], expected, 2e-15)) {
            return 24;
        }
    }

    // All six non-empty/proper conditioning subsets are exercised. Shared
    // and row-specific given-value layouts alternate; dense and factor owners
    // receive the same exact transcript.
    const std::array<std::vector<int>, 6> conditioning_layouts{
        std::vector<int>{0},
        std::vector<int>{1},
        std::vector<int>{2},
        std::vector<int>{0, 1},
        std::vector<int>{0, 2},
        std::vector<int>{1, 2},
    };
    const std::vector<double> full_residuals{
        -0.80, 0.10, 0.90,
        0.35, -0.45, 0.65,
    };
    for (std::size_t layout = 0;
         layout < conditioning_layouts.size();
         ++layout) {
        const std::vector<int>& given = conditioning_layouts[layout];
        const std::vector<int> free = free_coordinates(given);
        std::vector<double> given_uniforms;
        if (layout % 2 == 0) {
            given_uniforms.assign(given.size(), 0.4);
        } else {
            given_uniforms.assign(2 * given.size(), 0.4);
            for (std::size_t index = given.size();
                 index < given_uniforms.size();
                 ++index) {
                given_uniforms[index] = 0.6;
            }
        }
        std::vector<double> dense_normals;
        std::vector<double> expected;
        for (std::size_t row = 0; row < 2; ++row) {
            for (int coordinate : free) {
                const double draw = full_residuals[
                    row * kDimension + static_cast<std::size_t>(coordinate)];
                dense_normals.push_back(draw);
                expected.push_back(scar::math::normal_cdf(draw));
            }
        }
        const auto dense_conditional =
            scar::multivariate_gaussian_conditional_from_uniforms(
                view(identity),
                1,
                static_cast<int>(kDimension),
                given,
                view(given_uniforms),
                view(dense_normals),
                2,
                4);
        const auto factor_conditional =
            scar::multivariate_gaussian_conditional_factor(
                *zero_factor,
                given,
                view(given_uniforms),
                view(factor_draws),
                view(full_residuals),
                2,
                4);
        if (!dense_conditional.is_ok() || !factor_conditional.is_ok()
            || dense_conditional.n_free
                != static_cast<std::int64_t>(free.size())
            || factor_conditional.n_free
                != static_cast<std::int64_t>(free.size())
            || dense_conditional.correlation_factorizations != 1
            || factor_conditional.correlation_factorizations != 1
            || !close_vectors(dense_conditional.values, expected, 2e-15)
            || !close_vectors(factor_conditional.values, expected, 2e-15)) {
            return 25;
        }
    }
    std::vector<double> row_correlations;
    row_correlations.insert(
        row_correlations.end(), identity.begin(), identity.end());
    row_correlations.insert(
        row_correlations.end(), identity.begin(), identity.end());
    const std::vector<int> first_given{0};
    const std::vector<double> row_given{0.2, 0.8};
    const std::vector<double> row_normals{-0.2, 0.3, 0.4, -0.6};
    const auto row_conditional =
        scar::multivariate_gaussian_conditional_from_uniforms(
            view(row_correlations),
            2,
            static_cast<int>(kDimension),
            first_given,
            view(row_given),
            view(row_normals),
            2,
            2);
    if (!row_conditional.is_ok()
        || row_conditional.correlation_factorizations != 2) {
        return 26;
    }
    for (std::size_t index = 0; index < row_normals.size(); ++index) {
        if (!close(
                row_conditional.values[index],
                scar::math::normal_cdf(row_normals[index]),
                2e-15)) {
            return 27;
        }
    }

    // Rosenblatt and radial reduction agree exactly under independence for
    // dense and factor representations.
    const std::vector<double> residual_input{
        0.20, 0.40, 0.60,
        0.75, 0.35, 0.55,
    };
    const scar::ObservationView residual_view{
        residual_input.data(), 2, static_cast<int>(kDimension)};
    const auto dense_residuals = scar::gaussian_rosenblatt_dense(
        view(identity), static_cast<int>(kDimension), residual_view, 4);
    const auto factor_residuals = scar::gaussian_rosenblatt_factor(
        *zero_factor, residual_view, 4);
    const auto radial = scar::radial_uniform_summary(
        {dense_residuals.residuals.data(), 2, static_cast<int>(kDimension)},
        4);
    if (!dense_residuals.is_ok() || !factor_residuals.is_ok()
        || !radial.is_ok() || radial.values.size() != 2
        || !close_vectors(
            dense_residuals.residuals, residual_input, 2e-15)
        || !close_vectors(
            factor_residuals.residuals, residual_input, 2e-15)
        || !finite_values(radial.values)) {
        return 28;
    }

    // Failure contracts: invalid threads/layouts, non-finite input, and a
    // non-positive-definite sampling matrix return typed status/context.
    const auto duplicate_given =
        scar::multivariate_gaussian_conditional_from_uniforms(
            view(identity), 1, static_cast<int>(kDimension), {0, 0},
            view(row_given), view(row_normals), 2, 1);
    const auto bad_threads = scar::multivariate_gaussian_sample_dense(
        view(identity), static_cast<int>(kDimension), view(normal_draws), 2, 0);
    std::vector<double> nonfinite_draws = normal_draws;
    nonfinite_draws[2] = std::numeric_limits<double>::quiet_NaN();
    const auto nonfinite_sample = scar::multivariate_gaussian_sample_dense(
        view(identity), static_cast<int>(kDimension),
        view(nonfinite_draws), 2, 1);
    const auto indefinite_sample = scar::multivariate_gaussian_sample_dense(
        view(indefinite), static_cast<int>(kDimension),
        view(normal_draws), 2, 1);
    if (duplicate_given.status != scar::Status::InvalidParameter
        || bad_threads.status != scar::Status::InvalidSize
        || nonfinite_sample.status != scar::Status::InvalidParameter
        || indefinite_sample.status != scar::Status::NumericalFailure
        || indefinite_sample.failure.index != 0) {
        return 29;
    }

    return 0;
}
