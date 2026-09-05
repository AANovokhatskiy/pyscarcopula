#include "scar/copula.hpp"
#include "scar/copula/capability.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/ou.hpp"
#include "scar/model_policy.hpp"
#include "scar/scar_ou/initialization.hpp"
#include "scar/scar_ou/parameterization.hpp"
#include "scar/scar_ou/policy.hpp"
#include "scar/scar_ou/quadrature.hpp"
#include "scar/scar_ou/sampling.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <numeric>
#include <vector>

namespace {

struct PairOuCase {
    scar::CopulaFamily family;
    scar::NativeModelId model_id;
    scar::Rotation rotation;
};

scar::DoubleView view(const std::vector<double>& values) {
    return {values.data(), values.size()};
}

bool close(double first, double second, double tolerance = 3e-8) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

bool finite_values(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool close_vectors(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double tolerance = 3e-8) {

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

bool normalized(const std::vector<double>& probability, double tolerance) {
    if (!finite_values(probability)) {
        return false;
    }
    const double sum = std::accumulate(
        probability.begin(), probability.end(), 0.0);
    return std::abs(sum - 1.0) <= tolerance;
}

bool normalized_rows(
    const std::vector<double>& probability,
    std::size_t rows,
    std::size_t columns,
    double tolerance) {

    if (probability.size() != rows * columns
        || !finite_values(probability)) {
        return false;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        const auto begin = probability.begin()
            + static_cast<std::ptrdiff_t>(row * columns);
        const double sum = std::accumulate(
            begin,
            begin + static_cast<std::ptrdiff_t>(columns),
            0.0);
        if (std::abs(sum - 1.0) > tolerance) {
            return false;
        }
    }
    return true;
}

scar::CopulaSpec pair_spec(const PairOuCase& test) {
    scar::CopulaSpec spec = scar::default_pair_copula_spec(test.family);
    spec.rotation = test.rotation;
    return spec;
}

scar::CopulaSpec equicorr_spec(int dimension) {
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::EquicorrGaussian;
    spec.dim = dimension;
    spec.transform = scar::Transform::GaussianTanh;
    spec.correlation_kind = scar::CorrelationKind::Equicorrelation;
    return spec;
}

scar::CopulaSpec dense_student_spec(
    const std::vector<double>& correlation,
    int dimension) {

    const auto prepared = scar::prepare_dense_correlation(
        view(correlation), dimension);
    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = dimension;
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0 + 1e-6;
    spec.correlation_kind = scar::CorrelationKind::DenseCholesky;
    spec.dense_inverse_cholesky() = prepared.inverse_cholesky;
    spec.dense_log_determinant() = prepared.log_determinant;
    return spec;
}

scar::CopulaSpec factor_student_spec(
    const std::shared_ptr<const scar::FactorCorrelationOperator>& factor) {

    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Student;
    spec.dim = static_cast<int>(factor->dimension());
    spec.transform = scar::Transform::Softplus;
    spec.offset = 2.0 + 1e-6;
    spec.correlation_kind = scar::CorrelationKind::Factor;
    spec.factor_operator() = factor;
    spec.dense_log_determinant() = factor->logdet();
    return spec;
}

bool emission_blocks_match_rows(
    const scar::CopulaSpec& spec,
    const std::vector<double>& observations) {

    scar::PreparedDynamicEmission emission(spec);
    std::vector<double> parameters, derivatives;
    // The last state is outside the Student cache and exercises exact fallback.
    for (const std::vector<double>& states : {
            std::vector<double>{-2.0, 0.2, 3.0},
            std::vector<double>{-2.0, 0.2, 1100.0}}) {
        if (spec.family != scar::CopulaFamily::Student
            && states.back() > 100.0) {
            continue;
        }
        emission.prepare_grid_transform(states, parameters, derivatives);
        const auto K = parameters.size();
        std::vector<double> density, gradient, scales;
        std::vector<double> row_density(K), row_gradient(K);
        for (int threads : {1, 4}) {
            if (!emission.fill_density_and_gradient_block(
                    observations.data(), 1, 3, parameters, derivatives,
                    density, gradient, scales, threads)
                || density.size() != 3 * K || scales.size() != 3) {
                return false;
            }
            for (std::size_t row = 0; row < 3; ++row) {
                double scale = 0.0;
                emission.fill_density_and_gradient_row(
                    observations.data(), static_cast<std::int64_t>(row + 1),
                    parameters, derivatives, row_density.data(),
                    row_gradient.data(), &scale);
                if (!close(scale, scales[row], 2e-10)) {
                    return false;
                }
                for (std::size_t j = 0; j < K; ++j) {
                    if (!close(density[row * K + j], row_density[j], 2e-10)
                        || !close(gradient[row * K + j], row_gradient[j], 2e-10)) {
                        return false;
                    }
                }
            }
        }
        if (emission.fill_density_and_gradient_block(
                observations.data(), -1, 3, parameters, derivatives,
                density, gradient, scales, 1)
            || emission.fill_density_and_gradient_block(
                observations.data(), 1, std::numeric_limits<std::int64_t>::max(),
                parameters, derivatives, density, gradient, scales, 1)) {
            return false;
        }
    }
    return true;
}

scar::OuNumericalConfig matrix_config(scar::OuGridMethod method) {
    scar::OuNumericalConfig config;
    config.K = 17;
    config.grid_range = 3.5;
    config.adaptive = false;
    config.max_K = 17;
    config.gh_order = 5;
    config.spectral_basis_order = 12;
    config.spectral_quad_order = 32;
    config.grid_method = method;
    config.n_threads = 2;
    return config;
}

double finite_difference(
    const scar::ScarOuEvaluator& evaluator,
    const scar::OuParams& params,
    const scar::CopulaSpec& spec,
    scar::ObservationView observations,
    const scar::OuNumericalConfig& config,
    std::size_t coordinate,
    double step) {

    scar::OuParams plus = params;
    scar::OuParams minus = params;
    double* plus_values[] = {&plus.kappa, &plus.mu, &plus.nu};
    double* minus_values[] = {&minus.kappa, &minus.mu, &minus.nu};
    *plus_values[coordinate] += step;
    *minus_values[coordinate] -= step;
    const auto plus_result = evaluator.loglik_matrix(
        plus, spec, observations, config);
    const auto minus_result = evaluator.loglik_matrix(
        minus, spec, observations, config);
    if (!plus_result.is_ok() || !minus_result.is_ok()) {
        return std::nan("");
    }
    return -(plus_result.log_likelihood - minus_result.log_likelihood)
        / (2.0 * step);
}

}  // namespace

int run_ou_model_tests() {
    // Native ownership of optimizer/physical coordinates and initialization.
    const std::vector<double> physical{2.0, 0.3, 4.0, 0.25};
    const auto optimizer = scar::ou_to_log_stationary(physical);
    const auto recovered = scar::ou_from_log_stationary(optimizer.value);
    const std::vector<double> physical_gradient{1.0, 0.5, 3.0, 0.75};
    const auto optimizer_gradient = scar::ou_gradient_to_log_stationary(
        physical, physical_gradient);
    const auto recovered_gradient = scar::ou_gradient_from_log_stationary(
        physical, optimizer_gradient.value);
    const auto projection = scar::ou_project_optimizer_block(
        {-9.0, 30.0, 9.0, 0.25},
        {-3.0, -2.0, -4.0},
        {3.0, 2.0, 4.0});
    const auto default_initial = scar::ou_default_initial_point(0.2);
    const auto heuristic = scar::ou_heuristic_initial_point(
        21, 0.4, 0.95, 0.3);
    const auto student_initial = scar::ou_stochastic_student_initial_point(
        21, 5.0, 0.4, 2.0);
    if (!optimizer.is_ok() || !recovered.is_ok()
        || !optimizer_gradient.is_ok() || !recovered_gradient.is_ok()
        || !projection.is_ok() || !default_initial.is_ok()
        || !heuristic.is_ok() || !student_initial.is_ok()
        || !close_vectors(recovered.value, physical, 1e-14)
        || !close_vectors(
            recovered_gradient.value, physical_gradient, 1e-14)
        || projection.value != std::vector<double>{-3.0, 2.0, 4.0, 0.25}
        || default_initial.value.params.mu != 0.2
        || heuristic.value.params.mu != 0.4
        || student_initial.value.params.mu != 0.4) {
        return 1;
    }

    // Exact OU evolution and fixed-draw chunk continuation.
    const scar::OuParams trajectory_params{1.4, 0.2, 0.9};
    const std::vector<double> normals{0.5, -1.0, 0.25, 0.75, -0.4};
    const auto sampled = scar::sample_ou_trajectory(
        trajectory_params, view(normals));
    const double rho = std::exp(-trajectory_params.kappa / 4.0);
    const double sigma = std::sqrt(
        trajectory_params.nu * trajectory_params.nu
        / (2.0 * trajectory_params.kappa)
        * (1.0 - rho * rho));
    const double x0 = trajectory_params.mu
        + trajectory_params.nu / std::sqrt(2.0 * trajectory_params.kappa)
        * normals[0];
    const auto full_continuation = scar::ou_trajectory_from_innovations(
        x0,
        trajectory_params.mu,
        rho,
        sigma,
        {normals.data() + 1, normals.size() - 1});
    const auto first_chunk = scar::ou_trajectory_from_innovations(
        x0, trajectory_params.mu, rho, sigma, {normals.data() + 1, 2});
    const auto second_chunk = scar::ou_trajectory_from_innovations(
        first_chunk.values.back(),
        trajectory_params.mu,
        rho,
        sigma,
        {normals.data() + 3, 2});
    std::vector<double> chunked = first_chunk.values;
    chunked.insert(
        chunked.end(),
        second_chunk.values.begin() + 1,
        second_chunk.values.end());
    const auto first_block = scar::sample_ou_trajectory_block(
        trajectory_params, normals.size(), 0.0, true, {normals.data(), 3});
    if (!first_block.is_ok() || first_block.values.size() != 3) return 2;
    const auto second_block = scar::sample_ou_trajectory_block(
        trajectory_params, normals.size(), first_block.values.back(), false,
        {normals.data() + 3, 2});
    std::vector<double> blocked = first_block.values;
    blocked.insert(
        blocked.end(), second_block.values.begin(), second_block.values.end());
    if (!sampled.is_ok() || !full_continuation.is_ok()
        || !first_chunk.is_ok() || !second_chunk.is_ok()
        || !second_block.is_ok()
        || sampled.values.size() != normals.size()
        || blocked != sampled.values
        || !close_vectors(sampled.values, full_continuation.values, 1e-14)
        || !close_vectors(chunked, full_continuation.values, 1e-14)
        || scar::validate_ou_trajectory_parameters(
            scar::OuParams{-1.0, 0.0, 1.0}, normals.size())
            != scar::Status::InvalidParameter) {
        return 2;
    }
    if (scar::sample_ou_trajectory_block(
            trajectory_params, 2, 0.0, true, view(normals)).status
            != scar::Status::InvalidSize
        || scar::sample_ou_trajectory_block(
            trajectory_params, normals.size(),
            std::numeric_limits<double>::quiet_NaN(), false,
            view(normals)).status != scar::Status::InvalidParameter
        || scar::sample_ou_trajectory_block(
            trajectory_params, normals.size(), 0.0, true,
            {nullptr, 1}).status != scar::Status::InvalidSize) {
        return 2;
    }

    const std::vector<double> state_grid{-1.0, 0.0, 2.0};
    const std::vector<double> state_probability{0.2, 0.3, 0.5};
    const std::vector<double> selection{0.1, 0.4, 0.7, 0.95};
    const std::vector<double> jitter{0.25, 0.5, 0.75, 0.1};
    const auto full_state_sample = scar::sample_ou_state_distribution(
        view(state_grid), view(state_probability), view(selection), view(jitter),
        scar::OuStateSamplingMode::Histogram);
    const auto state_first = scar::sample_ou_state_distribution(
        view(state_grid),
        view(state_probability),
        {selection.data(), 2},
        {jitter.data(), 2},
        scar::OuStateSamplingMode::Histogram);
    const auto state_second = scar::sample_ou_state_distribution(
        view(state_grid),
        view(state_probability),
        {selection.data() + 2, 2},
        {jitter.data() + 2, 2},
        scar::OuStateSamplingMode::Histogram);
    std::vector<double> chunked_state = state_first.value.values;
    chunked_state.insert(
        chunked_state.end(),
        state_second.value.values.begin(),
        state_second.value.values.end());
    if (!full_state_sample.is_ok() || !state_first.is_ok()
        || !state_second.is_ok()
        || full_state_sample.value.selection_draws_used != 4
        || full_state_sample.value.jitter_draws_used != 4
        || state_first.value.selection_draws_used != 2
        || state_second.value.selection_draws_used != 2
        || !close_vectors(
            chunked_state, full_state_sample.value.values, 1e-15)) {
        return 3;
    }

    // Hermite/spectral policy and matrix/local/sparse transition contracts.
    const auto hermite = scar::ou_hermite_rule(5, 4);
    const auto default_order = scar::ou_default_quad_order(32);
    const auto small_backend = scar::ou_auto_backend(0.01, 11, 0.01);
    const auto regular_backend = scar::ou_auto_backend(1.0, 11, 0.01);
    const auto adaptive_order = scar::ou_adaptive_spectral_basis_order(
        0.01, 11);
    if (!hermite.is_ok() || hermite.value.nodes.size() != 5
        || hermite.value.weights.size() != 5
        || hermite.value.basis.size() != 20
        || !finite_values(hermite.value.nodes)
        || !finite_values(hermite.value.weights)
        || !finite_values(hermite.value.basis)
        || !default_order.is_ok() || default_order.value != 80
        || !small_backend.is_ok()
        || small_backend.value != scar::OuBackend::LocalGh
        || !regular_backend.is_ok()
        || regular_backend.value != scar::OuBackend::Spectral
        || !adaptive_order.is_ok() || adaptive_order.value != 128) {
        return 4;
    }

    std::vector<double> emissions(5 * 17, 1.0);
    const scar::OuNumericalConfig dense_config = matrix_config(
        scar::OuGridMethod::Dense);
    const scar::OuNumericalConfig sparse_config = matrix_config(
        scar::OuGridMethod::Sparse);
    const auto dense_filter = scar::filter_ou_grid_emissions(
        trajectory_params,
        view(emissions),
        5,
        17,
        dense_config,
        scar::OuBackend::Matrix);
    const auto sparse_filter = scar::filter_ou_grid_emissions(
        trajectory_params,
        view(emissions),
        5,
        17,
        sparse_config,
        scar::OuBackend::Matrix);
    const auto local_filter = scar::filter_ou_grid_emissions(
        trajectory_params,
        view(emissions),
        5,
        17,
        sparse_config,
        scar::OuBackend::LocalGh);
    if (!dense_filter.is_ok() || !sparse_filter.is_ok()
        || !local_filter.is_ok() || dense_filter.sparse
        || !sparse_filter.sparse
        || dense_filter.backend != scar::OuBackend::Matrix
        || local_filter.backend != scar::OuBackend::LocalGh
        || dense_filter.smoothed_weights.size() != emissions.size()
        || sparse_filter.smoothed_weights.size() != emissions.size()
        || local_filter.smoothed_weights.size() != emissions.size()
        || !normalized_rows(dense_filter.smoothed_weights, 5, 17, 2e-14)
        || !normalized_rows(sparse_filter.smoothed_weights, 5, 17, 2e-14)
        || !normalized_rows(local_filter.smoothed_weights, 5, 17, 2e-14)) {
        return 5;
    }

    constexpr std::array<PairOuCase, 14> pair_cases{{
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R0},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R90},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R180},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R270},
        {scar::CopulaFamily::Frank,
         scar::NativeModelId::Frank, scar::Rotation::R0},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R0},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R90},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R180},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R270},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R0},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R90},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R180},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R270},
        {scar::CopulaFamily::Gaussian,
         scar::NativeModelId::BivariateGaussian, scar::Rotation::R0},
    }};
    const std::vector<double> pair_values{
        0.20, 0.70,
        0.35, 0.55,
        0.80, 0.25,
        0.60, 0.45,
        0.15, 0.85,
    };
    const scar::ObservationView pair_observations{
        pair_values.data(), 5, 2};
    const scar::OuParams params{1.2, 0.1, 0.7};
    const scar::ScarOuEvaluator evaluator;

    // Every registered pair identity owns objective/gradient/state/residual.
    for (std::size_t index = 0; index < pair_cases.size(); ++index) {
        const int base = 20 + static_cast<int>(index) * 5;
        const PairOuCase& test = pair_cases[index];
        const scar::CopulaSpec spec = pair_spec(test);
        if (spec.model_descriptor().model_id() != test.model_id) {
            return base;
        }
        for (scar::NativeOperation operation : {
                 scar::NativeOperation::ParameterTransformBoundsInitialization,
                 scar::NativeOperation::RowGridDensityGradient,
                 scar::NativeOperation::LikelihoodObjectiveGradient,
                 scar::NativeOperation::StateFilterSmoother,
                 scar::NativeOperation::RosenblattResidual}) {
            const auto capability = scar::query_capability(
                spec.model_descriptor(),
                operation,
                scar::DynamicsKind::ScarTmOu);
            if (!capability.supported || !capability.reason.empty()) {
                return base + 1;
            }
        }
        const auto likelihood = evaluator.loglik_matrix(
            params, spec, pair_observations, dense_config);
        const auto objective = evaluator.neg_loglik_with_grad_matrix(
            params, spec, pair_observations, dense_config);
        const auto residual = evaluator.forward_rosenblatt_matrix(
            params, spec, pair_observations, dense_config);
        const auto mixture_pair = evaluator.mixture_h_pair_matrix(
            params, spec, pair_observations, dense_config);
        if (!likelihood.is_ok() || !objective.is_ok()
            || !residual.is_ok() || !mixture_pair.is_ok()
            || !std::isfinite(likelihood.log_likelihood)
            || !close(
                objective.neg_log_likelihood,
                -likelihood.log_likelihood,
                2e-12)
            || objective.neg_gradient.size() != 3
            || !finite_values(objective.neg_gradient)
            || residual.values.size() != pair_values.size()
            || mixture_pair.values.size() != pair_values.size()
            || !finite_values(residual.values)
            || !finite_values(mixture_pair.values)) {
            return base + 2;
        }
    }

    const scar::CopulaSpec representative = pair_spec(pair_cases[0]);
    const auto representative_gradient = evaluator.neg_loglik_with_grad_matrix(
        params, representative, pair_observations, dense_config);
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
        const double step = 1e-5;
        const double reference = finite_difference(
            evaluator,
            params,
            representative,
            pair_observations,
            dense_config,
            coordinate,
            step);
        if (!close(
                representative_gradient.neg_gradient[coordinate],
                reference,
                3e-5)) {
            return 95 + static_cast<int>(coordinate);
        }
    }

    // Spectral/local/matrix/auto, prepared evaluator, states and smoothing.
    const auto spectral = evaluator.loglik_spectral(
        params, representative, pair_observations, dense_config);
    const auto local = evaluator.loglik_local_gh(
        params, representative, pair_observations, sparse_config);
    const auto automatic = evaluator.loglik_auto(
        params, representative, pair_observations, dense_config);
    const auto predictive = evaluator.predictive_mean_auto(
        params, representative, pair_observations, dense_config);
    const auto current = evaluator.state_distribution_matrix(
        params, representative, pair_observations, dense_config, false);
    const auto next = evaluator.state_distribution_matrix(
        params, representative, pair_observations, dense_config, true);
    const auto smoothed = evaluator.smoothed_state_distribution_matrix(
        params, representative, pair_observations, sparse_config);
    scar::PreparedScarOuEvaluator prepared_pair(
        representative,
        pair_values,
        5,
        2,
        dense_config,
        "matrix");
    const auto prepared_likelihood = prepared_pair.loglik(params);
    const auto prepared_gradient = prepared_pair.neg_loglik_with_grad(params);
    const auto prepared_mixture = prepared_pair.mixture_h_pair(params);
    if (!spectral.is_ok() || !local.is_ok() || !automatic.is_ok()
        || !predictive.is_ok() || !current.is_ok() || !next.is_ok()
        || !smoothed.is_ok() || !prepared_likelihood.is_ok()
        || !prepared_gradient.is_ok() || !prepared_mixture.is_ok()
        || spectral.backend != scar::OuBackend::Spectral
        || local.backend != scar::OuBackend::LocalGh
        || automatic.fallback_chain.size() > 2
        || predictive.values.size() != 5
        || current.z_grid.size() != 17 || current.prob.size() != 17
        || next.z_grid.size() != 17 || next.prob.size() != 17
        || !normalized(current.prob, 2e-14)
        || !normalized(next.prob, 2e-14)
        || smoothed.n_obs != 5 || smoothed.K != 17
        || !normalized_rows(smoothed.weights, 5, 17, 2e-14)
        || !close(
            prepared_likelihood.log_likelihood,
            evaluator.loglik_matrix(
                params, representative, pair_observations, dense_config)
                .log_likelihood,
            1e-14)
        || !close_vectors(
            prepared_gradient.neg_gradient,
            representative_gradient.neg_gradient,
            1e-14)) {
        return 100;
    }

    // Equicorrelation owns Gaussian Rosenblatt and the complete OU state path.
    const int dimension = 3;
    const std::vector<double> multivariate_values{
        0.20, 0.45, 0.70,
        0.35, 0.60, 0.25,
        0.80, 0.30, 0.55,
        0.65, 0.75, 0.40,
        0.15, 0.50, 0.85,
    };
    const scar::ObservationView multivariate_observations{
        multivariate_values.data(), 5, dimension};
    const scar::CopulaSpec equicorr = equicorr_spec(dimension);
    const auto equicorr_gradient = evaluator.neg_loglik_with_grad_matrix(
        params, equicorr, multivariate_observations, dense_config);
    const auto equicorr_residual = evaluator.gaussian_rosenblatt_matrix(
        params, equicorr, multivariate_observations, dense_config);
    const auto equicorr_state = evaluator.state_distribution_local_gh(
        params, equicorr, multivariate_observations, sparse_config, true);
    const auto equicorr_smoothed =
        evaluator.smoothed_state_distribution_local_gh(
            params, equicorr, multivariate_observations, sparse_config);
    if (!equicorr_gradient.is_ok() || !equicorr_residual.is_ok()
        || !equicorr_state.is_ok() || !equicorr_smoothed.is_ok()
        || equicorr_gradient.neg_gradient.size() != 3
        || equicorr_residual.values.size() != multivariate_values.size()
        || !finite_values(equicorr_gradient.neg_gradient)
        || !finite_values(equicorr_residual.values)
        || !normalized(equicorr_state.prob, 2e-14)
        || !normalized_rows(
            equicorr_smoothed.weights, 5, 17, 2e-14)) {
        return 110;
    }

    // Stochastic Student dense and factor correlation modes share all OU
    // operations, while dynamic joint factor estimation remains unsupported.
    const std::vector<double> identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const auto zero_factor =
        std::make_shared<scar::FactorCorrelationOperator>(
            std::vector<double>(dimension, 0.0), dimension, 1, 1e-6);
    const std::array<scar::CopulaSpec, 2> student_specs{{
        dense_student_spec(identity, dimension),
        factor_student_spec(zero_factor),
    }};
    for (std::size_t index = 0; index < student_specs.size(); ++index) {
        const int base = 120 + static_cast<int>(index) * 10;
        const scar::CopulaSpec& spec = student_specs[index];
        if (!emission_blocks_match_rows(spec, multivariate_values)) {
            return 145 + static_cast<int>(index);
        }
        const auto objective = evaluator.neg_loglik_with_grad_matrix(
            params, spec, multivariate_observations, dense_config);
        if (!objective.is_ok() || objective.neg_gradient.size() != 3
            || !finite_values(objective.neg_gradient)) {
            return base;
        }
        const auto full_correlation =
            evaluator.neg_loglik_with_grad_and_corr_matrix(
                params, spec, multivariate_observations, dense_config);
        const auto residual = evaluator.student_rosenblatt_matrix(
            params, spec, multivariate_observations, dense_config);
        const auto state = evaluator.state_distribution_matrix(
            params, spec, multivariate_observations, dense_config, false);
        const auto smooth = evaluator.smoothed_state_distribution_matrix(
            params, spec, multivariate_observations, sparse_config);
        const double observation[] = {0.25, 0.50, 0.75};
        const auto conditioned = scar::condition_ou_state_distribution(
            spec,
            view(state_grid),
            view(state_probability),
            {observation, 1, dimension});
        if (index == 0) {
            if (!full_correlation.is_ok()
                || full_correlation.neg_gradient.size() != 3
                || full_correlation.neg_corr_gradient.empty()
                || !finite_values(full_correlation.neg_corr_gradient)) {
                return base + 1;
            }
            auto blocked_config = dense_config;
            blocked_config.corr_gradient_block_bytes =
                24U * static_cast<std::uint64_t>(dense_config.K);
            const auto blocked = evaluator.neg_loglik_with_grad_and_corr_matrix(
                params, spec, multivariate_observations, blocked_config);
            if (!blocked.is_ok()
                || !close(blocked.neg_log_likelihood, full_correlation.neg_log_likelihood)
                || !close_vectors(blocked.neg_gradient, full_correlation.neg_gradient)
                || !close_vectors(blocked.neg_corr_gradient,
                                  full_correlation.neg_corr_gradient)) {
                return base + 8;
            }
            blocked_config.corr_gradient_block_bytes = 24U;
            const auto too_small = evaluator.neg_loglik_with_grad_and_corr_matrix(
                params, spec, multivariate_observations, blocked_config);
            if (too_small.status != scar::Status::InvalidSize) {
                return base + 9;
            }
        } else if (full_correlation.status
                   != scar::Status::InvalidTransform) {
            return base + 1;
        }
        if (!residual.is_ok()) {
            return base + 2;
        }
        if (residual.values.size() != multivariate_values.size()) {
            return base + 6;
        }
        if (!finite_values(residual.values)) {
            return base + 7;
        }
        if (!state.is_ok() || !normalized(state.prob, 2e-14)) {
            return base + 3;
        }
        if (!smooth.is_ok()
            || !normalized_rows(smooth.weights, 5, 17, 2e-14)) {
            return base + 4;
        }
        if (!conditioned.is_ok()
            || !normalized(conditioned.prob, 2e-14)) {
            return base + 5;
        }
    }

    const auto dynamic_joint = scar::make_typed_model_descriptor(
        scar::NativeModelId::StochasticStudent,
        dimension,
        scar::CorrelationKind::Factor,
        1,
        scar::FactorEstimationKind::Joint);
    const auto dynamic_joint_capability = scar::query_capability(
        dynamic_joint,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::DynamicsKind::ScarTmOu);
    const auto invalid = evaluator.neg_loglik_with_grad_matrix(
        scar::OuParams{-1.0, 0.0, 1.0},
        representative,
        pair_observations,
        dense_config);
    scar::OuNumericalConfig invalid_threads = dense_config;
    invalid_threads.n_threads = 0;
    const auto invalid_config = evaluator.loglik_matrix(
        params, representative, pair_observations, invalid_threads);
    if (dynamic_joint_capability.supported
        || dynamic_joint_capability.reason.empty()
        || invalid.status != scar::Status::InvalidParameter
        || invalid.failure.index != -1
        || invalid_config.status != scar::Status::InvalidParameter) {
        return 140;
    }

    if (!emission_blocks_match_rows(representative, pair_values)
        || !emission_blocks_match_rows(equicorr, multivariate_values)) {
        return 147;
    }
    for (int d : {2, 10}) {
        std::vector<double> R(static_cast<std::size_t>(d * d), 0.2);
        for (int j = 0; j < d; ++j) {
            R[static_cast<std::size_t>(j * d + j)] = 1.0;
        }
        auto spec = dense_student_spec(R, d);
        std::vector<double> observations(static_cast<std::size_t>(19 * d));
        for (std::size_t j = 0; j < observations.size(); ++j) {
            observations[j] = 0.1 + 0.8 * static_cast<double>((j * 31) % 101) / 101.;
        }
        auto table = scar::copula::multivariate::student::prepare_ppf_table(
            view(observations), {});
        if (!table.is_ok() || !table.value.has_table) {
            return 148;
        }
        spec.student_ppf_observation_count() = 19;
        spec.student_ppf_nodes() = std::move(table.value.nodes);
        spec.student_ppf_table() = std::move(table.value.table);
        if (!emission_blocks_match_rows(spec, observations)) {
            return 149;
        }
    }
    return 0;
}
