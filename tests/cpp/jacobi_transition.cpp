#include "scar/jacobi.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

bool close(double lhs, double rhs, double tolerance = 5e-12) {
    return std::abs(lhs - rhs)
        <= tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

scar::JacobiTransitionConfig config(
    scar::JacobiTransitionMethod method,
    scar::JacobiTransitionStorage storage,
    int order = 32,
    int basis_order = 8,
    int gh_order = 5,
    int n_obs = 33) {
    scar::JacobiTransitionConfig value;
    value.method = method;
    value.storage = storage;
    value.numerical.quad_order = order;
    value.numerical.basis_order = basis_order;
    value.numerical.gh_order = gh_order;
    value.numerical.n_obs = n_obs;
    value.numerical.matrix = storage == scar::JacobiTransitionStorage::Dense;
    return value;
}

std::vector<double> sparse_to_dense(
    const scar::JacobiSparseTransition& transition) {
    const std::size_t order = static_cast<std::size_t>(transition.order);
    const std::size_t width = static_cast<std::size_t>(transition.max_width);
    std::vector<double> dense(order * order, 0.0);
    for (std::size_t row = 0; row < order; ++row) {
        const std::size_t count = static_cast<std::size_t>(
            transition.counts[row]);
        for (std::size_t slot = 0; slot < count; ++slot) {
            const std::size_t offset = row * width + slot;
            dense[row * order
                + static_cast<std::size_t>(transition.indices[offset])] =
                transition.probabilities[offset];
        }
    }
    return dense;
}

}  // namespace

int run_jacobi_transition_tests() {
    const scar::JacobiParams params{1.2, 0.4, 0.25};
    const scar::JacobiIntResult default_order =
        scar::default_jacobi_quad_order(8);
    if (!default_order.is_ok() || default_order.value != 48) {
        return 17;
    }
    scar::JacobiTransitionConfig workspace_config = config(
        scar::JacobiTransitionMethod::Local,
        scar::JacobiTransitionStorage::Sparse,
        16,
        4,
        3,
        8);
    workspace_config.correction = scar::JacobiStationarityCorrection::IpFp;
    const scar::JacobiMemoryResult sparse_workspace =
        scar::estimate_jacobi_sparse_workspace(workspace_config);
    if (!sparse_workspace.is_ok()
        || sparse_workspace.value.bytes != 8744) {
        return 18;
    }
    const scar::JacobiMemoryResult sparse_storage =
        scar::estimate_jacobi_sparse_storage(workspace_config);
    if (!sparse_storage.is_ok()
        || sparse_storage.value.bytes != 4864) {
        return 20;
    }
    const scar::JacobiVectorResult powers = scar::jacobi_transition_powers(
        params, 5, 4);
    if (!powers.is_ok() || powers.value.size() != 4
        || !close(powers.value[0], 1.0)
        || !close(powers.value[1], std::exp(-1.2 * 0.25))) {
        return 1;
    }
    scar::JacobiTransitionConfig invalid_time_grid = workspace_config;
    invalid_time_grid.numerical.n_obs = 1;
    if (scar::jacobi_transition_powers(params, 1, 4).is_ok()
        || scar::validate_jacobi_transition_config(invalid_time_grid)
            != scar::Status::InvalidSize) {
        return 24;
    }
    scar::JacobiTransitionConfig coefficient_config = config(
        scar::JacobiTransitionMethod::SpectralCoeff,
        scar::JacobiTransitionStorage::Dense,
        16,
        4,
        1,
        5);
    const scar::JacobiCoefficientTransitionResult coefficient_transition =
        scar::build_jacobi_coefficient_transition(
            params, coefficient_config);
    const scar::JacobiVectorResult propagated_coefficients =
        coefficient_transition.is_ok()
        ? scar::apply_jacobi_coefficient_transition(
            coefficient_transition.value, {1.0, 2.0, 3.0, 4.0})
        : scar::JacobiVectorResult{};
    if (!coefficient_transition.is_ok()
        || coefficient_transition.value.basis.size() != 16 * 4
        || coefficient_transition.value.diagnostics.method_used
            != scar::JacobiTransitionMethod::SpectralCoeff
        || !propagated_coefficients.is_ok()
        || !close(
            propagated_coefficients.value[2],
            3.0 * coefficient_transition.value.spectral_powers[2])) {
        return 19;
    }

    scar::JacobiTransitionConfig spectral_config = config(
        scar::JacobiTransitionMethod::SpectralMatrix,
        scar::JacobiTransitionStorage::Dense,
        48,
        16,
        5,
        2);
    const scar::JacobiDenseTransitionResult spectral =
        scar::build_jacobi_spectral_transition(params, spectral_config);
    if (!spectral.is_ok()
        || spectral.value.diagnostics.method_used
            != scar::JacobiTransitionMethod::SpectralMatrix
        || spectral.value.diagnostics.max_row_sum_error_before_normalization
            > 2e-11
        || spectral.value.diagnostics.stationary_error > 2e-11) {
        return 2;
    }
    for (std::size_t row = 0; row < spectral.value.tau.size(); ++row) {
        double conditional_mean = 0.0;
        for (std::size_t column = 0;
             column < spectral.value.tau.size(); ++column) {
            conditional_mean += spectral.value.probabilities[
                row * spectral.value.tau.size() + column]
                * spectral.value.tau[column];
        }
        const double expected = params.m
            + (spectral.value.tau[row] - params.m)
                * std::exp(-params.kappa);
        if (!close(conditional_mean, expected, 2e-11)) {
            return 3;
        }
    }

    scar::JacobiTransitionConfig auto_config = config(
        scar::JacobiTransitionMethod::Auto,
        scar::JacobiTransitionStorage::Dense,
        40,
        6,
        5,
        2);
    auto_config.numerical.n_obs = 1000001;
    const scar::JacobiDenseTransitionResult automatic =
        scar::build_jacobi_dense_transition(params, auto_config);
    if (!automatic.is_ok()
        || automatic.value.diagnostics.method_used
            != scar::JacobiTransitionMethod::Local
        || automatic.value.diagnostics.method_requested
            != scar::JacobiTransitionMethod::Auto) {
        return 4;
    }

    scar::JacobiTransitionConfig local_dense_config = config(
        scar::JacobiTransitionMethod::Local,
        scar::JacobiTransitionStorage::Dense,
        48,
        8,
        5,
        33);
    scar::JacobiTransitionConfig local_sparse_config = local_dense_config;
    local_sparse_config.storage = scar::JacobiTransitionStorage::Sparse;
    local_sparse_config.numerical.matrix = false;
    const scar::JacobiDenseTransitionResult local_dense =
        scar::build_jacobi_dense_transition(params, local_dense_config);
    const scar::JacobiSparseTransitionResult local_sparse =
        scar::build_jacobi_sparse_transition(params, local_sparse_config);
    if (!local_dense.is_ok() || !local_sparse.is_ok()
        || local_sparse.value.max_width != 10
        || local_sparse.value.diagnostics.nnz > 480) {
        return 5;
    }
    const std::vector<double> reconstructed =
        sparse_to_dense(local_sparse.value);
    for (std::size_t index = 0; index < reconstructed.size(); ++index) {
        if (!close(
                reconstructed[index],
                local_dense.value.probabilities[index],
                2e-14)) {
            return 6;
        }
    }
    const scar::JacobiVectorResult propagated =
        scar::jacobi_sparse_left_multiply(
            local_sparse.value, local_sparse.value.weights);
    if (!propagated.is_ok()
        || propagated.value.size() != local_sparse.value.weights.size()) {
        return 7;
    }

    scar::JacobiTransitionConfig fixed_dense_config = config(
        scar::JacobiTransitionMethod::LocalFixed,
        scar::JacobiTransitionStorage::Dense,
        32,
        1,
        5,
        37);
    fixed_dense_config.derivatives = true;
    fixed_dense_config.numerical.gradient = true;
    scar::JacobiTransitionConfig fixed_sparse_config = fixed_dense_config;
    fixed_sparse_config.storage = scar::JacobiTransitionStorage::Sparse;
    fixed_sparse_config.numerical.matrix = false;
    fixed_sparse_config.numerical.gradient = false;
    const scar::JacobiDenseTransitionResult fixed_dense =
        scar::build_jacobi_dense_transition(params, fixed_dense_config);
    const scar::JacobiSparseTransitionResult fixed_sparse =
        scar::build_jacobi_sparse_transition(params, fixed_sparse_config);
    if (!fixed_dense.is_ok() || !fixed_sparse.is_ok()
        || fixed_dense.value.derivatives.size() != 3 * 32 * 32
        || fixed_sparse.value.derivatives.size() != 3 * 32 * 10
        || fixed_dense.value.diagnostics.dense_bytes != 4 * 32 * 32 * 8) {
        return 8;
    }
    const scar::JacobiHorizonResult fixed_horizon =
        scar::jacobi_sparse_full_horizon_diagnostics(
            params,
            fixed_sparse.value.tau,
            fixed_sparse.value.weights,
            fixed_sparse.value,
            36);
    if (!fixed_horizon.is_ok()
        || fixed_horizon.value.conditional_mean_rmse > 1e-3
        || std::abs(fixed_horizon.value.lag_one_correlation_error) > 1e-2) {
        return 23;
    }
    const double step = 1e-6;
    scar::JacobiTransitionConfig value_config = fixed_dense_config;
    value_config.derivatives = false;
    value_config.numerical.gradient = false;
    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
        scar::JacobiParams plus = params;
        scar::JacobiParams minus = params;
        if (parameter == 0) {
            plus.kappa += step;
            minus.kappa -= step;
        } else if (parameter == 1) {
            plus.m += step;
            minus.m -= step;
        } else {
            plus.xi += step;
            minus.xi -= step;
        }
        const scar::JacobiDenseTransitionResult plus_transition =
            scar::build_jacobi_dense_transition(plus, value_config);
        const scar::JacobiDenseTransitionResult minus_transition =
            scar::build_jacobi_dense_transition(minus, value_config);
        if (!plus_transition.is_ok() || !minus_transition.is_ok()) {
            return 9;
        }
        for (std::size_t index = 0; index < 32 * 32; ++index) {
            const double finite_difference = (
                plus_transition.value.probabilities[index]
                - minus_transition.value.probabilities[index])
                / (2.0 * step);
            if (!close(
                    finite_difference,
                    fixed_dense.value.derivatives[
                        parameter * 32 * 32 + index],
                    3e-6)) {
                return 10;
            }
        }
    }

    scar::JacobiTransitionConfig mh_config = local_sparse_config;
    mh_config.correction =
        scar::JacobiStationarityCorrection::MetropolisHastings;
    const scar::JacobiSparseTransitionResult mh =
        scar::build_jacobi_sparse_transition(params, mh_config);
    if (!mh.is_ok()
        || mh.value.diagnostics.stationary_error > 2e-15
        || mh.value.diagnostics.detailed_balance_error > 2e-15
        || !(mh.value.diagnostics.acceptance_mass_ratio > 0.0)) {
        return 11;
    }

    scar::JacobiTransitionConfig ipfp_config = local_sparse_config;
    ipfp_config.correction = scar::JacobiStationarityCorrection::IpFp;
    const scar::JacobiSparseTransitionResult ipfp =
        scar::build_jacobi_sparse_transition(params, ipfp_config);
    if (!ipfp.is_ok()
        || ipfp.value.diagnostics.ipfp_iterations <= 0
        || ipfp.value.diagnostics.ipfp_stationary_residual > 1e-15) {
        return 12;
    }
    const scar::JacobiHorizonResult horizon =
        scar::jacobi_sparse_full_horizon_diagnostics(
            params,
            ipfp.value.tau,
            ipfp.value.weights,
            ipfp.value,
            32);
    if (!horizon.is_ok()
        || horizon.value.full_horizon_stationary_tv > 1e-12
        || !std::isfinite(horizon.value.conditional_mean_rmse)
        || !std::isfinite(horizon.value.lag_one_correlation_error)) {
        return 13;
    }

    scar::JacobiTransitionConfig adaptive_config = local_sparse_config;
    adaptive_config.numerical.n_obs = 17;
    const scar::JacobiAdaptiveSelectionResult adaptive =
        scar::select_sparse_jacobi_order(
            params,
            adaptive_config,
            {48, 128},
            scar::JacobiAdaptiveThresholds{},
            false);
    if (!adaptive.is_ok()
        || adaptive.value.selected_quad_order != 128
        || !adaptive.value.passed
        || adaptive.value.candidates.size() != 2) {
        return 14;
    }

    scar::JacobiTransitionConfig limited = local_dense_config;
    limited.numerical.memory_budget_bytes = 1;
    const scar::JacobiDenseTransitionResult memory_failure =
        scar::build_jacobi_dense_transition(params, limited);
    if (memory_failure.is_ok()
        || memory_failure.status != scar::Status::InvalidSize
        || memory_failure.value.diagnostics.estimated_workspace_bytes <= 1) {
        return 15;
    }

    scar::JacobiTransitionConfig large_n = local_sparse_config;
    large_n.numerical.n_obs = 1000000;
    const scar::JacobiMemoryResult large_n_memory =
        scar::estimate_jacobi_sparse_workspace(large_n);
    large_n.numerical.n_obs = 2;
    const scar::JacobiMemoryResult minimum_n_memory =
        scar::estimate_jacobi_sparse_workspace(large_n);
    if (!large_n_memory.is_ok() || !minimum_n_memory.is_ok()
        || large_n_memory.value.bytes != minimum_n_memory.value.bytes) {
        return 21;
    }

    scar::JacobiTransitionConfig limited_sparse = ipfp_config;
    limited_sparse.numerical.memory_budget_bytes = 1;
    const scar::JacobiSparseTransitionResult sparse_memory_failure =
        scar::build_jacobi_sparse_transition(params, limited_sparse);
    if (sparse_memory_failure.is_ok()
        || sparse_memory_failure.value.diagnostics.method_requested
            != scar::JacobiTransitionMethod::Local
        || sparse_memory_failure.value.diagnostics.storage
            != scar::JacobiTransitionStorage::Sparse
        || sparse_memory_failure.value.diagnostics.correction
            != scar::JacobiStationarityCorrection::IpFp) {
        return 22;
    }

    scar::JacobiTransitionConfig invalid = local_sparse_config;
    invalid.method = scar::JacobiTransitionMethod::SpectralMatrix;
    if (scar::ok(scar::validate_jacobi_transition_config(invalid))) {
        return 16;
    }
    return 0;
}
