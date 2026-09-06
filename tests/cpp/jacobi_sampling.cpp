#include "scar/jacobi.hpp"

#include "scar/copula/prepared_pair_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

scar::JacobiTransitionConfig grid_config(
    scar::JacobiTransitionStorage storage) {
    scar::JacobiTransitionConfig config;
    config.method = scar::JacobiTransitionMethod::LocalFixed;
    config.storage = storage;
    config.numerical.n_obs = 8;
    config.numerical.quad_order = 16;
    config.numerical.basis_order = 4;
    config.numerical.gh_order = 3;
    config.numerical.matrix = storage == scar::JacobiTransitionStorage::Dense;
    return config;
}

bool finite_unit(const std::vector<double>& values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) {
            return std::isfinite(value) && value >= 0.0 && value <= 1.0;
        });
}

}  // namespace

int run_jacobi_sampling_tests() {
    const scar::JacobiParams params{1.2, 0.4, 0.25};
    const std::vector<double> uniforms{
        0.02, 0.17, 0.31, 0.48, 0.59, 0.73, 0.84, 0.96};
    const scar::JacobiTrajectoryResult dense =
        scar::sample_jacobi_grid_trajectory(
            params,
            grid_config(scar::JacobiTransitionStorage::Dense),
            uniforms);
    const scar::JacobiTrajectoryResult sparse =
        scar::sample_jacobi_grid_trajectory(
            params,
            grid_config(scar::JacobiTransitionStorage::Sparse),
            uniforms);
    if (!dense.is_ok() || !sparse.is_ok()
        || dense.value.tau.size() != uniforms.size()
        || sparse.value.tau.size() != uniforms.size()
        || dense.value.draws_used != 8
        || sparse.value.draws_used != 8
        || !finite_unit(dense.value.tau)
        || !finite_unit(sparse.value.tau)
        || dense.value.transition.storage
            != scar::JacobiTransitionStorage::Dense
        || sparse.value.transition.storage
            != scar::JacobiTransitionStorage::Sparse) {
        return 1;
    }

    const scar::JacobiScalarResult initial = scar::jacobi_lamperti(
        0.37, params.xi);
    scar::JacobiLampertiSamplingConfig lamperti_config;
    lamperti_config.n_obs = 5;
    lamperti_config.substeps = 2;
    const std::vector<double> normals{
        0.2, -0.1, 0.3, 0.4, -0.5, 0.7, -0.2, 0.1};
    const scar::JacobiTrajectoryResult full =
        scar::sample_jacobi_lamperti_chunk(
            params, lamperti_config, initial.value, normals);
    const scar::JacobiTrajectoryResult first =
        scar::sample_jacobi_lamperti_chunk(
            params,
            lamperti_config,
            initial.value,
            std::vector<double>(normals.begin(), normals.begin() + 4));
    const scar::JacobiTrajectoryResult second =
        scar::sample_jacobi_lamperti_chunk(
            params,
            lamperti_config,
            first.value.final_lamperti_value,
            std::vector<double>(normals.begin() + 4, normals.end()));
    std::vector<double> chunked = first.value.tau;
    chunked.insert(
        chunked.end(), second.value.tau.begin(), second.value.tau.end());
    if (!initial.is_ok()
        || !full.is_ok() || !first.is_ok() || !second.is_ok()
        || full.value.tau != chunked
        || full.value.normal_draws_used != 8
        || first.value.normal_draws_used != 4
        || second.value.normal_draws_used != 4
        || full.value.euler_steps != 8
        || !finite_unit(full.value.tau)) {
        return 2;
    }

    scar::JacobiLampertiSamplingConfig boundary_config;
    boundary_config.n_obs = 3;
    boundary_config.substeps = 1;
    boundary_config.boundary = scar::JacobiBoundaryPolicy::Reflect;
    const scar::JacobiParams boundary_params{0.1, 0.5, 1.0};
    const scar::JacobiScalarResult boundary_initial = scar::jacobi_lamperti(
        0.5, boundary_params.xi);
    const scar::JacobiTrajectoryResult boundary =
        scar::sample_jacobi_lamperti_chunk(
            boundary_params,
            boundary_config,
            boundary_initial.value,
            {100.0, -100.0});
    if (!boundary.is_ok()
        || boundary.value.boundary_interventions != 2
        || boundary.value.normal_draws_used != 2
        || !finite_unit(boundary.value.tau)) {
        return 3;
    }

    scar::CopulaSpec copula = scar::default_pair_copula_spec(
        scar::CopulaFamily::Gumbel);
    const scar::JacobiStateSampleResult state =
        scar::sample_jacobi_state_distribution(
            copula,
            {0.2, 0.6},
            {0.25, 0.75},
            {0.1, 0.4, 0.9},
            {},
            scar::JacobiStateSamplingMode::Grid,
            2.0);
    if (!state.is_ok()
        || state.value.tau != std::vector<double>({0.2, 0.6, 0.6})
        || state.value.parameters != std::vector<double>({1.25, 2.0, 2.0})
        || state.value.selection_draws_used != 3
        || state.value.jitter_draws_used != 0) {
        return 4;
    }
    const scar::JacobiStateSampleResult histogram =
        scar::sample_jacobi_state_distribution(
            copula,
            {0.2, 0.6},
            {0.25, 0.75},
            {0.1, 0.9},
            {0.5, 0.5},
            scar::JacobiStateSamplingMode::Histogram,
            std::numeric_limits<double>::quiet_NaN());
    if (!histogram.is_ok()
        || histogram.value.selection_draws_used != 2
        || histogram.value.jitter_draws_used != 2
        || !finite_unit(histogram.value.tau)) {
        return 5;
    }
    const scar::JacobiHistogramCellsResult cells =
        scar::jacobi_state_histogram_cells({0.2, 0.6}, {0, 1});
    if (!cells.is_ok()
        || cells.value.left != std::vector<double>({0.2, 0.4})
        || cells.value.right != std::vector<double>({0.4, 0.6})) {
        return 6;
    }
    const scar::JacobiStateSampleResult invalid =
        scar::sample_jacobi_state_distribution(
            copula,
            {0.2, 0.6},
            {0.25, 0.75},
            {1.0},
            {},
            scar::JacobiStateSamplingMode::Grid,
            std::numeric_limits<double>::quiet_NaN());
    if (invalid.is_ok() || invalid.failure.index != 0) {
        return 7;
    }
    for (double scale : {0.2, 2.0, 1e-200, 1e200,
                         2.0 * std::numeric_limits<double>::denorm_min()}) {
        const auto scaled = scar::sample_jacobi_state_distribution(
            copula, {0.2, 0.8}, {0.5 * scale, 0.5 * scale},
            {0.0, 0.2, 0.5, 0.9}, {}, scar::JacobiStateSamplingMode::Grid, 10.0);
        if (!scaled.is_ok()
            || scaled.value.tau != std::vector<double>{0.2, 0.2, 0.8, 0.8}
            || scaled.value.selection_draws_used != 4) {
            return 8;
        }
    }
    for (const auto& probability : std::vector<std::vector<double>>{
             {0.0, 0.0}, {-0.1, 1.1},
             {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}}) {
        const auto bad_mass = scar::sample_jacobi_state_distribution(
            copula, {0.2, 0.8}, probability, {0.2}, {},
            scar::JacobiStateSamplingMode::Grid, 10.0);
        if (bad_mass.is_ok() || !bad_mass.value.tau.empty()) {
            return 9;
        }
    }
    return 0;
}
