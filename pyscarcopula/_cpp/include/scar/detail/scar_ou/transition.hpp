#pragma once

#include "scar/detail/copula.hpp"
#include "scar/detail/scar_ou/grid.hpp"

#include <vector>

namespace scar_internal {

int select_grid_transition_backend(const OuGrid& grid, double r_gh);

template <typename AdvancePhi, typename OnRow>
bool forward_filter_grid(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const double* u,
    std::int64_t n_obs,
    AdvancePhi advance_phi,
    OnRow on_row) {

    std::vector<double> phi = grid.p0;
    std::vector<double> weights(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> phi_next(static_cast<std::size_t>(grid.K), 0.0);

    for (std::int64_t t = 0; t < n_obs; ++t) {
        if (!predictive_weights_from_phi(grid, phi, weights)) {
            return false;
        }
        copula_fi_row_on_grid(copula, u, t, grid.x_grid, fi_row);
        on_row(t, weights, fi_row);

        if (t < n_obs - 1) {
            if (!advance_phi(phi, fi_row, phi_next)) {
                return false;
            }
            phi.swap(phi_next);
        }
    }

    return true;
}

bool build_dense_transition_matrix(const OuGrid& grid, std::vector<double>& matrix);
void dense_matvec(
    const std::vector<double>& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out);
void dense_predict_matvec(
    const std::vector<double>& matrix,
    const OuGrid& grid,
    const std::vector<double>& source,
    std::vector<double>& out_density);
bool matrix_backward_loglik(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& matrix,
    const double* u,
    std::int64_t n_obs,
    double& loglik);
bool matrix_forward_predictive_mean(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& matrix,
    const double* u,
    std::int64_t n_obs,
    double* out);
bool matrix_forward_mixture_h(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& matrix,
    const double* u,
    std::int64_t n_obs,
    double* out);
bool local_forward_predictive_mean(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const double* u,
    std::int64_t n_obs,
    double* out);
bool local_forward_mixture_h(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const double* u,
    std::int64_t n_obs,
    double* out);

}  // namespace scar_internal
