#pragma once

#include "scar/detail/copula.hpp"
#include "scar/detail/scar_ou/grid.hpp"

#include <vector>

namespace scar {
enum class OuGridMethod : int;
}

namespace scar_internal {

struct SparseTransitionMatrix {
    std::vector<double> data;
    std::vector<int> indices;
    std::vector<int> indptr;
};

struct MatrixTransitionOperator {
    int K = 0;
    bool sparse = false;
    std::vector<double> dense;
    SparseTransitionMatrix csr;
};

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
bool build_sparse_transition_matrix(
    const std::vector<double>& z,
    double rho,
    double sigma_cond,
    const std::vector<double>& trap_w,
    int K,
    int band,
    SparseTransitionMatrix& matrix,
    const std::vector<double>* i_centers = nullptr);
void sparse_matvec(
    const SparseTransitionMatrix& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out);
void sparse_transpose_matvec(
    const SparseTransitionMatrix& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out);
int matrix_transition_band(const OuGrid& grid);
bool build_matrix_transition_operator(
    const OuGrid& grid,
    scar::OuGridMethod method,
    MatrixTransitionOperator& op);
void matrix_matvec(
    const MatrixTransitionOperator& op,
    const std::vector<double>& v,
    std::vector<double>& out);
void matrix_transpose_matvec(
    const MatrixTransitionOperator& op,
    const std::vector<double>& v,
    std::vector<double>& out);
void matrix_predict_matvec(
    const MatrixTransitionOperator& op,
    const OuGrid& grid,
    const std::vector<double>& source,
    std::vector<double>& out_density);
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
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double& loglik);
bool matrix_forward_predictive_mean(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double* out);
bool matrix_forward_mixture_h(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double* out,
    double* out_reverse = nullptr);
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
    double* out,
    double* out_reverse = nullptr);

}  // namespace scar_internal
