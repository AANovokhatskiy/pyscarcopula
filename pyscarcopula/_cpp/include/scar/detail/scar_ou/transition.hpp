#pragma once

#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"

#include <vector>

namespace scar {
enum class OuBackend : int;
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

struct GridTransitionOperator {
    int K = 0;
    bool local_gh = false;
    MatrixTransitionOperator matrix;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
};

struct ForwardFilterOptions {
    bool store_predictive_weights = false;
    bool store_filtered_weights = false;
};

struct ForwardFilterResult {
    std::int64_t n_obs = 0;
    int K = 0;
    std::vector<double> predictive_weights;
    std::vector<double> filtered_weights;
    std::vector<double> final_filtered_density;
};

struct BackwardFilterResult {
    std::int64_t n_obs = 0;
    int K = 0;
    std::vector<double> messages;
};

struct SmoothedStateResult {
    std::int64_t n_obs = 0;
    int K = 0;
    std::vector<double> weights;
};

int select_grid_transition_backend(const OuGrid& grid, double r_gh);
bool normalize_density_by_max(std::vector<double>& values);

template <typename AdvancePhi, typename OnRow>
bool forward_filter_grid(
    const scar::PreparedDynamicEmission& emission,
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
        emission.fill_density_row_on_state_grid(
            u, t, grid.x_grid, fi_row);
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
bool build_grid_transition_operator(
    const OuGrid& grid,
    scar::OuBackend backend,
    scar::OuGridMethod method,
    int gh_order,
    GridTransitionOperator& op);
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
void grid_backward_matvec(
    const GridTransitionOperator& op,
    const OuGrid& grid,
    const std::vector<double>& values,
    std::vector<double>& out);
void grid_predict_matvec(
    const GridTransitionOperator& op,
    const OuGrid& grid,
    const std::vector<double>& source,
    std::vector<double>& out_density);
bool advance_matrix_forward_density(
    const MatrixTransitionOperator& op,
    const OuGrid& grid,
    const std::vector<double>& phi,
    const std::vector<double>& emission,
    std::vector<double>& source,
    std::vector<double>& phi_next);
bool advance_local_forward_density(
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& phi,
    const std::vector<double>& emission,
    std::vector<double>& source,
    std::vector<double>& phi_next);
bool forward_filter_emissions(
    const OuGrid& grid,
    const GridTransitionOperator& transition,
    const double* emissions,
    std::int64_t n_obs,
    const ForwardFilterOptions& options,
    ForwardFilterResult& result);
bool backward_filter_emissions(
    const OuGrid& grid,
    const GridTransitionOperator& transition,
    const double* emissions,
    std::int64_t n_obs,
    BackwardFilterResult& result);
bool smooth_state_emissions(
    const OuGrid& grid,
    const GridTransitionOperator& transition,
    const double* emissions,
    std::int64_t n_obs,
    SmoothedStateResult& result);
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
    const scar::PreparedDynamicEmission& emission,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double& loglik);
bool matrix_forward_predictive_mean(
    const scar::PreparedDynamicEmission& emission,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double* out);
bool matrix_forward_mixture_h(
    const scar::PreparedDynamicEmission& emission,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double* out,
    double* out_reverse = nullptr);
bool local_forward_predictive_mean(
    const scar::PreparedDynamicEmission& emission,
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const double* u,
    std::int64_t n_obs,
    double* out);
bool local_forward_mixture_h(
    const scar::PreparedDynamicEmission& emission,
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const double* u,
    std::int64_t n_obs,
    double* out,
    double* out_reverse = nullptr);

}  // namespace scar_internal
