#pragma once

#include "scar/copula.hpp"
#include "scar/numerical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace scar_internal {

inline constexpr double kOffset = 0.0001;
inline constexpr double kPdfEps = scar::numerical::kPdfFloor;
inline constexpr double kHEps = scar::numerical::kHFunctionEps;
inline constexpr double kPseudoObsEps =
    scar::numerical::kPseudoObservationEps;
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr std::size_t kMaxGridSize = 100000;
inline constexpr std::size_t kMaxDenseGridSize = 10000;
inline constexpr std::size_t kMaxSpectralOrder = 1024;
// The process-wide Hermite cache is bounded independently by entry count and
// by the vector storage retained for nodes, weights, and basis matrices.
inline constexpr std::size_t kHermiteRuleCacheMaxEntries = 16;
inline constexpr std::size_t kHermiteRuleCacheMaxBytes =
    8 * 1024 * 1024;

inline double clip_pseudo_observation(double value) noexcept {
    return std::clamp(value, kPseudoObsEps, 1.0 - kPseudoObsEps);
}

inline bool checked_size_mul(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t& result) noexcept {

    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool checked_size_add(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t& result) noexcept {

    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

inline bool checked_nonnegative_size(
    std::int64_t value,
    std::size_t& result) noexcept {

    if (value < 0
        || static_cast<std::uint64_t>(value)
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    result = static_cast<std::size_t>(value);
    return true;
}

inline bool checked_positive_int_size(
    int value,
    std::size_t limit,
    std::size_t& result) noexcept {

    if (value <= 0) {
        return false;
    }
    result = static_cast<std::size_t>(value);
    return result <= limit;
}

inline bool checked_row_offset(
    std::size_t row,
    std::size_t width,
    std::size_t total,
    std::size_t& result) noexcept {

    return checked_size_mul(row, width, result) && result <= total;
}

inline bool valid_student_dimension(int dim, std::size_t& square) noexcept {
    if (dim <= 0) {
        return false;
    }
    const std::size_t dim_size = static_cast<std::size_t>(dim);
    return checked_size_mul(dim_size, dim_size, square);
}

inline bool valid_student_correlation_count(
    int dim,
    std::size_t& count) noexcept {

    if (dim < 2) {
        return false;
    }
    const std::size_t dim_size = static_cast<std::size_t>(dim);
    std::size_t product = 0;
    if (!checked_size_mul(dim_size, dim_size - 1, product)) {
        return false;
    }
    count = product / 2;
    return true;
}

inline bool valid_spectral_dimensions(
    int quad_order,
    int basis_order,
    std::size_t& basis_elements) noexcept {

    std::size_t quad_size = 0;
    std::size_t basis_size = 0;
    return checked_positive_int_size(
               quad_order, kMaxSpectralOrder, quad_size)
        && checked_positive_int_size(
               basis_order, kMaxSpectralOrder, basis_size)
        && quad_order >= basis_order
        && checked_size_mul(quad_size, basis_size, basis_elements);
}

struct OuGrid {
    int K = 0;
    double rho = 0.0;
    double sigma = 0.0;
    double sigma_cond = 0.0;
    double dz = 0.0;
    double r_kernel_grid = 0.0;
    bool adaptive_was_capped = false;
    std::vector<double> z;
    std::vector<double> x_grid;
    std::vector<double> trap_w;
    std::vector<double> p0;
};

struct HermiteRuleCacheInfo {
    std::size_t entries = 0;
    std::size_t bytes = 0;
    std::size_t max_entries = 0;
    std::size_t max_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t insertions = 0;
    std::uint64_t evictions = 0;
    std::uint64_t oversized_skips = 0;
    std::uint64_t duplicate_builds = 0;
};

bool is_valid_rotation(int rotation);
using ConditionalKernel = double (*)(double, double, double);
int select_grid_transition_backend(const OuGrid& grid, double r_gh);
double softplus(double x);
double d_softplus(double x);
double log1mexp(double x);
double logsumexp(double a, double b);
double normal_quantile(double p);
double normal_quantile_refined(double p);
double equicorr_transform(const scar::CopulaSpec& spec, double x);
double equicorr_inverse_transform(const scar::CopulaSpec& spec, double rho);
double equicorr_dtransform(const scar::CopulaSpec& spec, double x);
double equicorr_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double rho,
    double* dlog_drho);
double copula_transform(const scar::CopulaSpec& spec, double x);
double copula_inverse_transform(const scar::CopulaSpec& spec, double r);
double copula_dtransform(const scar::CopulaSpec& spec, double x);
double copula_tau_to_param(const scar::CopulaSpec& spec, double tau);
double copula_param_to_tau(const scar::CopulaSpec& spec, double r);
void apply_rotation(double u1, double u2, int rotation, double& v1, double& v2);
double evaluate_rotated_conditional(
    double first,
    double second,
    double parameter,
    int rotation,
    ConditionalKernel kernel);

double clayton_log_pdf_unrotated(double u1, double u2, double r);
double clayton_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void clayton_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx);
double clayton_h_rotated(double u, double v, double r, int rotation);
double clayton_h_inverse_rotated(double q, double given, double r, int rotation);
double gumbel_log_pdf_unrotated(double u1, double u2, double r);
double gumbel_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void gumbel_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx);
double gumbel_h_rotated(double u, double v, double r, int rotation);
double gumbel_h_inverse_rotated(double q, double given, double r, int rotation);
double frank_log_pdf_unrotated(double u1, double u2, double r);
double frank_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void frank_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx);
double frank_h_rotated(double u, double v, double r, int rotation);
double frank_h_inverse_rotated(double q, double given, double r, int rotation);
double joe_log_pdf_unrotated(double u1, double u2, double r);
double joe_dlog_pdf_dr_unrotated(double u1, double u2, double r);
void joe_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double offset,
    double& pdf,
    double& d_pdf_dx);
double joe_h_rotated(double u, double v, double r, int rotation);
double joe_h_inverse_rotated(double q, double given, double r, int rotation);
double gaussian_log_pdf_unrotated(double u1, double u2, double rho);
double gaussian_dlog_pdf_dr_unrotated(double u1, double u2, double rho);
void gaussian_pdf_and_grad_x_unrotated(
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx);
double gaussian_h_rotated(double u, double v, double rho, int rotation);
double gaussian_h_inverse_rotated(double q, double given, double rho, int rotation);
double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index);
bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf);
double student_quantile_value(double p, double df);
bool student_precision_matrix(
    const scar::CopulaSpec& spec,
    std::vector<double>& precision);
bool student_corr_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    double* scores);
void student_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row);
void student_fill_row_from_x_grid(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& x_grid,
    double* fi_row);
bool student_fill_grid_bivariate(
    const scar::CopulaSpec& spec,
    std::int64_t n_obs,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi,
    double* dfi_dx);

bool copula_is_supported(const scar::CopulaSpec& spec);
bool copula_is_supported_for_ou(const scar::CopulaSpec& spec);
double copula_log_pdf_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r);
double copula_dlog_pdf_dr_unrotated(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double r);
double copula_pdf_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x);
void copula_pdf_and_grad_x(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    double x,
    double& pdf,
    double& d_pdf_dx);
void copula_prepare_grid_transform(
    const scar::CopulaSpec& spec,
    const std::vector<double>& x_grid,
    std::vector<double>& r_grid,
    std::vector<double>& dpsi_grid);
void copula_pdf_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    double* fi_row);
void copula_pdf_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    double* fi_row);
void copula_pdf_and_grad_row_precomputed(
    const scar::CopulaSpec& spec,
    double u1,
    double u2,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row);
void copula_pdf_and_grad_row_precomputed_flat(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row);
void copula_pdf_and_grad_grid_precomputed(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t n_obs,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    std::vector<double>& fi,
    std::vector<double>& dfi_dx);
double copula_h_rotated(
    const scar::CopulaSpec& spec,
    double u,
    double v,
    double r);
double copula_h_inverse_rotated(
    const scar::CopulaSpec& spec,
    double q,
    double given,
    double r);

bool build_ou_grid(
    double kappa,
    double mu,
    double nu,
    std::int64_t n_obs,
    int K,
    double grid_range,
    bool adaptive,
    int pts_per_sigma,
    int max_K,
    OuGrid& grid);
bool predictive_weights_from_phi(
    const OuGrid& grid,
    const std::vector<double>& phi,
    std::vector<double>& weights);
void copula_fi_row_on_grid(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& x_grid,
    std::vector<double>& fi_row);

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

bool standard_normal_hermite_rule(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis);
bool standard_normal_hermite_rule_with_weighted_basis(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis,
    std::vector<double>& weighted_basis);
HermiteRuleCacheInfo hermite_rule_cache_info();
void clear_hermite_rule_cache();
void set_hermite_rule_cache_limits_for_testing(
    std::size_t max_entries,
    std::size_t max_bytes);
void reset_hermite_rule_cache_limits_for_testing();
bool physicists_hermite_normal_rule(
    int order,
    std::vector<double>& nodes,
    std::vector<double>& weights);
void project_multiply(
    const std::vector<double>& coeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    std::vector<double>& out);
void project_multiply_with_grad(
    const std::vector<double>& coeff,
    const std::vector<double>& dcoeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& dfi_dx_row,
    const std::vector<double>& dx_dalpha,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    std::vector<double>& out,
    std::vector<double>& dout);
void project_multiply_with_score_grad(
    const std::vector<double>& coeff,
    const std::vector<double>& dcoeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& scores,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    int n_params,
    std::vector<double>& out,
    std::vector<double>& dout);
void local_gh_matvec(
    const std::vector<double>& z,
    double rho,
    double sigma_cond,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& v,
    std::vector<double>& out);
void local_gh_predict_matvec(
    const std::vector<double>& z,
    const std::vector<double>& trap_w,
    double rho,
    double sigma_cond,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& source,
    std::vector<double>& out_density);

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
