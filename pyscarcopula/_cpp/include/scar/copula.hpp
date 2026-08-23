#pragma once

#include "scar/copula/rotation.hpp"
#include "scar/copula/spec.hpp"
#include "scar/copula/transforms.hpp"
#include "scar/copula/multivariate/gaussian/conditional.hpp"
#include "scar/copula/multivariate/student/conditional.hpp"
#include "scar/copula/multivariate/student/rosenblatt.hpp"
#include "scar/core/span.hpp"
#include "scar/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace scar {

using Observations = std::vector<std::vector<double>>;

/// Flattened row-major values with explicit two-dimensional shape.
struct GridValues {
    std::vector<double> values;  ///< Row-major values.
    std::int64_t n_obs = 0;      ///< Number of observation rows.
    std::int64_t n_grid = 0;     ///< Number of grid columns.
};

/// Copula density and its derivative on the same grid.
struct GridValuesWithGrad {
    GridValues pdf;
    GridValues d_pdf_dx;
};

/// Per-row multivariate log densities and scalar-parameter derivatives.
struct MultivariateRowsResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_dr;
    int status = 0;
    std::int64_t failure_index = -1;
    std::uint64_t student_ppf_cache_values = 0;
    std::uint64_t student_ppf_exact_values = 0;
    std::uint64_t student_ppf_asymptotic_values = 0;
    std::uint64_t student_workspace_growth_events = 0;
    std::size_t student_workspace_peak_bytes = 0;
    int n_threads_requested = 1;
    int row_parallel_blocks = 0;
};

struct MultivariateGridResult {
    GridValues pdf;
    GridValues d_pdf_dx;
    int status = 0;
    std::int64_t failure_index = -1;
    std::uint64_t student_ppf_cache_values = 0;
    std::uint64_t student_ppf_exact_values = 0;
    std::uint64_t student_ppf_asymptotic_values = 0;
    std::uint64_t student_workspace_growth_events = 0;
    std::size_t student_workspace_peak_bytes = 0;
    int n_threads_requested = 1;
    int student_parallel_blocks = 0;
    int equicorr_parallel_blocks = 0;
};

/// Per-row sufficient statistics for an equicorrelation Gaussian copula.
///
/// The result owns only O(n_obs) output plus bounded tile-reduction
/// diagnostics; it never materializes a dense dimension-by-dimension matrix.
struct EquicorrPreparationResult {
    std::vector<double> sum_z;
    std::vector<double> sum_z2;
    int status = 0;
    std::int64_t failure_index = -1;
    int n_threads_requested = 1;
    int parallel_blocks = 0;
    int parallel_axis = 0;  ///< 0 sequential, 1 rows, 2 dimension tiles.
    std::size_t dimension_tiles = 0;
    std::size_t temporary_values = 0;
    std::uint64_t clipping_events = 0;
    std::uint64_t nonfinite_values = 0;
};

/// Static negative log-likelihood objective and requested gradients.
struct StaticObjectiveResult {
    double negative_log_likelihood = 0.0;
    double negative_gradient = 0.0;
    std::vector<double> negative_correlation_gradient;
    int status = 0;
    std::int64_t failure_index = -1;
    int n_threads_requested = 1;
    int parallel_blocks = 0;
};

/// Reusable evaluator for a fixed copula specification and observation set.
///
/// Construction validates and precomputes family-specific sufficient
/// statistics. Instances can evaluate multiple scalar parameters without
/// rebuilding those caches.
class StaticCopulaEvaluator {
public:
    StaticCopulaEvaluator(CopulaSpec spec, Observations u, int n_threads = 1);
    StaticCopulaEvaluator(
        CopulaSpec spec,
        std::vector<double> equicorr_sums,
        std::vector<double> equicorr_sum_squares,
        int n_threads = 1);

    StaticObjectiveResult objective(
        double parameter,
        bool correlation_gradient = false) const;
    StaticObjectiveResult gaussian_objective(
        const CopulaSpec& spec,
        bool correlation_gradient = true) const;
    StaticObjectiveResult objective_value(double parameter) const;
    std::vector<double> log_pdf_rows(double parameter) const;
    int status() const noexcept;

private:
    StaticObjectiveResult evaluate_objective(
        double parameter,
        bool parameter_gradient_requested,
        bool correlation_gradient_requested) const;
    StaticObjectiveResult evaluate_gaussian_objective(
        const CopulaSpec& spec,
        bool correlation_gradient_requested) const;
    CopulaSpec spec_;
    Observations u_;
    std::vector<double> gaussian_scores_;
    std::vector<double> equicorr_sums_;
    std::vector<double> equicorr_sum_squares_;
    std::size_t n_obs_ = 0;
    int n_threads_ = 1;
    int status_ = 0;
};

bool is_supported(const CopulaSpec& spec);

std::vector<double> copula_transform(
    const CopulaSpec& spec,
    const std::vector<double>& x);

std::vector<double> copula_inverse_transform(
    const CopulaSpec& spec,
    const std::vector<double>& r);

std::vector<double> copula_dtransform(
    const CopulaSpec& spec,
    const std::vector<double>& x);

std::vector<double> copula_tau_to_param(
    const CopulaSpec& spec,
    const std::vector<double>& tau);

std::vector<double> copula_param_to_tau(
    const CopulaSpec& spec,
    const std::vector<double>& r);

std::vector<double> copula_log_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_dlog_pdf_dr(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_h(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::pair<std::vector<double>, std::vector<double>> copula_h_pair(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_h_inverse(
    const CopulaSpec& spec,
    const Observations& q_given,
    const std::vector<double>& r);

GridValues copula_pdf_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid);

GridValuesWithGrad copula_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid);

GridValues copula_pdf_parameter_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r_grid);

GridValues copula_h_parameter_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r_grid);

MultivariateRowsResult multivariate_log_pdf_and_grad(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r,
    std::int64_t row_offset = 0,
    int n_threads = 1);

MultivariateRowsResult equicorr_log_pdf_and_grad_from_stats(
    const CopulaSpec& spec,
    DoubleView sum_z,
    DoubleView sum_z2,
    const std::vector<double>& r,
    int n_threads = 1);

MultivariateGridResult multivariate_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid,
    std::int64_t row_offset = 0,
    int n_threads = 1);

MultivariateGridResult equicorr_pdf_and_grad_grid_from_stats(
    const CopulaSpec& spec,
    DoubleView sum_z,
    DoubleView sum_z2,
    const std::vector<double>& x_grid,
    int n_threads = 1);

EquicorrPreparationResult prepare_equicorr_sufficient_statistics(
    ObservationView u,
    std::size_t dimension_tile = 16384,
    int n_threads = 1);

}  // namespace scar
