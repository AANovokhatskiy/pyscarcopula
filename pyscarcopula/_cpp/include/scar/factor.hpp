#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

/// Immutable low-rank correlation operator R = D + B B^T.
///
/// The diagonal D is derived from B so diag(R) == 1. All prepared storage is
/// O(d*k + k*k); no dense d*d matrix is retained.
class FactorCorrelationOperator {
public:
    FactorCorrelationOperator(
        std::vector<double> loadings,
        std::size_t dimension,
        std::size_t rank,
        double uniqueness_min);

    std::size_t dimension() const noexcept;
    std::size_t rank() const noexcept;
    double uniqueness_min() const noexcept;
    double logdet() const noexcept;
    double condition_estimate() const noexcept;

    const std::vector<double>& loadings() const noexcept;
    const std::vector<double>& uniqueness() const noexcept;
    const std::vector<double>& inverse_uniqueness() const noexcept;
    const std::vector<double>& weighted_loadings() const noexcept;
    const std::vector<double>& cholesky_m() const noexcept;

    void matvec_rows(
        const double* values,
        std::size_t rows,
        double* output,
        int n_threads = 1) const;

    void solve_rows(
        const double* values,
        std::size_t rows,
        double* output,
        int n_threads = 1) const;

    void quadratic_forms(
        const double* values,
        std::size_t rows,
        double* output,
        int n_threads = 1) const;

    void sample_normal_inplace(
        const double* factor_draws,
        double* residual_draws,
        std::size_t rows,
        int n_threads = 1) const;

    void solve_core_inplace(double* values) const;

private:
    std::vector<double> loadings_;
    std::vector<double> uniqueness_;
    std::vector<double> inverse_uniqueness_;
    std::vector<double> weighted_loadings_;
    std::vector<double> cholesky_m_;
    std::size_t dimension_ = 0;
    std::size_t rank_ = 0;
    double uniqueness_min_ = 0.0;
    double logdet_ = 0.0;
    double condition_estimate_ = 0.0;
};

/// Result of applying the Student copula adapter to observation rows.
struct FactorStudentRowsResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_ddf;
    std::int64_t failure_index = -1;
    int n_threads_requested = 1;
    int row_parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;
};

/// Student copula log density and df derivative over a factor correlation.
///
/// `df_count` must be one or equal to `rows`. The implementation uses only
/// O(d*k + k*k) prepared model storage and O(d) workspace per active worker.
FactorStudentRowsResult factor_student_log_pdf_and_dlog_ddf(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    const double* df,
    std::size_t df_count,
    int n_threads = 1);

/// Aggregate static Student likelihood and analytical gradients with respect
/// to one common df and every factor loading. Fixed reduction blocks make the
/// result independent of the requested thread count.
struct FactorStudentJointResult {
    double log_likelihood = 0.0;
    double dlog_likelihood_ddf = 0.0;
    std::vector<double> dlog_likelihood_dloadings;
    std::int64_t failure_index = -1;
    int n_threads_requested = 1;
    int reduction_blocks = 0;
    int parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;
    std::size_t reduction_workspace_bytes = 0;
};

FactorStudentJointResult factor_student_joint_likelihood_gradient(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    double df,
    int n_threads = 1);

/// Tiled Student emission grid without an O(rows*grid*dimension) cache.
struct FactorStudentGridResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_ddf;
    std::size_t rows = 0;
    std::size_t grid_size = 0;
    std::size_t dimension_tiles = 0;
    std::int64_t failure_index = -1;
    int n_threads_requested = 1;
    int parallel_axis = 0;  ///< 0 sequential, 1 cells, 2 dimension tiles.
    int parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;
    std::size_t partial_workspace_peak_bytes = 0;
    std::uint64_t ppf_exact_values = 0;
};

FactorStudentGridResult factor_student_log_pdf_and_dlog_ddf_grid(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    const double* df_grid,
    std::size_t grid_size,
    std::size_t dimension_tile,
    int n_threads = 1);

}  // namespace scar
