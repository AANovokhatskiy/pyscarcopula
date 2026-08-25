#pragma once

#include "scar/copula/multivariate/correlation/factor.hpp"
#include "scar/core/result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

struct FactorStudentRowsResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_ddf;
    double log_likelihood = 0.0;
    double dlog_likelihood_ddf = 0.0;
    double negative_log_likelihood = 0.0;
    double dnegative_log_likelihood_ddf = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int row_parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

FactorStudentRowsResult factor_student_log_pdf_and_dlog_ddf(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    const double* df,
    std::size_t df_count,
    int n_threads = 1);

struct FactorStudentJointResult {
    double log_likelihood = 0.0;
    double dlog_likelihood_ddf = 0.0;
    std::vector<double> dlog_likelihood_dloadings;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int reduction_blocks = 0;
    int parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;
    std::size_t reduction_workspace_bytes = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

FactorStudentJointResult factor_student_joint_likelihood_gradient(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    double df,
    int n_threads = 1);

struct FactorStudentPenalizedObjectiveResult {
    double objective = 0.0;
    double log_likelihood = 0.0;
    double condition_estimate = 0.0;
    std::vector<double> gradient;
    std::vector<double> loadings;
    Status status = Status::Ok;
    FailureContext failure{};
    int n_threads_requested = 1;
    int reduction_blocks = 0;
    int parallel_blocks = 0;
    std::size_t worker_workspace_peak_bytes = 0;
    std::size_t reduction_workspace_bytes = 0;

    bool is_ok() const noexcept {
        return ok(status);
    }
};

FactorStudentPenalizedObjectiveResult
factor_student_penalized_parameterized_objective_gradient(
    const double* observations,
    std::size_t rows,
    double df,
    const double* parameters,
    std::size_t parameter_count,
    const double* free_rows,
    const double* free_columns,
    const double* diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    double max_norm,
    double uniqueness_min,
    double condition_max,
    double penalty,
    int n_threads = 1);

}  // namespace scar
