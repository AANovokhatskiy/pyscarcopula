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

}  // namespace scar
