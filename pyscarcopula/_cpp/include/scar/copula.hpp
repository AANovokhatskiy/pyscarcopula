#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace scar {

enum class CopulaFamily : int {
    Independent = 0,
    Clayton = 1,
    Gumbel = 2,
    Frank = 3,
    Joe = 4,
    Gaussian = 5,
    Student = 6,
    EquicorrGaussian = 7,
    MultivariateGaussian = 8,
};

enum class Rotation : int {
    R0 = 0,
    R90 = 90,
    R180 = 180,
    R270 = 270,
};

enum class Transform : int {
    Softplus = 1,
    XTanh = 2,
    GaussianTanh = 3,
};

struct CopulaSpec {
    CopulaFamily family = CopulaFamily::Clayton;
    Rotation rotation = Rotation::R0;
    Transform transform = Transform::Softplus;
    double offset = 0.0001;
    int dim = 2;
    std::vector<double> l_inv;
    double log_det = 0.0;
    std::int64_t ppf_n_obs = 0;
    std::vector<double> ppf_nodes;
    std::vector<double> ppf_table;
};

using Observations = std::vector<std::vector<double>>;

struct GridValues {
    std::vector<double> values;
    std::int64_t n_obs = 0;
    std::int64_t n_grid = 0;
};

struct GridValuesWithGrad {
    GridValues pdf;
    GridValues d_pdf_dx;
};

struct MultivariateRowsResult {
    std::vector<double> log_pdf;
    std::vector<double> dlog_dr;
    int status = 0;
    std::int64_t failure_index = -1;
};

struct MultivariateGridResult {
    GridValues pdf;
    GridValues d_pdf_dx;
    int status = 0;
    std::int64_t failure_index = -1;
};

struct ConditionalSampleResult {
    std::vector<double> values;
    std::int64_t n_rows = 0;
    std::int64_t n_free = 0;
    int status = 0;
    std::int64_t failure_index = -1;
};

struct StaticObjectiveResult {
    double negative_log_likelihood = 0.0;
    double negative_gradient = 0.0;
    std::vector<double> negative_correlation_gradient;
    int status = 0;
    std::int64_t failure_index = -1;
};

class StaticCopulaEvaluator {
public:
    StaticCopulaEvaluator(CopulaSpec spec, Observations u);

    StaticObjectiveResult objective(
        double parameter,
        bool correlation_gradient = false) const;
    std::vector<double> log_pdf_rows(double parameter) const;
    int status() const noexcept;

private:
    CopulaSpec spec_;
    Observations u_;
    std::vector<double> gaussian_scores_;
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
    std::int64_t row_offset = 0);

MultivariateGridResult multivariate_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid,
    std::int64_t row_offset = 0);

ConditionalSampleResult multivariate_gaussian_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& normal_draws,
    std::int64_t n_rows);

ConditionalSampleResult multivariate_student_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& df,
    const std::vector<double>& normal_draws,
    const std::vector<double>& chi_square_draws,
    std::int64_t n_rows);

}  // namespace scar
