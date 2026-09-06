#pragma once

#include "scar/copula/grid_values.hpp"
#include "scar/copula/model_statistics.hpp"
#include "scar/copula/conditional_options.hpp"
#include "scar/copula/result.hpp"
#include "scar/copula/spec.hpp"
#include "scar/core/span.hpp"
#include "scar/observation.hpp"
#include "scar/static/result.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace scar {

using Observations = std::vector<std::vector<double>>;

class PreparedDynamicEmission;

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
    ~StaticCopulaEvaluator();
    StaticCopulaEvaluator(const StaticCopulaEvaluator&) = delete;
    StaticCopulaEvaluator& operator=(const StaticCopulaEvaluator&) = delete;
    StaticCopulaEvaluator(StaticCopulaEvaluator&&) noexcept;
    StaticCopulaEvaluator& operator=(StaticCopulaEvaluator&&) noexcept;

    StaticObjectiveResult objective(
        double parameter,
        bool correlation_gradient = false) const;
    StaticObjectiveResult transformed_objective(
        double optimizer_parameter) const;
    StaticObjectiveResult gaussian_objective(
        const CopulaSpec& spec,
        bool correlation_gradient = true) const;
    StaticObjectiveResult objective_value(double parameter) const;
    std::vector<double> log_pdf_rows(double parameter) const;
    Status status() const noexcept;

private:
    StaticObjectiveResult evaluate_objective(
        double parameter,
        bool parameter_gradient_requested,
        bool correlation_gradient_requested) const;
    StaticObjectiveResult evaluate_gaussian_objective(
        const CopulaSpec& spec,
        bool correlation_gradient_requested) const;
    CopulaSpec spec_;
    std::unique_ptr<PreparedDynamicEmission> emission_;
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

std::vector<double> copula_tau_to_param_capped(
    const CopulaSpec& spec,
    const std::vector<double>& tau,
    double theta_cap,
    bool has_theta_cap);

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

/// Transform caller-owned independent uniforms into pair-copula samples.
///
/// Each input row contains ``(u_first, conditional_quantile)``.  The first
/// coordinate is preserved and the second coordinate is obtained from the
/// inverse conditional distribution using the transposed pair orientation.
/// Random-number generation deliberately remains outside the computational
/// boundary.
Observations copula_sample_from_uniforms(
    const CopulaSpec& spec,
    const Observations& uniforms,
    const std::vector<double>& r);

/// Contiguous variant; input and output contain two values per observation.
void copula_sample_from_uniforms_into(
    const CopulaSpec& spec, const double* uniforms, std::size_t n,
    const std::vector<double>& r, double* output);

Observations copula_conditional_sample_from_uniforms(
    const CopulaSpec& spec,
    const std::vector<double>& uniforms,
    const std::vector<double>& r,
    int given_coordinate,
    double given_value,
    const HInverseOptions* options = nullptr);

/// Gaussian normal-draw transform or pair conditional inversion.
/// The compatibility auxiliary argument must be empty.
Observations copula_sample_from_rng_draws(
    const CopulaSpec& spec,
    const Observations& draws,
    const Observations& auxiliary,
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
