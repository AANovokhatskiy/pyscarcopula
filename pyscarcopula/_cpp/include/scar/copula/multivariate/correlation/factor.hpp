#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace scar {
struct CopulaSpec;

/// Immutable low-rank correlation operator R = D + B B^T.
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

}  // namespace scar

namespace scar::copula::multivariate::correlation {

/// Prepared factor-correlation contract, distinct from dense storage.
struct FactorCorrelation {
    std::shared_ptr<const scar::FactorCorrelationOperator> factor;
    std::size_t dimension_tile = 16384;
    double log_determinant = 0.0;
};

FactorCorrelation& factor(CopulaSpec& spec);
const FactorCorrelation& factor(const CopulaSpec& spec);

}  // namespace scar::copula::multivariate::correlation
