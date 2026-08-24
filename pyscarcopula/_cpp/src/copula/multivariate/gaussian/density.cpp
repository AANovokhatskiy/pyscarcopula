#include "scar/copula/multivariate/gaussian/density.hpp"

#include "density_internal.hpp"

#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar::copula::multivariate::gaussian {

double log_pdf(
    const FactorCorrelationOperator& correlation,
    const double* scores,
    std::vector<double>& factor_projection,
    std::vector<double>& factor_solved) {

    double marginal_quad = 0.0;
    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    const std::vector<double>& inverse_uniqueness =
        correlation.inverse_uniqueness();
    const std::vector<double>& weighted_loadings =
        correlation.weighted_loadings();
    factor_projection.assign(rank, 0.0);
    double diagonal_quad = 0.0;
    for (std::size_t column = 0; column < dimension; ++column) {
        const double value = scores[column];
        marginal_quad += value * value;
        diagonal_quad +=
            inverse_uniqueness[column] * value * value;
        const double* weighted =
            weighted_loadings.data() + column * rank;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            factor_projection[factor] += weighted[factor] * value;
        }
    }
    factor_solved.assign(
        factor_projection.begin(), factor_projection.end());
    correlation.solve_core_inplace(factor_solved.data());
    double correction = 0.0;
    for (std::size_t factor = 0; factor < rank; ++factor) {
        correction +=
            factor_projection[factor] * factor_solved[factor];
    }
    const double joint_quad = diagonal_quad - correction;
    if (!std::isfinite(joint_quad) || joint_quad < -1e-10) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (
        -0.5 * correlation.logdet()
        - 0.5 * (std::max(0.0, joint_quad) - marginal_quad)
    );
}

double log_pdf(
    const correlation::DenseCorrelation& correlation,
    int dimension,
    const double* scores) {

    double marginal_quad = 0.0;
    const auto& inverse_cholesky = correlation.inverse_cholesky;
    double joint_quad = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double xi = scores[i];
        marginal_quad += xi * xi;
        double whitened = 0.0;
        for (int j = 0; j <= i; ++j) {
            whitened += inverse_cholesky[
                static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(dimension)
                + static_cast<std::size_t>(j)]
                * scores[j];
        }
        joint_quad += whitened * whitened;
    }
    return -0.5 * correlation.log_determinant
        - 0.5 * (joint_quad - marginal_quad);
}

bool accumulate_log_pdf(
    const correlation::DenseCorrelation& correlation,
    int dimension,
    const double* scores,
    std::size_t rows,
    double& log_likelihood,
    std::int64_t& failure_index) {

    const std::size_t dimension_size =
        static_cast<std::size_t>(dimension);
    log_likelihood = 0.0;
    failure_index = -1;
    for (std::size_t row = 0; row < rows; ++row) {
        const double row_log_pdf = log_pdf(
            correlation,
            dimension,
            scores + row * dimension_size);
        if (!std::isfinite(row_log_pdf)) {
            failure_index = static_cast<std::int64_t>(row);
            return false;
        }
        log_likelihood += row_log_pdf;
    }
    return true;
}

double log_pdf(
    const CopulaSpec& spec,
    const double* scores,
    std::vector<double>& factor_projection,
    std::vector<double>& factor_solved) {

    const FactorCorrelationOperator* factor_operator =
        spec.correlation_kind == CorrelationKind::Factor
            ? spec.factor_operator().get()
            : nullptr;
    if (factor_operator != nullptr) {
        return log_pdf(
            *factor_operator,
            scores,
            factor_projection,
            factor_solved);
    }
    return log_pdf(
        correlation::dense(spec), spec.dim, scores);
}

}  // namespace scar::copula::multivariate::gaussian
