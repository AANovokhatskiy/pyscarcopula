#include "scar/factor.hpp"

#include "scar/detail/linalg.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace scar {
namespace {

void solve_cholesky_inplace(
    const std::vector<double>& lower,
    std::size_t dimension,
    std::vector<double>& values) {

    for (std::size_t row = 0; row < dimension; ++row) {
        double value = values[row];
        for (std::size_t column = 0; column < row; ++column) {
            value -= lower[row * dimension + column] * values[column];
        }
        values[row] = value / lower[row * dimension + row];
    }
    for (std::size_t row = dimension; row-- > 0;) {
        double value = values[row];
        for (std::size_t column = row + 1; column < dimension; ++column) {
            value -= lower[column * dimension + row] * values[column];
        }
        values[row] = value / lower[row * dimension + row];
    }
}

void require_finite_row(
    const double* values,
    std::size_t dimension) {

    for (std::size_t column = 0; column < dimension; ++column) {
        if (!std::isfinite(values[column])) {
            throw std::invalid_argument(
                "factor operator inputs must contain only finite values");
        }
    }
}

}  // namespace

FactorCorrelationOperator::FactorCorrelationOperator(
    std::vector<double> loadings,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min)
    : loadings_(std::move(loadings)),
      dimension_(dimension),
      rank_(rank),
      uniqueness_min_(uniqueness_min) {

    if (dimension_ < 2 || rank_ < 1 || rank_ >= dimension_) {
        throw std::invalid_argument(
            "factor dimensions must satisfy 1 <= rank < dimension");
    }
    if (!std::isfinite(uniqueness_min_)
        || !(uniqueness_min_ > 0.0)
        || !(uniqueness_min_ < 1.0)) {
        throw std::invalid_argument(
            "uniqueness_min must be finite and in (0, 1)");
    }

    std::size_t loading_count = 0;
    std::size_t small_count = 0;
    if (!scar_internal::checked_size_mul(
            dimension_, rank_, loading_count)
        || !scar_internal::checked_size_mul(
            rank_, rank_, small_count)
        || loadings_.size() != loading_count) {
        throw std::invalid_argument(
            "factor loading shape is not representable");
    }

    uniqueness_.assign(dimension_, 0.0);
    inverse_uniqueness_.assign(dimension_, 0.0);
    weighted_loadings_.assign(loading_count, 0.0);
    std::vector<double> matrix_m(small_count, 0.0);
    for (std::size_t diagonal = 0; diagonal < rank_; ++diagonal) {
        matrix_m[diagonal * rank_ + diagonal] = 1.0;
    }

    double diagonal_logdet = 0.0;
    for (std::size_t row = 0; row < dimension_; ++row) {
        const double* loading = loadings_.data() + row * rank_;
        double norm_squared = 0.0;
        for (std::size_t factor = 0; factor < rank_; ++factor) {
            if (!std::isfinite(loading[factor])) {
                throw std::invalid_argument(
                    "factor loadings must contain only finite values");
            }
            norm_squared += loading[factor] * loading[factor];
        }
        const double uniqueness = 1.0 - norm_squared;
        if (!std::isfinite(uniqueness)
            || uniqueness < uniqueness_min_) {
            throw std::invalid_argument(
                "factor loading row violates uniqueness_min");
        }
        uniqueness_[row] = uniqueness;
        inverse_uniqueness_[row] = 1.0 / uniqueness;
        diagonal_logdet += std::log(uniqueness);

        double* weighted = weighted_loadings_.data() + row * rank_;
        for (std::size_t factor = 0; factor < rank_; ++factor) {
            weighted[factor] =
                loading[factor] * inverse_uniqueness_[row];
        }
        for (std::size_t left = 0; left < rank_; ++left) {
            for (std::size_t right = 0; right < rank_; ++right) {
                matrix_m[left * rank_ + right] +=
                    loading[left] * weighted[right];
            }
        }
    }

    double applied_jitter = 0.0;
    if (!scar_internal::linalg::cholesky_symmetric_with_jitter(
            matrix_m.data(),
            rank_,
            cholesky_m_,
            &applied_jitter)
        || applied_jitter != 0.0) {
        throw std::invalid_argument(
            "failed to factorize factor correlation core");
    }

    double small_logdet = 0.0;
    double min_diagonal = std::numeric_limits<double>::infinity();
    double max_diagonal = 0.0;
    for (std::size_t factor = 0; factor < rank_; ++factor) {
        const double diagonal = cholesky_m_[factor * rank_ + factor];
        small_logdet += 2.0 * std::log(diagonal);
        const double matrix_diagonal =
            matrix_m[factor * rank_ + factor];
        min_diagonal = std::min(min_diagonal, matrix_diagonal);
        max_diagonal = std::max(max_diagonal, matrix_diagonal);
    }
    logdet_ = diagonal_logdet + small_logdet;
    condition_estimate_ = max_diagonal / min_diagonal;
    if (!std::isfinite(logdet_)
        || !std::isfinite(condition_estimate_)) {
        throw std::invalid_argument(
            "factor correlation preparation produced non-finite values");
    }
}

std::size_t FactorCorrelationOperator::dimension() const noexcept {
    return dimension_;
}

std::size_t FactorCorrelationOperator::rank() const noexcept {
    return rank_;
}

double FactorCorrelationOperator::uniqueness_min() const noexcept {
    return uniqueness_min_;
}

double FactorCorrelationOperator::logdet() const noexcept {
    return logdet_;
}

double FactorCorrelationOperator::condition_estimate() const noexcept {
    return condition_estimate_;
}

const std::vector<double>&
FactorCorrelationOperator::loadings() const noexcept {
    return loadings_;
}

const std::vector<double>&
FactorCorrelationOperator::uniqueness() const noexcept {
    return uniqueness_;
}

const std::vector<double>&
FactorCorrelationOperator::inverse_uniqueness() const noexcept {
    return inverse_uniqueness_;
}

const std::vector<double>&
FactorCorrelationOperator::weighted_loadings() const noexcept {
    return weighted_loadings_;
}

const std::vector<double>&
FactorCorrelationOperator::cholesky_m() const noexcept {
    return cholesky_m_;
}

void FactorCorrelationOperator::matvec_rows(
    const double* values,
    std::size_t rows,
    double* output,
    int n_threads) const {

    scar_internal::validate_thread_count(n_threads);
    const int threads = scar_internal::worker_count_for_items(
        n_threads, rows, 4);
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(rows),
        1,
        threads,
        [&](std::int64_t begin, std::int64_t end, std::size_t) {
            std::vector<double> scores(rank_, 0.0);
            for (std::int64_t row = begin; row < end; ++row) {
                const double* input =
                    values + static_cast<std::size_t>(row) * dimension_;
                double* target =
                    output + static_cast<std::size_t>(row) * dimension_;
                require_finite_row(input, dimension_);
                std::fill(scores.begin(), scores.end(), 0.0);
                for (std::size_t column = 0; column < dimension_; ++column) {
                    const double* loading =
                        loadings_.data() + column * rank_;
                    for (std::size_t factor = 0;
                         factor < rank_;
                         ++factor) {
                        scores[factor] += loading[factor] * input[column];
                    }
                }
                for (std::size_t column = 0; column < dimension_; ++column) {
                    const double* loading =
                        loadings_.data() + column * rank_;
                    double value = uniqueness_[column] * input[column];
                    for (std::size_t factor = 0;
                         factor < rank_;
                         ++factor) {
                        value += loading[factor] * scores[factor];
                    }
                    target[column] = value;
                }
            }
        });
}

void FactorCorrelationOperator::solve_rows(
    const double* values,
    std::size_t rows,
    double* output,
    int n_threads) const {

    scar_internal::validate_thread_count(n_threads);
    const int threads = scar_internal::worker_count_for_items(
        n_threads, rows, 4);
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(rows),
        1,
        threads,
        [&](std::int64_t begin, std::int64_t end, std::size_t) {
            std::vector<double> small(rank_, 0.0);
            for (std::int64_t row = begin; row < end; ++row) {
                const double* input =
                    values + static_cast<std::size_t>(row) * dimension_;
                double* target =
                    output + static_cast<std::size_t>(row) * dimension_;
                require_finite_row(input, dimension_);
                std::fill(small.begin(), small.end(), 0.0);
                for (std::size_t column = 0; column < dimension_; ++column) {
                    const double* weighted =
                        weighted_loadings_.data() + column * rank_;
                    for (std::size_t factor = 0;
                         factor < rank_;
                         ++factor) {
                        small[factor] += weighted[factor] * input[column];
                    }
                }
                solve_cholesky_inplace(cholesky_m_, rank_, small);
                for (std::size_t column = 0; column < dimension_; ++column) {
                    const double* weighted =
                        weighted_loadings_.data() + column * rank_;
                    double value =
                        inverse_uniqueness_[column] * input[column];
                    for (std::size_t factor = 0;
                         factor < rank_;
                         ++factor) {
                        value -= weighted[factor] * small[factor];
                    }
                    target[column] = value;
                }
            }
        });
}

void FactorCorrelationOperator::quadratic_forms(
    const double* values,
    std::size_t rows,
    double* output,
    int n_threads) const {

    scar_internal::validate_thread_count(n_threads);
    const int threads = scar_internal::worker_count_for_items(
        n_threads, rows, 4);
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(rows),
        1,
        threads,
        [&](std::int64_t begin, std::int64_t end, std::size_t) {
            std::vector<double> small(rank_, 0.0);
            std::vector<double> solved(rank_, 0.0);
            for (std::int64_t row = begin; row < end; ++row) {
                const double* input =
                    values + static_cast<std::size_t>(row) * dimension_;
                require_finite_row(input, dimension_);
                std::fill(small.begin(), small.end(), 0.0);
                double diagonal_term = 0.0;
                for (std::size_t column = 0; column < dimension_; ++column) {
                    const double value = input[column];
                    diagonal_term +=
                        inverse_uniqueness_[column] * value * value;
                    const double* weighted =
                        weighted_loadings_.data() + column * rank_;
                    for (std::size_t factor = 0;
                         factor < rank_;
                         ++factor) {
                        small[factor] += weighted[factor] * value;
                    }
                }
                solved = small;
                solve_cholesky_inplace(cholesky_m_, rank_, solved);
                double correction = 0.0;
                for (std::size_t factor = 0; factor < rank_; ++factor) {
                    correction += small[factor] * solved[factor];
                }
                output[static_cast<std::size_t>(row)] =
                    diagonal_term - correction;
            }
        });
}

void FactorCorrelationOperator::sample_normal_inplace(
    const double* factor_draws,
    double* residual_draws,
    std::size_t rows,
    int n_threads) const {

    scar_internal::validate_thread_count(n_threads);
    const int threads = scar_internal::worker_count_for_items(
        n_threads, rows, 4);
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(rows),
        1,
        threads,
        [&](std::int64_t begin, std::int64_t end, std::size_t) {
            for (std::int64_t row = begin; row < end; ++row) {
                const double* factors =
                    factor_draws + static_cast<std::size_t>(row) * rank_;
                double* residuals =
                    residual_draws
                    + static_cast<std::size_t>(row) * dimension_;
                require_finite_row(factors, rank_);
                require_finite_row(residuals, dimension_);
                for (std::size_t column = 0; column < dimension_; ++column) {
                    const double* loading =
                        loadings_.data() + column * rank_;
                    double value =
                        std::sqrt(uniqueness_[column]) * residuals[column];
                    for (std::size_t factor = 0;
                         factor < rank_;
                         ++factor) {
                        value += loading[factor] * factors[factor];
                    }
                    residuals[column] = value;
                }
            }
        });
}

void FactorCorrelationOperator::solve_core_inplace(
    double* values) const {

    if (values == nullptr) {
        throw std::invalid_argument(
            "factor core solve values must not be null");
    }
    for (std::size_t row = 0; row < rank_; ++row) {
        double value = values[row];
        for (std::size_t column = 0; column < row; ++column) {
            value -=
                cholesky_m_[row * rank_ + column] * values[column];
        }
        values[row] =
            value / cholesky_m_[row * rank_ + row];
    }
    for (std::size_t row = rank_; row-- > 0;) {
        double value = values[row];
        for (std::size_t column = row + 1;
             column < rank_;
             ++column) {
            value -=
                cholesky_m_[column * rank_ + row] * values[column];
        }
        values[row] =
            value / cholesky_m_[row * rank_ + row];
    }
}

}  // namespace scar
