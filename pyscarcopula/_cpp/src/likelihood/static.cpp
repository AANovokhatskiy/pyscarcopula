#include "scar/copula.hpp"

#include "scar/copula/rotation.hpp"
#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/copula/multivariate/gaussian/density.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/dispatch.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/factor.hpp"
#include "scar/math/normal.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace scar {
namespace {

constexpr std::size_t kExpensiveMinRows = 8;
constexpr std::size_t kExpensiveMinWork = 4096;
constexpr std::size_t kCheapMinRows = 1024;
constexpr std::size_t kCheapMinWork = 262144;

struct StaticObjectiveBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
    double log_likelihood = 0.0;
    double gradient = 0.0;
    std::vector<double> correlation_gradient;
};

bool static_parallel_worthwhile(
    const CopulaSpec& spec,
    std::size_t rows,
    int n_threads) {

    if (n_threads <= 1) {
        return false;
    }
    if (spec.family == CopulaFamily::Student
        || spec.family == CopulaFamily::MultivariateGaussian) {
        const std::size_t dim = static_cast<std::size_t>(spec.dim);
        if (
                spec.family == CopulaFamily::MultivariateGaussian
                && spec.correlation_kind == CorrelationKind::Factor
                && spec.factor_operator() != nullptr) {
            return scar_internal::grid_parallel_worthwhile(
                rows,
                dim * spec.factor_operator()->rank(),
                kExpensiveMinRows,
                kExpensiveMinWork);
        }
        return scar_internal::grid_parallel_worthwhile(
            rows, dim * dim, kExpensiveMinRows, kExpensiveMinWork);
    }
    return scar_internal::grid_parallel_worthwhile(
        rows, 1, kCheapMinRows, kCheapMinWork);
}

std::int64_t static_min_rows(const CopulaSpec& spec) {
    return static_cast<std::int64_t>(
        spec.family == CopulaFamily::Student
            || spec.family == CopulaFamily::MultivariateGaussian
        ? kExpensiveMinRows
        : kCheapMinRows);
}

bool valid_dense_factor(const CopulaSpec& spec) {
    std::size_t square = 0;
    if (spec.dim < 2
        || !scar_internal::valid_student_dimension(spec.dim, square)) {
        return false;
    }
    const auto& correlation =
        copula::multivariate::correlation::dense(spec);
    if (correlation.inverse_cholesky.size() != square
        || !std::isfinite(correlation.log_determinant)) {
        return false;
    }
    for (int i = 0; i < spec.dim; ++i) {
        for (int j = 0; j < spec.dim; ++j) {
            const double value = correlation.inverse_cholesky[
                static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(spec.dim)
                + static_cast<std::size_t>(j)];
            if (!std::isfinite(value)
                || (j > i && std::abs(value) > 1e-14)) {
                return false;
            }
        }
    }
    return true;
}

bool valid_multivariate_gaussian_correlation(const CopulaSpec& spec) {
    if (spec.correlation_kind == CorrelationKind::Factor) {
        const auto& factor = spec.factor_operator();
        return (
            spec.dim >= 2
            && factor != nullptr
            && factor->dimension()
                == static_cast<std::size_t>(spec.dim)
            && std::isfinite(factor->logdet())
            && std::isfinite(spec.dense_log_determinant())
        );
    }
    return valid_dense_factor(spec);
}

int validate(const CopulaSpec& spec, const Observations& u) {
    if (u.empty()) {
        return SCAR_INVALID_SIZE;
    }
    if (spec.family == CopulaFamily::EquicorrGaussian) {
        if (spec.rotation != Rotation::R0
            || spec.transform != Transform::GaussianTanh
            || spec.dim < 2) {
            return SCAR_INVALID_FAMILY;
        }
    } else if (spec.family == CopulaFamily::MultivariateGaussian) {
        if (
                spec.rotation != Rotation::R0
                || !valid_multivariate_gaussian_correlation(spec)) {
            return SCAR_INVALID_FAMILY;
        }
    } else if (!scar_internal::copula_is_supported(spec)) {
        return SCAR_INVALID_FAMILY;
    }

    const int dim = spec.model_descriptor().expected_dimension();
    for (const auto& row : u) {
        if (row.size() != static_cast<std::size_t>(dim)) {
            return SCAR_INVALID_SIZE;
        }
        if (!std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            return SCAR_INVALID_PARAMETER;
        }
    }
    return SCAR_OK;
}

}  // namespace

StaticCopulaEvaluator::StaticCopulaEvaluator(
    CopulaSpec spec,
    Observations u,
    int n_threads)
    : spec_(std::move(spec)),
      u_(std::move(u)),
      n_obs_(u_.size()),
      n_threads_(n_threads),
      status_(validate(spec_, u_)) {

    if (!scar_internal::valid_thread_count(n_threads_)) {
        status_ = SCAR_INVALID_PARAMETER;
    }
    if (status_ != SCAR_OK) {
        return;
    }
    if (spec_.family == CopulaFamily::EquicorrGaussian) {
        equicorr_sums_.resize(u_.size(), 0.0);
        equicorr_sum_squares_.resize(u_.size(), 0.0);
        for (std::size_t i = 0; i < u_.size(); ++i) {
            scar_internal::EquicorrStats stats;
            if (!scar_internal::equicorr_sufficient_statistics(
                    spec_, u_[i].data(), stats)) {
                status_ = SCAR_NUMERICAL_FAILURE;
                equicorr_sums_.clear();
                equicorr_sum_squares_.clear();
                return;
            }
            equicorr_sums_[i] = stats.sum;
            equicorr_sum_squares_[i] = stats.sum_squares;
        }
        return;
    }
    if (spec_.family != CopulaFamily::Gaussian
        && spec_.family != CopulaFamily::MultivariateGaussian) {
        return;
    }
    const int dim = spec_.model_descriptor().expected_dimension();
    std::size_t score_count = 0;
    if (!scar_internal::checked_size_mul(
            u_.size(), static_cast<std::size_t>(dim), score_count)) {
        status_ = SCAR_INVALID_SIZE;
        return;
    }
    gaussian_scores_.resize(score_count, 0.0);
    for (std::size_t i = 0; i < u_.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            gaussian_scores_[
                i * static_cast<std::size_t>(dim)
                + static_cast<std::size_t>(j)] =
                scar::math::normal_quantile(
                    scar_internal::clip_pseudo_observation(
                        u_[i][static_cast<std::size_t>(j)]));
        }
    }
}

StaticCopulaEvaluator::StaticCopulaEvaluator(
    CopulaSpec spec,
    std::vector<double> equicorr_sums,
    std::vector<double> equicorr_sum_squares,
    int n_threads)
    : spec_(std::move(spec)),
      equicorr_sums_(std::move(equicorr_sums)),
      equicorr_sum_squares_(std::move(equicorr_sum_squares)),
      n_obs_(equicorr_sums_.size()),
      n_threads_(n_threads) {

    if (spec_.family != CopulaFamily::EquicorrGaussian
        || spec_.rotation != Rotation::R0
        || spec_.transform != Transform::GaussianTanh
        || spec_.dim < 2) {
        status_ = SCAR_INVALID_FAMILY;
        return;
    }
    if (!scar_internal::valid_thread_count(n_threads_)) {
        status_ = SCAR_INVALID_PARAMETER;
        return;
    }
    if (n_obs_ == 0
        || equicorr_sum_squares_.size() != n_obs_) {
        status_ = SCAR_INVALID_SIZE;
        return;
    }
    for (std::size_t row = 0; row < n_obs_; ++row) {
        if (!std::isfinite(equicorr_sums_[row])
            || !std::isfinite(equicorr_sum_squares_[row])
            || equicorr_sum_squares_[row] < 0.0) {
            status_ = SCAR_INVALID_PARAMETER;
            return;
        }
    }
    status_ = SCAR_OK;
}

StaticObjectiveResult StaticCopulaEvaluator::objective(
    double parameter,
    bool correlation_gradient_requested) const {

    return evaluate_objective(
        parameter, true, correlation_gradient_requested);
}

StaticObjectiveResult StaticCopulaEvaluator::objective_value(
    double parameter) const {

    return evaluate_objective(parameter, false, false);
}

StaticObjectiveResult StaticCopulaEvaluator::gaussian_objective(
    const CopulaSpec& spec,
    bool correlation_gradient) const {

    return evaluate_gaussian_objective(spec, correlation_gradient);
}

StaticObjectiveResult StaticCopulaEvaluator::evaluate_gaussian_objective(
    const CopulaSpec& spec,
    bool correlation_gradient_requested) const {

    StaticObjectiveResult out;
    out.status = status_;
    out.n_threads_requested = n_threads_;
    const std::size_t dim = static_cast<std::size_t>(spec.dim);
    std::size_t square = 0;
    std::size_t n_corr = 0;
    std::size_t score_count = 0;
    if (out.status != SCAR_OK
        || spec_.family != CopulaFamily::MultivariateGaussian
        || spec.family != CopulaFamily::MultivariateGaussian
        || spec.dim != spec_.dim
        || !scar_internal::valid_student_dimension(spec.dim, square)
        || !scar_internal::checked_size_mul(n_obs_, dim, score_count)
        || gaussian_scores_.size() != score_count
        || !valid_multivariate_gaussian_correlation(spec)
        || (correlation_gradient_requested
            && spec.correlation_kind == CorrelationKind::Factor)
        || (correlation_gradient_requested
            && !scar_internal::valid_student_correlation_count(
                spec.dim, n_corr))) {
        out.status = out.status == SCAR_OK
            ? SCAR_INVALID_PARAMETER : out.status;
        out.negative_log_likelihood =
            std::numeric_limits<double>::infinity();
        return out;
    }

    std::vector<double> precision;
    if (correlation_gradient_requested
        && !scar_internal::student_precision_matrix(spec, precision)) {
        out.status = SCAR_INVALID_SIZE;
        out.negative_log_likelihood =
            std::numeric_limits<double>::infinity();
        return out;
    }
    const std::vector<double>* inverse_cholesky =
        correlation_gradient_requested
            ? &copula::multivariate::correlation::dense(spec)
                .inverse_cholesky
            : nullptr;
    const auto* dense_correlation =
        spec.correlation_kind == CorrelationKind::DenseCholesky
            ? &copula::multivariate::correlation::dense(spec)
            : nullptr;
    const FactorCorrelationOperator* factor_correlation =
        spec.correlation_kind == CorrelationKind::Factor
            ? spec.factor_operator().get()
            : nullptr;

    const bool use_threads = static_parallel_worthwhile(
        spec, n_obs_, n_threads_);
    std::vector<StaticObjectiveBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads_ : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(n_obs_),
        static_min_rows(spec),
        use_threads ? n_threads_ : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            StaticObjectiveBlockResult& block_result = block_results[block];
            block_result.ran = true;
            block_result.correlation_gradient.assign(n_corr, 0.0);
            std::vector<double> factor_projection;
            std::vector<double> factor_solved;
            std::vector<double> whitened(dim, 0.0);
            std::vector<double> precision_score(dim, 0.0);
            if (dense_correlation != nullptr
                && !correlation_gradient_requested) {
                std::int64_t relative_failure = -1;
                if (!copula::multivariate::gaussian::accumulate_log_pdf(
                        *dense_correlation,
                        spec.dim,
                        gaussian_scores_.data()
                            + static_cast<std::size_t>(begin) * dim,
                        static_cast<std::size_t>(end - begin),
                        block_result.log_likelihood,
                        relative_failure)) {
                    block_result.failure_index = begin + relative_failure;
                }
                return;
            }
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const double* scores = gaussian_scores_.data()
                    + static_cast<std::size_t>(row_index) * dim;
                const double log_pdf = dense_correlation != nullptr
                    ? copula::multivariate::gaussian::log_pdf(
                        *dense_correlation, spec.dim, scores)
                    : copula::multivariate::gaussian::log_pdf(
                        *factor_correlation,
                        scores,
                        factor_projection,
                        factor_solved);
                if (!std::isfinite(log_pdf)) {
                    block_result.failure_index = row_index;
                    return;
                }
                block_result.log_likelihood += log_pdf;
                if (!correlation_gradient_requested) {
                    continue;
                }
                for (int i = 0; i < spec.dim; ++i) {
                    double value = 0.0;
                    for (int j = 0; j <= i; ++j) {
                        value += (*inverse_cholesky)[
                            static_cast<std::size_t>(i) * dim
                            + static_cast<std::size_t>(j)]
                            * scores[static_cast<std::size_t>(j)];
                    }
                    whitened[static_cast<std::size_t>(i)] = value;
                }
                for (int i = 0; i < spec.dim; ++i) {
                    double value = 0.0;
                    for (int j = i; j < spec.dim; ++j) {
                        value += (*inverse_cholesky)[
                            static_cast<std::size_t>(j) * dim
                            + static_cast<std::size_t>(i)]
                            * whitened[static_cast<std::size_t>(j)];
                    }
                    precision_score[static_cast<std::size_t>(i)] = value;
                }
                std::size_t corr_index = 0;
                for (int i = 1; i < spec.dim; ++i) {
                    for (int j = 0; j < i; ++j) {
                        block_result.correlation_gradient[corr_index] +=
                            precision_score[static_cast<std::size_t>(i)]
                            * precision_score[static_cast<std::size_t>(j)]
                            - precision[
                                static_cast<std::size_t>(i) * dim
                                + static_cast<std::size_t>(j)];
                        ++corr_index;
                    }
                }
            }
        });

    double log_likelihood = 0.0;
    std::vector<double> correlation_gradient(n_corr, 0.0);
    for (const StaticObjectiveBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure_index < 0
                || block.failure_index < out.failure_index)) {
            out.failure_index = block.failure_index;
        }
        log_likelihood += block.log_likelihood;
        for (std::size_t index = 0; index < n_corr; ++index) {
            correlation_gradient[index] +=
                block.correlation_gradient[index];
        }
    }
    if (out.failure_index >= 0) {
        out.status = SCAR_NUMERICAL_FAILURE;
        out.negative_log_likelihood =
            std::numeric_limits<double>::infinity();
        return out;
    }
    out.negative_log_likelihood = -log_likelihood;
    out.negative_gradient = 0.0;
    out.negative_correlation_gradient.resize(n_corr);
    for (std::size_t index = 0; index < n_corr; ++index) {
        out.negative_correlation_gradient[index] =
            -correlation_gradient[index];
    }
    return out;
}

StaticObjectiveResult StaticCopulaEvaluator::evaluate_objective(
    double parameter,
    bool parameter_gradient_requested,
    bool correlation_gradient_requested) const {

    if (spec_.family == CopulaFamily::MultivariateGaussian) {
        if (!std::isfinite(parameter)) {
            StaticObjectiveResult invalid;
            invalid.status = SCAR_INVALID_PARAMETER;
            invalid.n_threads_requested = n_threads_;
            invalid.negative_log_likelihood =
                std::numeric_limits<double>::infinity();
            return invalid;
        }
        return evaluate_gaussian_objective(
            spec_, correlation_gradient_requested);
    }

    StaticObjectiveResult out;
    out.status = status_;
    out.n_threads_requested = n_threads_;
    if (status_ != SCAR_OK || !std::isfinite(parameter)) {
        if (out.status == SCAR_OK) {
            out.status = SCAR_INVALID_PARAMETER;
        }
        out.negative_log_likelihood =
            std::numeric_limits<double>::infinity();
        return out;
    }

    const int dim = spec_.model_descriptor().expected_dimension();
    std::vector<double> precision;
    std::vector<double> df_grid;
    std::size_t n_corr = 0;
    if (
        correlation_gradient_requested
        && spec_.family == CopulaFamily::Student
    ) {
        if (!scar_internal::valid_student_correlation_count(
                spec_.dim, n_corr)
            || !scar_internal::student_precision_matrix(spec_, precision)) {
            out.status = SCAR_INVALID_SIZE;
            out.negative_log_likelihood =
                std::numeric_limits<double>::infinity();
            return out;
        }
        df_grid.push_back(parameter);
    }

    const bool use_threads = static_parallel_worthwhile(
        spec_, n_obs_, n_threads_);
    std::vector<StaticObjectiveBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads_ : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(n_obs_),
        static_min_rows(spec_),
        use_threads ? n_threads_ : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            StaticObjectiveBlockResult& block_result = block_results[block];
            block_result.ran = true;
            block_result.correlation_gradient.assign(n_corr, 0.0);
            std::vector<double> corr_scores(n_corr, 0.0);
            scar_internal::StudentWorkspace student_workspace;
            std::vector<double> gaussian_factor_projection;
            std::vector<double> gaussian_factor_solved;
            if (spec_.family == CopulaFamily::Student) {
                student_workspace.reserve_x(static_cast<std::size_t>(dim));
                student_workspace.reserve_dx_ddf(
                    static_cast<std::size_t>(dim));
            }
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t i =
                    static_cast<std::size_t>(row_index);
                double log_pdf = 0.0;
                double dlog = 0.0;
                bool ok = true;
                const double* row =
                    spec_.family == CopulaFamily::EquicorrGaussian
                    ? nullptr
                    : u_[i].data();

                if (spec_.family == CopulaFamily::Student) {
                    if (!parameter_gradient_requested) {
                        log_pdf = scar_internal::student_log_pdf(
                            spec_, row, parameter, row_index,
                            student_workspace);
                    } else {
                        ok = scar_internal::student_log_pdf_and_dlog_ddf(
                            spec_, row, parameter, row_index,
                            log_pdf, dlog, student_workspace);
                    }
                    if (ok && correlation_gradient_requested) {
                        ok = scar_internal::student_corr_score_row(
                            spec_,
                            row,
                            row_index,
                            df_grid,
                            precision,
                            corr_scores.data());
                        if (ok) {
                            for (std::size_t p = 0; p < n_corr; ++p) {
                                block_result.correlation_gradient[p] +=
                                    corr_scores[p];
                            }
                        }
                    }
                } else if (
                    spec_.family == CopulaFamily::EquicorrGaussian) {
                    const scar_internal::EquicorrStats stats{
                        equicorr_sums_[i], equicorr_sum_squares_[i]};
                    log_pdf = scar_internal::equicorr_log_pdf_from_stats(
                        spec_, stats, parameter,
                        parameter_gradient_requested ? &dlog : nullptr);
                } else if (
                    spec_.family == CopulaFamily::MultivariateGaussian) {
                    log_pdf = copula::multivariate::gaussian::log_pdf(
                        spec_,
                        gaussian_scores_.data()
                            + i * static_cast<std::size_t>(dim),
                        gaussian_factor_projection,
                        gaussian_factor_solved);
                } else {
                    double u1 = 0.0;
                    double u2 = 0.0;
                    scar::copula::apply_rotation(
                        row[0], row[1],
                        static_cast<int>(spec_.rotation), u1, u2);
                    log_pdf = scar_internal::copula_log_pdf_unrotated(
                        spec_, u1, u2, parameter);
                    if (parameter_gradient_requested) {
                        dlog = scar_internal::copula_dlog_pdf_dr_unrotated(
                            spec_, u1, u2, parameter);
                    }
                }

                if (!ok
                    || !std::isfinite(log_pdf)
                    || (parameter_gradient_requested
                        && !std::isfinite(dlog))) {
                    block_result.failure_index = row_index;
                    return;
                }
                block_result.log_likelihood += log_pdf;
                if (parameter_gradient_requested) {
                    block_result.gradient += dlog;
                }
            }
        });

    std::vector<double> corr_gradient(n_corr, 0.0);
    double log_likelihood = 0.0;
    double gradient = 0.0;
    for (const StaticObjectiveBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure_index < 0
                || block.failure_index < out.failure_index)) {
            out.failure_index = block.failure_index;
        }
        log_likelihood += block.log_likelihood;
        gradient += block.gradient;
        for (std::size_t p = 0; p < n_corr; ++p) {
            corr_gradient[p] += block.correlation_gradient[p];
        }
    }
    if (out.failure_index >= 0) {
        out.status = SCAR_NUMERICAL_FAILURE;
        out.negative_log_likelihood =
            std::numeric_limits<double>::infinity();
        out.negative_gradient = 0.0;
        return out;
    }
    out.negative_log_likelihood = -log_likelihood;
    out.negative_gradient = -gradient;
    out.negative_correlation_gradient.resize(corr_gradient.size());
    for (std::size_t p = 0; p < corr_gradient.size(); ++p) {
        out.negative_correlation_gradient[p] = -corr_gradient[p];
    }
    return out;
}

std::vector<double> StaticCopulaEvaluator::log_pdf_rows(
    double parameter) const {

    std::vector<double> out(
        n_obs_, -std::numeric_limits<double>::infinity());
    if (status_ != SCAR_OK || !std::isfinite(parameter)) {
        return out;
    }
    const int dim = spec_.model_descriptor().expected_dimension();
    const bool use_threads = static_parallel_worthwhile(
        spec_, n_obs_, n_threads_);
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(n_obs_),
        static_min_rows(spec_),
        use_threads ? n_threads_ : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t) {
            scar_internal::StudentWorkspace student_workspace;
            std::vector<double> gaussian_factor_projection;
            std::vector<double> gaussian_factor_solved;
            if (spec_.family == CopulaFamily::Student) {
                student_workspace.reserve_x(
                    static_cast<std::size_t>(dim));
            }
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t i =
                    static_cast<std::size_t>(row_index);
                const double* row =
                    spec_.family == CopulaFamily::EquicorrGaussian
                    ? nullptr
                    : u_[i].data();
                if (spec_.family == CopulaFamily::Student) {
                    out[i] = scar_internal::student_log_pdf(
                        spec_, row, parameter, row_index,
                        student_workspace);
                } else if (
                    spec_.family == CopulaFamily::EquicorrGaussian) {
                    const scar_internal::EquicorrStats stats{
                        equicorr_sums_[i], equicorr_sum_squares_[i]};
                    out[i] = scar_internal::equicorr_log_pdf_from_stats(
                        spec_, stats, parameter, nullptr);
                } else if (
                    spec_.family == CopulaFamily::MultivariateGaussian) {
                    out[i] = copula::multivariate::gaussian::log_pdf(
                        spec_,
                        gaussian_scores_.data()
                            + i * static_cast<std::size_t>(dim),
                        gaussian_factor_projection,
                        gaussian_factor_solved);
                } else {
                    double u1 = 0.0;
                    double u2 = 0.0;
                    scar::copula::apply_rotation(
                        row[0], row[1],
                        static_cast<int>(spec_.rotation), u1, u2);
                    out[i] = scar_internal::copula_log_pdf_unrotated(
                        spec_, u1, u2, parameter);
                }
            }
        });
    return out;
}

int StaticCopulaEvaluator::status() const noexcept {
    return status_;
}

}  // namespace scar
