#include "scar/copula.hpp"

#include "scar/detail/copula.hpp"
#include "scar/detail/parallel.hpp"
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

int expected_dimension(const CopulaSpec& spec) {
    if (spec.family == CopulaFamily::Student
        || spec.family == CopulaFamily::EquicorrGaussian
        || spec.family == CopulaFamily::MultivariateGaussian) {
        return spec.dim;
    }
    return 2;
}

bool valid_factor(const CopulaSpec& spec) {
    std::size_t square = 0;
    if (spec.dim < 2
        || !scar_internal::valid_student_dimension(spec.dim, square)
        || spec.l_inv.size() != square
        || !std::isfinite(spec.log_det)) {
        return false;
    }
    for (int i = 0; i < spec.dim; ++i) {
        for (int j = 0; j < spec.dim; ++j) {
            const double value = spec.l_inv[
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
        if (spec.rotation != Rotation::R0 || !valid_factor(spec)) {
            return SCAR_INVALID_FAMILY;
        }
    } else if (!scar_internal::copula_is_supported(spec)) {
        return SCAR_INVALID_FAMILY;
    }

    const int dim = expected_dimension(spec);
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

double multivariate_gaussian_log_pdf(
    const CopulaSpec& spec,
    const double* scores) {

    double marginal_quad = 0.0;
    double joint_quad = 0.0;
    for (int i = 0; i < spec.dim; ++i) {
        const double xi = scores[i];
        marginal_quad += xi * xi;
        double whitened = 0.0;
        for (int j = 0; j <= i; ++j) {
            whitened += spec.l_inv[
                static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(spec.dim)
                + static_cast<std::size_t>(j)]
                * scores[j];
        }
        joint_quad += whitened * whitened;
    }
    return -0.5 * spec.log_det - 0.5 * (joint_quad - marginal_quad);
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

    if (n_threads_ < 1 || n_threads_ > 256) {
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
    const int dim = expected_dimension(spec_);
    gaussian_scores_.resize(
        u_.size() * static_cast<std::size_t>(dim), 0.0);
    for (std::size_t i = 0; i < u_.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            gaussian_scores_[
                i * static_cast<std::size_t>(dim)
                + static_cast<std::size_t>(j)] =
                scar_internal::normal_quantile(
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
    if (n_threads_ < 1 || n_threads_ > 256) {
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

StaticObjectiveResult StaticCopulaEvaluator::evaluate_objective(
    double parameter,
    bool parameter_gradient_requested,
    bool correlation_gradient_requested) const {

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

    const int dim = expected_dimension(spec_);
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
                    log_pdf = multivariate_gaussian_log_pdf(
                        spec_,
                        gaussian_scores_.data()
                            + i * static_cast<std::size_t>(dim));
                } else {
                    double u1 = 0.0;
                    double u2 = 0.0;
                    scar_internal::apply_rotation(
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
    const int dim = expected_dimension(spec_);
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
                    out[i] = multivariate_gaussian_log_pdf(
                        spec_,
                        gaussian_scores_.data()
                            + i * static_cast<std::size_t>(dim));
                } else {
                    double u1 = 0.0;
                    double u2 = 0.0;
                    scar_internal::apply_rotation(
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
