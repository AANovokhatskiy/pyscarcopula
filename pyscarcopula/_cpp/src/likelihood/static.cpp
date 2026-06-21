#include "scar/copula.hpp"

#include "scar/ou.hpp"
#include "scar/detail/internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace scar {
namespace {

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
    Observations u)
    : spec_(std::move(spec)),
      u_(std::move(u)),
      status_(validate(spec_, u_)) {

    if (status_ != SCAR_OK
        || (spec_.family != CopulaFamily::Gaussian
            && spec_.family != CopulaFamily::MultivariateGaussian)) {
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

StaticObjectiveResult StaticCopulaEvaluator::objective(
    double parameter,
    bool correlation_gradient_requested) const {

    StaticObjectiveResult out;
    out.status = status_;
    if (status_ != SCAR_OK || !std::isfinite(parameter)) {
        if (out.status == SCAR_OK) {
            out.status = SCAR_INVALID_PARAMETER;
        }
        out.negative_log_likelihood =
            std::numeric_limits<double>::infinity();
        return out;
    }

    double log_likelihood = 0.0;
    double gradient = 0.0;
    const int dim = expected_dimension(spec_);
    std::vector<double> precision;
    std::vector<double> corr_gradient;
    std::vector<double> corr_scores;
    std::vector<double> df_grid;
    if (
        correlation_gradient_requested
        && spec_.family == CopulaFamily::Student
    ) {
        std::size_t n_corr = 0;
        if (!scar_internal::valid_student_correlation_count(
                spec_.dim, n_corr)
            || !scar_internal::student_precision_matrix(spec_, precision)) {
            out.status = SCAR_INVALID_SIZE;
            out.negative_log_likelihood =
                std::numeric_limits<double>::infinity();
            return out;
        }
        df_grid.push_back(parameter);
        corr_gradient.assign(n_corr, 0.0);
        corr_scores.assign(n_corr, 0.0);
    }
    for (std::size_t i = 0; i < u_.size(); ++i) {
        double log_pdf = 0.0;
        double dlog = 0.0;
        const double* row = u_[i].data();

        if (spec_.family == CopulaFamily::Student) {
            if (!scar_internal::student_log_pdf_and_dlog_ddf(
                    spec_, row, parameter, static_cast<std::int64_t>(i),
                    log_pdf, dlog)) {
                out.status = SCAR_NUMERICAL_FAILURE;
            } else if (
                correlation_gradient_requested
                && !scar_internal::student_corr_score_row(
                    spec_,
                    row,
                    static_cast<std::int64_t>(i),
                    df_grid,
                    precision,
                    corr_scores.data())
            ) {
                out.status = SCAR_NUMERICAL_FAILURE;
            } else if (correlation_gradient_requested) {
                for (std::size_t p = 0; p < corr_gradient.size(); ++p) {
                    corr_gradient[p] += corr_scores[p];
                }
            }
        } else if (spec_.family == CopulaFamily::EquicorrGaussian) {
            log_pdf = scar_internal::equicorr_log_pdf(
                spec_, row, parameter, &dlog);
        } else if (spec_.family == CopulaFamily::MultivariateGaussian) {
            log_pdf = multivariate_gaussian_log_pdf(
                spec_,
                gaussian_scores_.data()
                    + i * static_cast<std::size_t>(dim));
        } else {
            double u1 = 0.0;
            double u2 = 0.0;
            scar_internal::apply_rotation(
                row[0], row[1], static_cast<int>(spec_.rotation), u1, u2);
            log_pdf = scar_internal::copula_log_pdf_unrotated(
                spec_, u1, u2, parameter);
            dlog = scar_internal::copula_dlog_pdf_dr_unrotated(
                spec_, u1, u2, parameter);
        }

        if (out.status != SCAR_OK
            || !std::isfinite(log_pdf)
            || !std::isfinite(dlog)) {
            out.status = SCAR_NUMERICAL_FAILURE;
            out.failure_index = static_cast<std::int64_t>(i);
            out.negative_log_likelihood =
                std::numeric_limits<double>::infinity();
            out.negative_gradient = 0.0;
            return out;
        }
        log_likelihood += log_pdf;
        gradient += dlog;
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
        u_.size(), -std::numeric_limits<double>::infinity());
    if (status_ != SCAR_OK || !std::isfinite(parameter)) {
        return out;
    }
    const int dim = expected_dimension(spec_);
    for (std::size_t i = 0; i < u_.size(); ++i) {
        const double* row = u_[i].data();
        if (spec_.family == CopulaFamily::Student) {
            out[i] = scar_internal::student_log_pdf(
                spec_, row, parameter, static_cast<std::int64_t>(i));
        } else if (spec_.family == CopulaFamily::EquicorrGaussian) {
            out[i] = scar_internal::equicorr_log_pdf(
                spec_, row, parameter, nullptr);
        } else if (spec_.family == CopulaFamily::MultivariateGaussian) {
            out[i] = multivariate_gaussian_log_pdf(
                spec_,
                gaussian_scores_.data()
                    + i * static_cast<std::size_t>(dim));
        } else {
            double u1 = 0.0;
            double u2 = 0.0;
            scar_internal::apply_rotation(
                row[0], row[1], static_cast<int>(spec_.rotation), u1, u2);
            out[i] = scar_internal::copula_log_pdf_unrotated(
                spec_, u1, u2, parameter);
        }
    }
    return out;
}

int StaticCopulaEvaluator::status() const noexcept {
    return status_;
}

}  // namespace scar
