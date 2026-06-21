#include "scar/gas.hpp"

#include "scar/detail/copula.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace scar {
namespace {

struct RowEvaluation {
    double r = 0.0;
    double log_likelihood = 0.0;
    double score = 0.0;
    int status = SCAR_OK;
};

bool valid_params(const GasParams& params) {
    return std::isfinite(params.omega)
        && std::isfinite(params.gamma)
        && std::isfinite(params.beta);
}

bool valid_config(const GasConfig& config) {
    return (config.scaling == GasScaling::Unit
            || config.scaling == GasScaling::Fisher)
        && std::isfinite(config.score_eps)
        && config.score_eps > 0.0
        && std::isfinite(config.g_clip)
        && config.g_clip > 0.0
        && std::isfinite(config.score_clip)
        && config.score_clip > 0.0
        && std::isfinite(config.fisher_floor)
        && config.fisher_floor > 0.0
        && std::isfinite(config.stationary_beta_tol)
        && config.stationary_beta_tol >= 0.0
        && config.stationary_beta_tol < 1.0;
}

int validate_copula(const CopulaSpec& copula) {
    if (copula.family == CopulaFamily::Student) {
        return scar_internal::copula_is_supported(copula)
            ? SCAR_OK
            : SCAR_INVALID_FAMILY;
    }
    if (copula.family == CopulaFamily::EquicorrGaussian) {
        return copula.rotation == Rotation::R0
                && copula.transform == Transform::GaussianTanh
                && copula.dim >= 2
            ? SCAR_OK
            : SCAR_INVALID_FAMILY;
    }
    if (copula.dim != 2) {
        return SCAR_INVALID_FAMILY;
    }
    if (!scar_internal::is_valid_rotation(
            static_cast<int>(copula.rotation))) {
        return SCAR_INVALID_ROTATION;
    }
    if (copula.family == CopulaFamily::Gaussian
        && copula.transform != Transform::GaussianTanh) {
        return SCAR_INVALID_TRANSFORM;
    }
    if (copula.family != CopulaFamily::Independent
        && copula.family != CopulaFamily::Gaussian
        && copula.transform != Transform::Softplus
        && copula.transform != Transform::XTanh) {
        return SCAR_INVALID_TRANSFORM;
    }
    if (!scar_internal::copula_is_supported(copula)) {
        return SCAR_INVALID_FAMILY;
    }
    return SCAR_OK;
}

int expected_dimension(const CopulaSpec& copula) {
    if (copula.family == CopulaFamily::Student
        || copula.family == CopulaFamily::EquicorrGaussian) {
        return copula.dim;
    }
    return 2;
}

int validate_inputs(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) {

    if (!valid_params(params) || !valid_config(config)) {
        return SCAR_INVALID_PARAMETER;
    }
    const int copula_status = validate_copula(copula);
    if (copula_status != SCAR_OK) {
        return copula_status;
    }
    const int expected_dim = expected_dimension(copula);
    if (u.dim != expected_dim || u.n_obs == 0) {
        return SCAR_INVALID_SIZE;
    }
    if (u.values == nullptr) {
        return SCAR_NULL_POINTER;
    }
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.n_obs, static_cast<std::size_t>(u.dim), value_count)) {
        return SCAR_INVALID_SIZE;
    }
    for (std::size_t index = 0; index < value_count; ++index) {
        if (!std::isfinite(u.values[index])) {
            return SCAR_INVALID_PARAMETER;
        }
    }
    return SCAR_OK;
}

double initial_g(
    const GasParams& params,
    const GasConfig& config) {

    if (std::abs(params.beta) < 1.0 - config.stationary_beta_tol) {
        return params.omega / (1.0 - params.beta);
    }
    return params.omega;
}

double gas_transform(const CopulaSpec& copula, double g) {
    if (copula.family == CopulaFamily::Independent) {
        return 0.0;
    }
    if (copula.family == CopulaFamily::EquicorrGaussian) {
        return scar_internal::equicorr_transform(copula, g);
    }
    return scar_internal::copula_transform(copula, g);
}

double gas_dtransform(const CopulaSpec& copula, double g) {
    if (copula.family == CopulaFamily::Independent) {
        return 0.0;
    }
    if (copula.family == CopulaFamily::EquicorrGaussian) {
        return scar_internal::equicorr_dtransform(copula, g);
    }
    return scar_internal::copula_dtransform(copula, g);
}

double log_pdf_at_g(
    const CopulaSpec& copula,
    const double* row,
    std::int64_t row_index,
    double g) {

    const double r = gas_transform(copula, g);
    if (copula.family == CopulaFamily::Student) {
        return scar_internal::student_log_pdf(
            copula, row, r, row_index);
    }
    if (copula.family == CopulaFamily::EquicorrGaussian) {
        return scar_internal::equicorr_log_pdf(
            copula, row, r, nullptr);
    }
    double v1 = 0.0;
    double v2 = 0.0;
    scar_internal::apply_rotation(
        row[0], row[1], static_cast<int>(copula.rotation), v1, v2);
    return scar_internal::copula_log_pdf_unrotated(
        copula, v1, v2, r);
}

RowEvaluation evaluate_row(
    const CopulaSpec& copula,
    const double* row,
    std::int64_t row_index,
    double g,
    const GasConfig& config,
    bool need_score) {

    RowEvaluation out;
    out.r = gas_transform(copula, g);
    if (!std::isfinite(out.r)) {
        out.status = SCAR_NUMERICAL_FAILURE;
        return out;
    }

    double dlog_dr = std::numeric_limits<double>::quiet_NaN();
    if (copula.family == CopulaFamily::Student) {
        if (need_score && config.scaling == GasScaling::Unit) {
            if (!scar_internal::student_log_pdf_and_dlog_ddf(
                    copula,
                    row,
                    out.r,
                    row_index,
                    out.log_likelihood,
                    dlog_dr)) {
                out.status = SCAR_NUMERICAL_FAILURE;
                return out;
            }
        } else {
            out.log_likelihood = scar_internal::student_log_pdf(
                copula, row, out.r, row_index);
        }
    } else if (copula.family == CopulaFamily::EquicorrGaussian) {
        out.log_likelihood = scar_internal::equicorr_log_pdf(
            copula,
            row,
            out.r,
            need_score && config.scaling == GasScaling::Unit
                ? &dlog_dr
                : nullptr);
    } else {
        double v1 = 0.0;
        double v2 = 0.0;
        scar_internal::apply_rotation(
            row[0], row[1], static_cast<int>(copula.rotation), v1, v2);
        out.log_likelihood = scar_internal::copula_log_pdf_unrotated(
            copula, v1, v2, out.r);
        if (need_score && config.scaling == GasScaling::Unit) {
            dlog_dr = scar_internal::copula_dlog_pdf_dr_unrotated(
                copula, v1, v2, out.r);
        }
    }
    if (!std::isfinite(out.log_likelihood)) {
        out.status = SCAR_NUMERICAL_FAILURE;
        return out;
    }
    if (!need_score) {
        return out;
    }

    if (config.scaling == GasScaling::Unit) {
        out.score = dlog_dr * gas_dtransform(copula, g);
    } else {
        const double ll_plus = log_pdf_at_g(
            copula, row, row_index, g + config.score_eps);
        const double ll_minus = log_pdf_at_g(
            copula, row, row_index, g - config.score_eps);
        if (!std::isfinite(ll_plus) || !std::isfinite(ll_minus)) {
            out.status = SCAR_NUMERICAL_FAILURE;
            return out;
        }
        const double score_denominator = 2.0 * config.score_eps;
        const double curvature_denominator =
            config.score_eps * config.score_eps;
        const double gradient =
            (ll_plus - ll_minus) / score_denominator;
        const double second_derivative =
            (ll_plus - 2.0 * out.log_likelihood + ll_minus)
            / curvature_denominator;
        const double fisher =
            std::max(-second_derivative, config.fisher_floor);
        out.score = gradient / fisher;
    }

    if (!std::isfinite(out.score)) {
        out.status = SCAR_NUMERICAL_FAILURE;
        return out;
    }
    out.score = std::clamp(
        out.score, -config.score_clip, config.score_clip);
    return out;
}

double next_g(
    const GasParams& params,
    const GasConfig& config,
    double g,
    double score) {

    return std::clamp(
        params.omega + params.beta * g + params.gamma * score,
        -config.g_clip,
        config.g_clip);
}

template <typename Result>
void set_failure(Result& result, int status, std::int64_t index) {
    result.status = status;
    result.failure_index = index;
}

GasLogLikResult run_log_likelihood(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) {

    GasLogLikResult out;
    const int status = validate_inputs(params, copula, u, config);
    if (status != SCAR_OK) {
        set_failure(out, status, -1);
        return out;
    }

    double g = initial_g(params, config);
    if (!std::isfinite(g)) {
        set_failure(out, SCAR_NUMERICAL_FAILURE, -1);
        return out;
    }
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        const double* row =
            u.values + static_cast<std::size_t>(u.dim) * t;
        const bool need_score = t + 1 < u.n_obs;
        const RowEvaluation evaluation = evaluate_row(
            copula,
            row,
            static_cast<std::int64_t>(t),
            g,
            config,
            need_score);
        if (evaluation.status != SCAR_OK) {
            set_failure(
                out, evaluation.status, static_cast<std::int64_t>(t));
            return out;
        }
        out.log_likelihood += evaluation.log_likelihood;
        if (!std::isfinite(out.log_likelihood)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        if (need_score) {
            g = next_g(params, config, g, evaluation.score);
            if (!std::isfinite(g)) {
                set_failure(
                    out, SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(t));
                return out;
            }
        }
    }
    return out;
}

}  // namespace

GasStateResult GasEvaluator::initial_state(
    const GasParams& params,
    const CopulaSpec& copula,
    const GasConfig& config) const {

    GasStateResult out;
    if (!valid_params(params) || !valid_config(config)) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }
    out.status = validate_copula(copula);
    if (out.status != SCAR_OK) {
        return out;
    }
    out.g = initial_g(params, config);
    out.parameter = gas_transform(copula, out.g);
    if (!std::isfinite(out.g) || !std::isfinite(out.parameter)) {
        out.status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

GasFilterResult GasEvaluator::filter(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    GasFilterResult out;
    const int status = validate_inputs(params, copula, u, config);
    if (status != SCAR_OK) {
        set_failure(out, status, -1);
        return out;
    }

    out.g_path.resize(u.n_obs);
    out.r_path.resize(u.n_obs);
    out.score_path.resize(u.n_obs - 1);

    double g = initial_g(params, config);
    if (!std::isfinite(g)) {
        set_failure(out, SCAR_NUMERICAL_FAILURE, -1);
        return out;
    }
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        out.g_path[t] = g;
        const double* row =
            u.values + static_cast<std::size_t>(u.dim) * t;
        const bool need_score = t + 1 < u.n_obs;
        const RowEvaluation evaluation = evaluate_row(
            copula,
            row,
            static_cast<std::int64_t>(t),
            g,
            config,
            need_score);
        out.r_path[t] = evaluation.r;
        if (evaluation.status != SCAR_OK) {
            set_failure(
                out, evaluation.status, static_cast<std::int64_t>(t));
            return out;
        }
        out.log_likelihood += evaluation.log_likelihood;
        if (!std::isfinite(out.log_likelihood)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        if (need_score) {
            out.score_path[t] = evaluation.score;
            g = next_g(params, config, g, evaluation.score);
            if (!std::isfinite(g)) {
                set_failure(
                    out, SCAR_NUMERICAL_FAILURE,
                    static_cast<std::int64_t>(t));
                return out;
            }
        }
    }
    return out;
}

GasLogLikResult GasEvaluator::log_likelihood(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    return run_log_likelihood(params, copula, u, config);
}

GasLogLikResult GasEvaluator::negative_log_likelihood(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    GasLogLikResult out = run_log_likelihood(
        params, copula, u, config);
    if (out.status == SCAR_OK) {
        out.log_likelihood = -out.log_likelihood;
    }
    return out;
}

GasUpdateResult GasEvaluator::update_one(
    const GasParams& params,
    const CopulaSpec& copula,
    double g,
    double u1,
    double u2,
    const GasConfig& config) const {

    const double values[2] = {u1, u2};
    return update_observation(
        params, copula, g, {values, 1, 2}, config);
}

GasUpdateResult GasEvaluator::update_observation(
    const GasParams& params,
    const CopulaSpec& copula,
    double g,
    ObservationView observation,
    const GasConfig& config) const {

    GasUpdateResult out;
    if (!valid_params(params) || !valid_config(config)
        || !std::isfinite(g)) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }
    out.status = validate_copula(copula);
    if (out.status != SCAR_OK) {
        return out;
    }
    if (observation.values == nullptr
        || observation.n_obs != 1
        || observation.dim != expected_dimension(copula)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    for (int j = 0; j < observation.dim; ++j) {
        if (!std::isfinite(observation.values[j])) {
            out.status = SCAR_INVALID_PARAMETER;
            return out;
        }
    }

    const RowEvaluation evaluation = evaluate_row(
        copula, observation.values, 0, g, config, true);
    out.status = evaluation.status;
    out.r = evaluation.r;
    out.log_likelihood = evaluation.log_likelihood;
    out.score = evaluation.score;
    if (out.status != SCAR_OK) {
        return out;
    }
    out.g_next = next_g(params, config, g, evaluation.score);
    if (!std::isfinite(out.g_next)) {
        out.status = SCAR_NUMERICAL_FAILURE;
        return out;
    }
    out.r_next = gas_transform(copula, out.g_next);
    if (!std::isfinite(out.r_next)) {
        out.status = SCAR_NUMERICAL_FAILURE;
    }
    return out;
}

GasPredictResult GasEvaluator::predict_parameter(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config,
    bool horizon_next) const {

    GasPredictResult out;
    const GasFilterResult filtered = filter(
        params, copula, u, config);
    out.status = filtered.status;
    out.failure_index = filtered.failure_index;
    if (out.status != SCAR_OK) {
        return out;
    }
    out.parameter = filtered.r_path.back();
    if (!horizon_next) {
        return out;
    }

    const double* row = u.values
        + static_cast<std::size_t>(u.dim) * (u.n_obs - 1);
    const RowEvaluation evaluation = evaluate_row(
        copula,
        row,
        static_cast<std::int64_t>(u.n_obs - 1),
        filtered.g_path.back(),
        config,
        true);
    out.status = evaluation.status;
    if (out.status != SCAR_OK) {
        out.failure_index = static_cast<std::int64_t>(u.n_obs - 1);
        return out;
    }
    out.parameter = gas_transform(
        copula,
        next_g(
            params,
            config,
            filtered.g_path.back(),
            evaluation.score));
    return out;
}

GasPathResult GasEvaluator::h_path(
    const GasParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const GasConfig& config) const {

    GasPathResult out;
    if (expected_dimension(copula) != 2
        || copula.family == CopulaFamily::Student
        || copula.family == CopulaFamily::EquicorrGaussian) {
        set_failure(out, SCAR_INVALID_FAMILY, -1);
        return out;
    }
    const GasFilterResult filtered = filter(
        params, copula, u, config);
    out.status = filtered.status;
    out.failure_index = filtered.failure_index;
    if (out.status != SCAR_OK) {
        return out;
    }

    out.values.resize(u.n_obs);
    for (std::size_t t = 0; t < u.n_obs; ++t) {
        const double* row = u.values + 2 * t;
        const double value = scar_internal::copula_h_rotated(
            copula, row[1], row[0], filtered.r_path[t]);
        if (!std::isfinite(value)) {
            set_failure(
                out, SCAR_NUMERICAL_FAILURE,
                static_cast<std::int64_t>(t));
            return out;
        }
        out.values[t] = std::clamp(
            value,
            scar_internal::kHEps,
            1.0 - scar_internal::kHEps);
    }
    return out;
}

}  // namespace scar
