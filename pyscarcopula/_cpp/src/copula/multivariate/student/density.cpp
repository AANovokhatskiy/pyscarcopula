#include "scar/detail/copula/common.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/ppf_cache.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/copula/multivariate/correlation/dense.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/copula/multivariate/correlation/factor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace scar_internal {

PreparedStudentDensity prepare_student_density(
    const scar::CopulaSpec& spec) {

    PreparedStudentDensity model;
    model.dimension = spec.dim;
    model.ppf_cache =
        &scar::copula::multivariate::student::ppf_cache(spec);
    std::size_t matrix_elements = 0;
    if (spec.dim < 2
        || !valid_student_dimension(spec.dim, matrix_elements)) {
        return model;
    }
    if (spec.correlation_kind == scar::CorrelationKind::Factor) {
        model.factor = spec.factor_operator().get();
        model.valid = model.factor != nullptr
            && model.factor->dimension()
                == static_cast<std::size_t>(spec.dim)
            && std::isfinite(model.factor->logdet());
        return model;
    }
    model.dense =
        &scar::copula::multivariate::correlation::dense(spec);
    model.valid =
        model.dense->inverse_cholesky.size() == matrix_elements
        && std::isfinite(model.dense->log_determinant);
    return model;
}

namespace {

bool factor_precision_product(
    const scar::FactorCorrelationOperator& correlation,
    const std::vector<double>& values,
    StudentWorkspace& workspace) {

    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    if (values.size() != dimension) {
        return false;
    }
    workspace.resize_precision_x(dimension);
    workspace.resize_factor_small(rank);
    std::fill(
        workspace.factor_small.begin(),
        workspace.factor_small.end(),
        0.0);
    const std::vector<double>& weighted =
        correlation.weighted_loadings();
    for (std::size_t column = 0; column < dimension; ++column) {
        const double* loading = weighted.data() + column * rank;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            workspace.factor_small[factor] +=
                loading[factor] * values[column];
        }
    }
    correlation.solve_core_inplace(workspace.factor_small.data());
    const std::vector<double>& inverse_uniqueness =
        correlation.inverse_uniqueness();
    for (std::size_t column = 0; column < dimension; ++column) {
        const double* loading = weighted.data() + column * rank;
        double result = inverse_uniqueness[column] * values[column];
        for (std::size_t factor = 0; factor < rank; ++factor) {
            result -= loading[factor] * workspace.factor_small[factor];
        }
        if (!std::isfinite(result)) {
            return false;
        }
        workspace.precision_x[column] = result;
    }
    return true;
}

double student_log_pdf_with_work(
    const PreparedStudentDensity& model,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace,
    double* dlog_ddf) {

    const int d = model.dimension;
    const bool factor_correlation = model.factor != nullptr;
    const auto* factor = model.factor;
    const auto* dense = model.dense;
    const auto& cache = *model.ppf_cache;
    if (!model.valid || !std::isfinite(df)
        || df <= 2.0) {
        return -std::numeric_limits<double>::infinity();
    }

    workspace.resize_x(static_cast<std::size_t>(d));
    const bool use_cache =
        student_ppf_cache_available(cache, d, row_index)
        && df >= cache.nodes.front()
        && df <= cache.nodes.back();
    const bool compute_derivative = dlog_ddf != nullptr;
    if (compute_derivative) {
        workspace.resize_dx_ddf(static_cast<std::size_t>(d));
    } else {
        workspace.dx_ddf.clear();
    }
    PpfInterpolation interpolation;
    if (use_cache) {
        interpolation = make_ppf_interpolation(cache.nodes, df);
        workspace.diagnostics.ppf_cache_values +=
            static_cast<std::uint64_t>(d);
    } else if (use_large_df_quantile(cache, df)) {
        workspace.diagnostics.ppf_asymptotic_values +=
            static_cast<std::uint64_t>(d);
    } else {
        workspace.diagnostics.ppf_exact_values +=
            static_cast<std::uint64_t>(d);
    }
    if (use_cache) {
        interpolate_ppf_row(
            cache,
            d,
            interpolation,
            row_index,
            workspace.x.data(),
            compute_derivative ? workspace.dx_ddf.data() : nullptr);
    } else {
        for (int i = 0; i < d; ++i) {
            student_quantile_for_emission(
                cache,
                row[i],
                df,
                workspace.x[static_cast<std::size_t>(i)],
                compute_derivative
                    ? &workspace.dx_ddf[static_cast<std::size_t>(i)]
                    : nullptr);
        }
    }

    double quad = 0.0;
    double dquad_ddf = 0.0;
    if (factor_correlation) {
        if (!factor_precision_product(
                *factor, workspace.x, workspace)) {
            return -std::numeric_limits<double>::infinity();
        }
        for (int i = 0; i < d; ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            quad += workspace.x[index] * workspace.precision_x[index];
            if (compute_derivative) {
                dquad_ddf +=
                    2.0
                    * workspace.precision_x[index]
                    * workspace.dx_ddf[index];
            }
        }
    } else {
        for (int i = 0; i < d; ++i) {
            double yi = 0.0;
            double dyi_ddf = 0.0;
            const std::size_t row_offset =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
            for (int j = 0; j <= i; ++j) {
                yi += dense->inverse_cholesky[
                    row_offset + static_cast<std::size_t>(j)]
                    * workspace.x[static_cast<std::size_t>(j)];
                if (compute_derivative) {
                    dyi_ddf += dense->inverse_cholesky[
                        row_offset + static_cast<std::size_t>(j)]
                        * workspace.dx_ddf[static_cast<std::size_t>(j)];
                }
            }
            quad += yi * yi;
            if (compute_derivative) {
                dquad_ddf += 2.0 * yi * dyi_ddf;
            }
        }
    }

    double log_pdf = -std::numeric_limits<double>::infinity();
    if (!student_log_pdf_from_quantiles(
            workspace.x.data(),
            compute_derivative ? workspace.dx_ddf.data() : nullptr,
            static_cast<std::size_t>(d),
            df,
            factor_correlation
                ? factor->logdet()
                : dense->log_determinant,
            quad,
            dquad_ddf,
            log_pdf,
            dlog_ddf)) {
        return -std::numeric_limits<double>::infinity();
    }
    return log_pdf;
}

bool student_corr_score_row_impl(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    const std::vector<double>* direction,
    double* scores) {

    const int d = spec.dim;
    std::size_t matrix_elements = 0;
    if (!valid_student_dimension(d, matrix_elements)) {
        return false;
    }
    const auto& dense =
        scar::copula::multivariate::correlation::dense(spec);
    const auto& cache =
        scar::copula::multivariate::student::ppf_cache(spec);
    const std::size_t dim_size = static_cast<std::size_t>(d);
    std::size_t n_corr = 0;
    if (!valid_student_correlation_count(d, n_corr)) {
        return false;
    }
    std::size_t score_elements = 0;
    if (row == nullptr
        || scores == nullptr
        || d < 2
        || dense.inverse_cholesky.size() != matrix_elements
        || precision.size() != matrix_elements
        || (direction == nullptr
            && !checked_size_mul(
                df_grid.size(), n_corr, score_elements))
        || (direction != nullptr && direction->size() != n_corr)) {
        return false;
    }
    if (direction != nullptr) {
        for (double value : *direction) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }

    std::vector<double> x(static_cast<std::size_t>(d), 0.0);
    std::vector<double> whitened(static_cast<std::size_t>(d), 0.0);
    std::vector<double> precision_x(static_cast<std::size_t>(d), 0.0);
    for (std::size_t grid_index = 0;
         grid_index < df_grid.size();
         ++grid_index) {
        const double df = df_grid[grid_index];
        if (!std::isfinite(df) || df <= 2.0) {
            return false;
        }
        const bool use_cache =
            student_ppf_cache_available(cache, d, row_index)
            && df >= cache.nodes.front()
            && df <= cache.nodes.back();
        PpfInterpolation interpolation;
        if (use_cache) {
            interpolation = make_ppf_interpolation(cache.nodes, df);
        }
        if (use_cache) {
            interpolate_ppf_row(
                cache,
                d,
                interpolation,
                row_index,
                x.data(),
                nullptr);
        } else {
            for (int i = 0; i < d; ++i) {
                student_quantile_for_emission(
                    cache, row[i], df,
                    x[static_cast<std::size_t>(i)], nullptr);
            }
        }

        double quad = 0.0;
        for (int i = 0; i < d; ++i) {
            double value = 0.0;
            for (int j = 0; j <= i; ++j) {
                value += dense.inverse_cholesky[
                    static_cast<std::size_t>(i) * dim_size
                    + static_cast<std::size_t>(j)]
                    * x[static_cast<std::size_t>(j)];
            }
            whitened[static_cast<std::size_t>(i)] = value;
            quad += value * value;
        }
        for (int i = 0; i < d; ++i) {
            double value = 0.0;
            for (int j = i; j < d; ++j) {
                value += dense.inverse_cholesky[
                    static_cast<std::size_t>(j) * dim_size
                    + static_cast<std::size_t>(i)]
                    * whitened[static_cast<std::size_t>(j)];
            }
            precision_x[static_cast<std::size_t>(i)] = value;
        }

        const double shape_weight =
            (df + static_cast<double>(d)) / (df + quad);
        double directional_score = 0.0;
        std::size_t corr_index = 0;
        for (int i = 1; i < d; ++i) {
            for (int j = 0; j < i; ++j) {
                const double entry_score =
                    -precision[
                        static_cast<std::size_t>(i) * dim_size
                        + static_cast<std::size_t>(j)]
                    + shape_weight
                        * precision_x[static_cast<std::size_t>(i)]
                        * precision_x[static_cast<std::size_t>(j)];
                if (direction == nullptr) {
                    scores[grid_index * n_corr + corr_index] = entry_score;
                } else {
                    directional_score += (*direction)[corr_index]
                        * entry_score;
                }
                ++corr_index;
            }
        }
        if (direction != nullptr) {
            scores[grid_index] = directional_score;
        }
    }
    return true;
}

}  // namespace

bool student_log_pdf_from_quantiles(
    const double* quantiles,
    const double* quantile_derivatives,
    std::size_t dimension,
    double df,
    double logdet,
    double quadratic_form,
    double quadratic_form_derivative,
    double& log_pdf,
    double* dlog_ddf) {

    const bool compute_derivative = dlog_ddf != nullptr;
    if (quantiles == nullptr
        || dimension < 2
        || !std::isfinite(df)
        || df <= 2.0
        || !std::isfinite(logdet)
        || !std::isfinite(quadratic_form)
        || quadratic_form < 0.0
        || (compute_derivative
            && (quantile_derivatives == nullptr
                || !std::isfinite(quadratic_form_derivative)))) {
        return false;
    }

    double marginal_log = 0.0;
    double marginal_dlog_ddf = 0.0;
    double marginal_constant = 0.0;
    double marginal_constant_derivative = 0.0;
    if (!student_marginal_log_pdf_constants(
            df,
            marginal_constant,
            marginal_constant_derivative)) {
        return false;
    }
    for (std::size_t index = 0; index < dimension; ++index) {
        const double quantile = quantiles[index];
        if (!std::isfinite(quantile)
            || (compute_derivative
                && !std::isfinite(quantile_derivatives[index]))) {
            return false;
        }
        double marginal_value = 0.0;
        double marginal_derivative = 0.0;
        if (!student_marginal_log_pdf_from_quantile(
                quantile,
                compute_derivative ? quantile_derivatives[index] : 0.0,
                df,
                marginal_constant,
                marginal_constant_derivative,
                marginal_value,
                marginal_derivative)) {
            return false;
        }
        marginal_log += marginal_value;
        marginal_dlog_ddf += marginal_derivative;
    }

    return student_log_pdf_from_summaries(
        dimension,
        df,
        logdet,
        quadratic_form,
        quadratic_form_derivative,
        marginal_log,
        marginal_dlog_ddf,
        log_pdf,
        dlog_ddf);
}

bool student_marginal_log_pdf_from_quantile(
    double quantile,
    double quantile_derivative,
    double df,
    double marginal_constant,
    double marginal_constant_derivative,
    double& log_pdf,
    double& dlog_ddf) {

    if (!std::isfinite(quantile)
        || !std::isfinite(quantile_derivative)
        || !std::isfinite(df)
        || df <= 2.0
        || !std::isfinite(marginal_constant)
        || !std::isfinite(marginal_constant_derivative)) {
        return false;
    }
    const double quantile_squared = quantile * quantile;
    const double marginal_shape =
        std::log1p(quantile_squared / df);
    log_pdf = marginal_constant
        - 0.5 * (df + 1.0) * marginal_shape;
    const double quantile_squared_derivative =
        2.0 * quantile * quantile_derivative;
    const double marginal_shape_derivative =
        (df * quantile_squared_derivative - quantile_squared)
        / (df * (df + quantile_squared));
    dlog_ddf =
        marginal_constant_derivative
        - 0.5 * marginal_shape
        - 0.5 * (df + 1.0) * marginal_shape_derivative;
    return std::isfinite(log_pdf) && std::isfinite(dlog_ddf);
}

bool student_marginal_log_pdf_constants(
    double df,
    double& marginal_constant,
    double& marginal_constant_derivative) {

    if (!std::isfinite(df) || df <= 2.0) {
        return false;
    }
    marginal_constant =
        student_log_gamma(0.5 * (df + 1.0))
        - student_log_gamma(0.5 * df)
        - 0.5 * std::log(df * kPi);
    marginal_constant_derivative =
        0.5 * student_digamma_positive(0.5 * (df + 1.0))
        - 0.5 * student_digamma_positive(0.5 * df)
        - 0.5 / df;
    return std::isfinite(marginal_constant)
        && std::isfinite(marginal_constant_derivative);
}

bool student_log_pdf_from_summaries(
    std::size_t dimension,
    double df,
    double logdet,
    double quadratic_form,
    double quadratic_form_derivative,
    double marginal_log_pdf,
    double marginal_dlog_ddf,
    double& log_pdf,
    double* dlog_ddf) {

    const bool compute_derivative = dlog_ddf != nullptr;
    if (dimension < 2
        || !std::isfinite(df)
        || df <= 2.0
        || !std::isfinite(logdet)
        || !std::isfinite(quadratic_form)
        || quadratic_form < 0.0
        || !std::isfinite(marginal_log_pdf)
        || (compute_derivative
            && (!std::isfinite(quadratic_form_derivative)
                || !std::isfinite(marginal_dlog_ddf)))) {
        return false;
    }
    const double dimension_value = static_cast<double>(dimension);
    const double joint_shape = std::log1p(quadratic_form / df);
    const double joint_log =
        student_log_gamma(0.5 * (df + dimension_value))
        - student_log_gamma(0.5 * df)
        - 0.5 * dimension_value * std::log(df * kPi)
        - 0.5 * logdet
        - 0.5 * (df + dimension_value) * joint_shape;
    log_pdf = joint_log - marginal_log_pdf;
    if (compute_derivative) {
        const double joint_const_derivative =
            0.5 * student_digamma_positive(0.5 * (df + dimension_value))
            - 0.5 * student_digamma_positive(0.5 * df)
            - 0.5 * dimension_value / df;
        const double joint_shape_derivative =
            (df * quadratic_form_derivative - quadratic_form)
            / (df * (df + quadratic_form));
        const double joint_dlog_ddf =
            joint_const_derivative
            - 0.5 * joint_shape
            - 0.5 * (df + dimension_value) * joint_shape_derivative;
        *dlog_ddf = joint_dlog_ddf - marginal_dlog_ddf;
    }
    return std::isfinite(log_pdf)
        && (!compute_derivative || std::isfinite(*dlog_ddf));
}

double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index) {

    StudentWorkspace workspace;
    return student_log_pdf(
        spec, row, df, row_index, workspace);
}

double student_log_pdf(
    const PreparedStudentDensity& model,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace) {

    return student_log_pdf_with_work(
        model, row, df, row_index, workspace, nullptr);
}

double student_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    StudentWorkspace& workspace) {

    const PreparedStudentDensity model = prepare_student_density(spec);
    return student_log_pdf(
        model, row, df, row_index, workspace);
}

bool student_log_pdf_and_dlog_ddf(
    const PreparedStudentDensity& model,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf,
    StudentWorkspace& workspace) {

    log_pdf = student_log_pdf_with_work(
        model, row, df, row_index, workspace, &dlog_ddf);
    if (!std::isfinite(log_pdf)) {
        return false;
    }
    return std::isfinite(dlog_ddf);
}

bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf) {

    StudentWorkspace workspace;
    return student_log_pdf_and_dlog_ddf(
        spec, row, df, row_index, log_pdf, dlog_ddf, workspace);
}

bool student_log_pdf_and_dlog_ddf(
    const scar::CopulaSpec& spec,
    const double* row,
    double df,
    std::int64_t row_index,
    double& log_pdf,
    double& dlog_ddf,
    StudentWorkspace& workspace) {

    const PreparedStudentDensity model = prepare_student_density(spec);
    return student_log_pdf_and_dlog_ddf(
        model,
        row,
        df,
        row_index,
        log_pdf,
        dlog_ddf,
        workspace);
}

bool student_precision_matrix(
    const scar::CopulaSpec& spec,
    std::vector<double>& precision) {

    const int d = spec.dim;
    std::size_t matrix_elements = 0;
    const auto& dense =
        scar::copula::multivariate::correlation::dense(spec);
    if (d < 2
        || !valid_student_dimension(d, matrix_elements)
        || dense.inverse_cholesky.size() != matrix_elements) {
        return false;
    }
    precision.assign(matrix_elements, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = 0.0;
            for (int k = i; k < d; ++k) {
                value +=
                    dense.inverse_cholesky[
                        static_cast<std::size_t>(k)
                            * static_cast<std::size_t>(d)
                        + static_cast<std::size_t>(i)]
                    * dense.inverse_cholesky[
                        static_cast<std::size_t>(k)
                            * static_cast<std::size_t>(d)
                        + static_cast<std::size_t>(j)];
            }
            precision[
                static_cast<std::size_t>(i) * static_cast<std::size_t>(d)
                + static_cast<std::size_t>(j)] = value;
            precision[
                static_cast<std::size_t>(j) * static_cast<std::size_t>(d)
                + static_cast<std::size_t>(i)] = value;
        }
    }
    return true;
}

bool student_corr_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    double* scores) {

    return student_corr_score_row_impl(
        spec,
        row,
        row_index,
        df_grid,
        precision,
        nullptr,
        scores);
}

bool student_corr_directional_score_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& precision,
    const std::vector<double>& direction,
    double* scores) {

    return student_corr_score_row_impl(
        spec,
        row,
        row_index,
        df_grid,
        precision,
        &direction,
        scores);
}

void student_fill_row(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace::Diagnostics* diagnostics) {

    StudentWorkspace workspace;
    workspace.reserve_x(static_cast<std::size_t>(spec.dim));
    workspace.reserve_dx_ddf(static_cast<std::size_t>(spec.dim));
    student_fill_row_with_workspace(
        spec,
        row,
        row_index,
        df_grid,
        dpsi_grid,
        fi_row,
        dfi_dx_row,
        workspace);
    if (diagnostics != nullptr) {
        *diagnostics = workspace.diagnostics;
    }
}

void student_fill_row_with_workspace(
    const PreparedStudentDensity& model,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace& workspace) {

    for (std::size_t j = 0; j < df_grid.size(); ++j) {
        const double df = df_grid[j];
        double dlog = std::numeric_limits<double>::quiet_NaN();
        const double log_pdf = student_log_pdf_with_work(
            model,
            row,
            df,
            row_index,
            workspace,
            dfi_dx_row == nullptr ? nullptr : &dlog);
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;
        if (dfi_dx_row != nullptr) {
            dfi_dx_row[j] = pdf * dlog * dpsi_grid[j];
        }
    }
}

void student_fill_row_with_workspace(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi_row,
    double* dfi_dx_row,
    StudentWorkspace& workspace) {

    const PreparedStudentDensity model = prepare_student_density(spec);
    student_fill_row_with_workspace(
        model,
        row,
        row_index,
        df_grid,
        dpsi_grid,
        fi_row,
        dfi_dx_row,
        workspace);
}

void student_fill_row_from_x_grid(
    const scar::CopulaSpec& spec,
    const double* row,
    std::int64_t row_index,
    const std::vector<double>& x_grid,
    double* fi_row) {

    StudentWorkspace workspace;
    workspace.reserve_x(static_cast<std::size_t>(spec.dim));
    const PreparedStudentDensity model = prepare_student_density(spec);
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        const double df = copula_transform(spec, x_grid[j]);
        fi_row[j] = std::exp(student_log_pdf_with_work(
            model, row, df, row_index, workspace, nullptr));
    }
}

bool student_fill_grid_bivariate(
    const scar::CopulaSpec& spec,
    std::int64_t n_obs,
    const std::vector<double>& df_grid,
    const std::vector<double>& dpsi_grid,
    double* fi,
    double* dfi_dx,
    int n_threads) {

    if (spec.dim != 2) {
        return false;
    }
    const auto& dense =
        scar::copula::multivariate::correlation::dense(spec);
    const auto& cache =
        scar::copula::multivariate::student::ppf_cache(spec);
    if (n_obs <= 0
        || cache.observation_count != n_obs
        || !student_ppf_cache_available(cache, 2, 0)
        || df_grid.size() != dpsi_grid.size()
        || dense.inverse_cholesky.size() != 4) {
        return false;
    }
    if (!std::all_of(
            df_grid.begin(),
            df_grid.end(),
            [&cache](double df) {
                return std::isfinite(df)
                    && df > 2.0
                    && df >= cache.nodes.front()
                    && df <= cache.nodes.back();
            })) {
        // The optimized bivariate kernel assumes every quantile comes from
        // the interpolation table. Fall back to the general row evaluator,
        // which uses exact Student quantiles outside the cache range.
        return false;
    }

    const double l11 = dense.inverse_cholesky[3];
    if (!std::isfinite(l11) || std::abs(l11) < 1e-15) {
        return false;
    }
    const double rho = -dense.inverse_cholesky[2] / l11;
    const double one_minus_rho2 = 1.0 - rho * rho;
    if (!std::isfinite(rho) || one_minus_rho2 <= 0.0) {
        return false;
    }

    const std::size_t K = df_grid.size();
    const double log_determinant = dense.log_determinant;
    constexpr std::int64_t min_rows_per_block = 8;
    parallel_for_blocks(
        0,
        n_obs,
        min_rows_per_block,
        n_threads,
        [&](std::int64_t begin, std::int64_t end, std::size_t) {
            for (std::size_t j = 0; j < K; ++j) {
                const double df = df_grid[j];
                const PpfInterpolation interpolation =
                    make_ppf_interpolation(cache.nodes, df);

        const double half_df = 0.5 * df;
        const double log_df_pi = std::log(df * kPi);
        const double joint_const =
            student_log_gamma(half_df + 1.0)
            - student_log_gamma(half_df)
            - log_df_pi
            - 0.5 * log_determinant;
        const double marginal_const =
            student_log_gamma(half_df + 0.5)
            - student_log_gamma(half_df)
            - 0.5 * log_df_pi;
        const double copula_const = joint_const - 2.0 * marginal_const;

        const double digamma_half_df = student_digamma_positive(half_df);
        const double joint_const_derivative =
            0.5 * student_digamma_positive(half_df + 1.0)
            - 0.5 * digamma_half_df
            - 1.0 / df;
        const double marginal_const_derivative =
            0.5 * student_digamma_positive(half_df + 0.5)
            - 0.5 * digamma_half_df
            - 0.5 / df;
        const double copula_const_derivative =
            joint_const_derivative - 2.0 * marginal_const_derivative;

        for (std::int64_t t = begin; t < end; ++t) {
            double x1 = 0.0;
            double x2 = 0.0;
            double dx1 = 0.0;
            double dx2 = 0.0;
            interpolate_bivariate_ppf(
                cache,
                interpolation,
                static_cast<std::size_t>(t),
                x1,
                x2,
                dx1,
                dx2);

            const double x1_sq = x1 * x1;
            const double x2_sq = x2 * x2;
            const double cross = x1 * x2;
            const double quad =
                (x1_sq - 2.0 * rho * cross + x2_sq)
                / one_minus_rho2;
            const double dquad =
                2.0 * (
                    x1 * dx1
                    - rho * (dx1 * x2 + x1 * dx2)
                    + x2 * dx2
                ) / one_minus_rho2;
            const double joint_shape = std::log1p(quad / df);
            const double marginal_shape1 = std::log1p(x1_sq / df);
            const double marginal_shape2 = std::log1p(x2_sq / df);
            const double log_pdf =
                copula_const
                - 0.5 * (df + 2.0) * joint_shape
                + 0.5 * (df + 1.0)
                    * (marginal_shape1 + marginal_shape2);
            const double pdf = std::exp(log_pdf);

            const double joint_shape_derivative =
                (df * dquad - quad) / (df * (df + quad));
            const double marginal_shape1_derivative =
                (df * 2.0 * x1 * dx1 - x1_sq)
                / (df * (df + x1_sq));
            const double marginal_shape2_derivative =
                (df * 2.0 * x2 * dx2 - x2_sq)
                / (df * (df + x2_sq));
            const double dlog_ddf =
                copula_const_derivative
                - 0.5 * joint_shape
                - 0.5 * (df + 2.0) * joint_shape_derivative
                + 0.5 * (marginal_shape1 + marginal_shape2)
                + 0.5 * (df + 1.0)
                    * (
                        marginal_shape1_derivative
                        + marginal_shape2_derivative);

            const std::size_t output =
                static_cast<std::size_t>(t) * K + j;
            fi[output] = pdf;
            dfi_dx[output] = pdf * dlog_ddf * dpsi_grid[j];
            }
            }
        });
    return true;
}

}  // namespace scar_internal
