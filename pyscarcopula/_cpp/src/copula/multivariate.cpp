#include "scar/copula.hpp"

#include "scar/detail/copula.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

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

int validate(
    const CopulaSpec& spec,
    const Observations& u,
    std::int64_t row_offset) {

    if (u.empty() || row_offset < 0) {
        return SCAR_INVALID_SIZE;
    }
    if (spec.family == CopulaFamily::Student) {
        if (spec.rotation != Rotation::R0
            || spec.transform != Transform::Softplus
            || !valid_factor(spec)) {
            return SCAR_INVALID_FAMILY;
        }
        if ((!spec.ppf_nodes.empty() || !spec.ppf_table.empty())
            && (spec.ppf_n_obs <= 0
                || row_offset
                    > spec.ppf_n_obs
                        - static_cast<std::int64_t>(u.size()))) {
            return SCAR_INVALID_SIZE;
        }
    } else if (spec.family == CopulaFamily::EquicorrGaussian) {
        if (spec.rotation != Rotation::R0
            || spec.transform != Transform::GaussianTanh
            || spec.dim < 2) {
            return SCAR_INVALID_FAMILY;
        }
    } else {
        return SCAR_INVALID_FAMILY;
    }

    for (const auto& row : u) {
        if (row.size() != static_cast<std::size_t>(spec.dim)) {
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

double parameter_at(const std::vector<double>& r, std::size_t row) {
    return r.size() == 1 ? r[0] : r[row];
}

void initialize_grid(
    MultivariateGridResult& out,
    std::size_t n_obs,
    std::size_t n_grid) {

    out.pdf.n_obs = static_cast<std::int64_t>(n_obs);
    out.pdf.n_grid = static_cast<std::int64_t>(n_grid);
    out.d_pdf_dx.n_obs = out.pdf.n_obs;
    out.d_pdf_dx.n_grid = out.pdf.n_grid;
    std::size_t elements = 0;
    if (!scar_internal::checked_size_mul(n_obs, n_grid, elements)) {
        out.status = SCAR_INVALID_SIZE;
        return;
    }
    out.pdf.values.assign(elements, 0.0);
    out.d_pdf_dx.values.assign(elements, 0.0);
}

bool cholesky_with_jitter(
    const std::vector<double>& matrix,
    std::size_t dimension,
    std::vector<double>& lower) {

    lower.assign(dimension * dimension, 0.0);
    double jitter = 0.0;
    for (int attempt = 0; attempt < 7; ++attempt) {
        std::fill(lower.begin(), lower.end(), 0.0);
        bool valid = true;
        for (std::size_t i = 0; i < dimension && valid; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = 0.5 * (
                    matrix[i * dimension + j]
                    + matrix[j * dimension + i]);
                if (i == j) {
                    value += jitter;
                }
                for (std::size_t k = 0; k < j; ++k) {
                    value -= lower[i * dimension + k]
                        * lower[j * dimension + k];
                }
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        valid = false;
                        break;
                    }
                    lower[i * dimension + j] = std::sqrt(value);
                } else {
                    const double diagonal = lower[j * dimension + j];
                    if (!(diagonal > 0.0)) {
                        valid = false;
                        break;
                    }
                    lower[i * dimension + j] = value / diagonal;
                }
            }
        }
        if (valid) {
            return true;
        }
        jitter = jitter == 0.0 ? 1e-12 : jitter * 10.0;
    }
    return false;
}

bool solve_spd(
    const std::vector<double>& lower,
    std::size_t dimension,
    const std::vector<double>& rhs,
    std::size_t columns,
    std::vector<double>& solution) {

    solution.assign(dimension * columns, 0.0);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t i = 0; i < dimension; ++i) {
            double value = rhs[i * columns + column];
            for (std::size_t j = 0; j < i; ++j) {
                value -= lower[i * dimension + j]
                    * solution[j * columns + column];
            }
            const double diagonal = lower[i * dimension + i];
            if (!(diagonal > 0.0)) {
                return false;
            }
            solution[i * columns + column] = value / diagonal;
        }
        for (std::size_t reverse = dimension; reverse-- > 0;) {
            double value = solution[reverse * columns + column];
            for (std::size_t j = reverse + 1; j < dimension; ++j) {
                value -= lower[j * dimension + reverse]
                    * solution[j * columns + column];
            }
            const double diagonal = lower[reverse * dimension + reverse];
            solution[reverse * columns + column] = value / diagonal;
        }
    }
    return true;
}

ConditionalSampleResult conditional_latent(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>* df,
    const std::vector<double>& normal_draws,
    const std::vector<double>* chi_square_draws,
    std::int64_t n_rows) {

    ConditionalSampleResult out;
    out.n_rows = n_rows;
    if (dimension < 2 || n_rows <= 0 || given_indices.empty()
        || given_indices.size() >= static_cast<std::size_t>(dimension)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    const std::size_t d = static_cast<std::size_t>(dimension);
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = d - n_given;
    out.n_free = static_cast<std::int64_t>(n_free);
    if (correlation_rows != 1 && correlation_rows != n_rows) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    const std::size_t corr_rows =
        static_cast<std::size_t>(correlation_rows);
    if (correlations.size() != corr_rows * d * d
        || normal_draws.size() != rows * n_free
        || (given_latent.size() != n_given
            && given_latent.size() != rows * n_given)
        || (df != nullptr && df->size() != rows)
        || (chi_square_draws != nullptr
            && chi_square_draws->size() != rows)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }

    std::vector<bool> is_given(d, false);
    for (int index : given_indices) {
        if (index < 0 || index >= dimension
            || is_given[static_cast<std::size_t>(index)]) {
            out.status = SCAR_INVALID_PARAMETER;
            return out;
        }
        is_given[static_cast<std::size_t>(index)] = true;
    }
    std::vector<int> free_indices;
    free_indices.reserve(n_free);
    for (int index = 0; index < dimension; ++index) {
        if (!is_given[static_cast<std::size_t>(index)]) {
            free_indices.push_back(index);
        }
    }

    out.values.assign(rows * n_free, 0.0);
    std::vector<double> r_gg(n_given * n_given);
    std::vector<double> r_gf(n_given * n_free);
    std::vector<double> r_fg(n_free * n_given);
    std::vector<double> r_ff(n_free * n_free);
    std::vector<double> lower_gg;
    std::vector<double> solved_given;
    std::vector<double> solved_cross;
    std::vector<double> schur_base(n_free * n_free);
    std::vector<double> covariance(n_free * n_free);
    std::vector<double> lower_cov;
    std::vector<double> given_vector(n_given);
    std::size_t prepared_corr_row =
        std::numeric_limits<std::size_t>::max();

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t corr_row = correlation_rows == 1 ? 0 : row;
        for (std::size_t i = 0; i < n_given; ++i) {
            given_vector[i] = given_latent[
                given_latent.size() == n_given
                    ? i : row * n_given + i];
        }
        if (corr_row != prepared_corr_row) {
            const double* correlation =
                correlations.data() + corr_row * d * d;
            for (std::size_t i = 0; i < n_given; ++i) {
                const std::size_t gi =
                    static_cast<std::size_t>(given_indices[i]);
                for (std::size_t j = 0; j < n_given; ++j) {
                    const std::size_t gj =
                        static_cast<std::size_t>(given_indices[j]);
                    r_gg[i * n_given + j] =
                        correlation[gi * d + gj];
                }
                for (std::size_t j = 0; j < n_free; ++j) {
                    const std::size_t fj =
                        static_cast<std::size_t>(free_indices[j]);
                    r_gf[i * n_free + j] =
                        correlation[gi * d + fj];
                }
            }
            for (std::size_t i = 0; i < n_free; ++i) {
                const std::size_t fi =
                    static_cast<std::size_t>(free_indices[i]);
                for (std::size_t j = 0; j < n_given; ++j) {
                    const std::size_t gj =
                        static_cast<std::size_t>(given_indices[j]);
                    r_fg[i * n_given + j] =
                        correlation[fi * d + gj];
                }
                for (std::size_t j = 0; j < n_free; ++j) {
                    const std::size_t fj =
                        static_cast<std::size_t>(free_indices[j]);
                    r_ff[i * n_free + j] =
                        correlation[fi * d + fj];
                }
            }

            if (!cholesky_with_jitter(r_gg, n_given, lower_gg)
                || !solve_spd(
                    lower_gg, n_given, r_gf, n_free, solved_cross)) {
                out.status = SCAR_NUMERICAL_FAILURE;
                out.failure_index = static_cast<std::int64_t>(row);
                return out;
            }
            for (std::size_t i = 0; i < n_free; ++i) {
                for (std::size_t j = 0; j < n_free; ++j) {
                    double schur = r_ff[i * n_free + j];
                    for (std::size_t k = 0; k < n_given; ++k) {
                        schur -= r_fg[i * n_given + k]
                            * solved_cross[k * n_free + j];
                    }
                    schur_base[i * n_free + j] = schur;
                }
            }
            prepared_corr_row = corr_row;
        }

        if (!solve_spd(
                lower_gg, n_given, given_vector, 1, solved_given)) {
            out.status = SCAR_NUMERICAL_FAILURE;
            out.failure_index = static_cast<std::int64_t>(row);
            return out;
        }

        double covariance_scale = 1.0;
        double radial_scale = 1.0;
        if (df != nullptr) {
            const double degrees = (*df)[row];
            const double chi_square = (*chi_square_draws)[row];
            double delta = 0.0;
            for (std::size_t i = 0; i < n_given; ++i) {
                delta += given_vector[i] * solved_given[i];
            }
            const double conditional_df =
                degrees + static_cast<double>(n_given);
            if (!(degrees > 2.0) || !(conditional_df > 0.0)
                || !(chi_square > 0.0)
                || !std::isfinite(delta)) {
                out.status = SCAR_INVALID_PARAMETER;
                out.failure_index = static_cast<std::int64_t>(row);
                return out;
            }
            covariance_scale = (degrees + delta) / conditional_df;
            radial_scale = std::sqrt(conditional_df / chi_square);
        }

        for (std::size_t i = 0; i < n_free; ++i) {
            for (std::size_t j = 0; j < n_free; ++j) {
                covariance[i * n_free + j] =
                    covariance_scale * schur_base[i * n_free + j];
            }
        }
        if (!cholesky_with_jitter(covariance, n_free, lower_cov)) {
            out.status = SCAR_NUMERICAL_FAILURE;
            out.failure_index = static_cast<std::int64_t>(row);
            return out;
        }

        for (std::size_t i = 0; i < n_free; ++i) {
            double value = 0.0;
            for (std::size_t k = 0; k < n_given; ++k) {
                value += r_fg[i * n_given + k] * solved_given[k];
            }
            double innovation = 0.0;
            for (std::size_t j = 0; j <= i; ++j) {
                innovation += lower_cov[i * n_free + j]
                    * normal_draws[row * n_free + j];
            }
            value += radial_scale * innovation;
            if (!std::isfinite(value)) {
                out.status = SCAR_NUMERICAL_FAILURE;
                out.failure_index = static_cast<std::int64_t>(row);
                return out;
            }
            out.values[row * n_free + i] = value;
        }
    }
    return out;
}

}  // namespace

MultivariateRowsResult multivariate_log_pdf_and_grad(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r,
    std::int64_t row_offset) {

    MultivariateRowsResult out;
    out.status = validate(spec, u, row_offset);
    out.log_pdf.assign(
        u.size(), -std::numeric_limits<double>::infinity());
    out.dlog_dr.assign(
        u.size(), std::numeric_limits<double>::quiet_NaN());
    if (out.status != SCAR_OK) {
        return out;
    }
    if (r.size() != 1 && r.size() != u.size()) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }

    for (std::size_t i = 0; i < u.size(); ++i) {
        const double parameter = parameter_at(r, i);
        double log_pdf = 0.0;
        double dlog = 0.0;
        bool ok = std::isfinite(parameter);
        if (ok && spec.family == CopulaFamily::Student) {
            ok = scar_internal::student_log_pdf_and_dlog_ddf(
                spec,
                u[i].data(),
                parameter,
                row_offset + static_cast<std::int64_t>(i),
                log_pdf,
                dlog);
        } else if (ok) {
            log_pdf = scar_internal::equicorr_log_pdf(
                spec, u[i].data(), parameter, &dlog);
            ok = std::isfinite(log_pdf) && std::isfinite(dlog);
        }
        if (!ok) {
            out.status = SCAR_NUMERICAL_FAILURE;
            out.failure_index = static_cast<std::int64_t>(i);
            return out;
        }
        out.log_pdf[i] = log_pdf;
        out.dlog_dr[i] = dlog;
    }
    return out;
}

MultivariateGridResult multivariate_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid,
    std::int64_t row_offset) {

    MultivariateGridResult out;
    initialize_grid(out, u.size(), x_grid.size());
    if (out.status != SCAR_OK) {
        return out;
    }
    out.status = validate(spec, u, row_offset);
    if (out.status != SCAR_OK || x_grid.empty()) {
        if (out.status == SCAR_OK) {
            out.status = SCAR_INVALID_SIZE;
        }
        return out;
    }
    if (!std::all_of(x_grid.begin(), x_grid.end(), [](double value) {
            return std::isfinite(value);
        })) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }

    std::vector<double> parameter_grid(x_grid.size(), 0.0);
    std::vector<double> dpsi_grid(x_grid.size(), 0.0);
    for (std::size_t j = 0; j < x_grid.size(); ++j) {
        parameter_grid[j] =
            scar_internal::copula_transform(spec, x_grid[j]);
        dpsi_grid[j] =
            scar_internal::copula_dtransform(spec, x_grid[j]);
    }

    for (std::size_t i = 0; i < u.size(); ++i) {
        const std::size_t base = i * x_grid.size();
        if (spec.family == CopulaFamily::Student) {
            scar_internal::student_fill_row(
                spec,
                u[i].data(),
                row_offset + static_cast<std::int64_t>(i),
                parameter_grid,
                dpsi_grid,
                out.pdf.values.data() + base,
                out.d_pdf_dx.values.data() + base);
        } else {
            for (std::size_t j = 0; j < x_grid.size(); ++j) {
                double dlog = 0.0;
                const double log_pdf = scar_internal::equicorr_log_pdf(
                    spec, u[i].data(), parameter_grid[j], &dlog);
                const double pdf = std::exp(log_pdf);
                out.pdf.values[base + j] = pdf;
                out.d_pdf_dx.values[base + j] =
                    pdf * dlog * dpsi_grid[j];
            }
        }
        for (std::size_t j = 0; j < x_grid.size(); ++j) {
            if (!std::isfinite(out.pdf.values[base + j])
                || !std::isfinite(out.d_pdf_dx.values[base + j])) {
                out.status = SCAR_NUMERICAL_FAILURE;
                out.failure_index = static_cast<std::int64_t>(i);
                return out;
            }
        }
    }
    return out;
}

ConditionalSampleResult multivariate_gaussian_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& normal_draws,
    std::int64_t n_rows) {

    return conditional_latent(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_latent,
        nullptr,
        normal_draws,
        nullptr,
        n_rows);
}

ConditionalSampleResult multivariate_student_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& df,
    const std::vector<double>& normal_draws,
    const std::vector<double>& chi_square_draws,
    std::int64_t n_rows) {

    return conditional_latent(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_latent,
        &df,
        normal_draws,
        &chi_square_draws,
        n_rows);
}

}  // namespace scar
