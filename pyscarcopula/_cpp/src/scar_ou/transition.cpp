#include "scar/ou.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar_internal {

int select_grid_transition_backend(const OuGrid& grid, double r_gh) {
    if (grid.adaptive_was_capped) {
        return 1;
    }
    if (grid.r_kernel_grid <= r_gh) {
        return 1;
    }
    return 0;
}

bool build_dense_transition_matrix(const OuGrid& grid, std::vector<double>& matrix) {
    std::size_t K = 0;
    std::size_t matrix_size = 0;
    if (!checked_positive_int_size(grid.K, kMaxDenseGridSize, K)
        || !checked_size_mul(K, K, matrix_size)) {
        matrix.clear();
        return false;
    }
    matrix.assign(matrix_size, 0.0);
    const double coeff = 1.0 / (grid.sigma_cond * std::sqrt(2.0 * kPi));
    for (int row = 0; row < grid.K; ++row) {
        const double mean = grid.rho * grid.z[static_cast<std::size_t>(row)];
        const std::size_t row_offset = static_cast<std::size_t>(row) * K;
        for (int col = 0; col < grid.K; ++col) {
            const std::size_t idx =
                row_offset + static_cast<std::size_t>(col);
            const double diff = grid.z[static_cast<std::size_t>(col)] - mean;
            matrix[idx] = coeff
                * std::exp(-0.5 * (diff / grid.sigma_cond) * (diff / grid.sigma_cond))
                * grid.trap_w[static_cast<std::size_t>(col)];
        }
    }
    return true;
}

bool build_sparse_transition_matrix(
    const std::vector<double>& z,
    double rho,
    double sigma_cond,
    const std::vector<double>& trap_w,
    int K,
    int band,
    SparseTransitionMatrix& matrix,
    const std::vector<double>* i_centers) {

    matrix = {};
    std::size_t grid_size = 0;
    if (!checked_positive_int_size(K, kMaxGridSize, grid_size)
        || K < 2
        || band < 0
        || z.size() != grid_size
        || trap_w.size() != grid_size
        || (i_centers != nullptr && i_centers->size() != grid_size)
        || !std::isfinite(rho)
        || !std::isfinite(sigma_cond)
        || sigma_cond <= 0.0) {
        return false;
    }

    const double z0 = z.front();
    const double dz = z[1] - z[0];
    if (!std::isfinite(z0) || !std::isfinite(dz) || dz <= 0.0) {
        return false;
    }
    for (std::size_t i = 0; i < grid_size; ++i) {
        if (!std::isfinite(z[i])
            || !std::isfinite(trap_w[i])
            || trap_w[i] < 0.0) {
            return false;
        }
    }

    matrix.indptr.resize(grid_size + 1, 0);
    const double inv_dz = 1.0 / dz;
    const double coeff = 1.0 / (sigma_cond * std::sqrt(2.0 * kPi));

    std::size_t nnz = 0;
    for (int row = 0; row < K; ++row) {
        const double center = rho * z[static_cast<std::size_t>(row)];
        const double i_center = i_centers == nullptr
            ? (center - z0) * inv_dz
            : (*i_centers)[static_cast<std::size_t>(row)];
        if (!std::isfinite(i_center)) {
            matrix = {};
            return false;
        }

        const double lo_value = std::floor(i_center) - band;
        const double hi_value = std::ceil(i_center) + band + 1.0;
        const int lo = lo_value <= 0.0
            ? 0
            : (lo_value >= static_cast<double>(K)
                ? K
                : static_cast<int>(lo_value));
        const int hi = hi_value <= 0.0
            ? 0
            : (hi_value >= static_cast<double>(K)
                ? K
                : static_cast<int>(hi_value));

        const std::size_t width = hi > lo
            ? static_cast<std::size_t>(hi - lo)
            : 0;
        if (!checked_size_add(nnz, width, nnz)
            || nnz > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            matrix = {};
            return false;
        }
        matrix.indptr[static_cast<std::size_t>(row) + 1] =
            static_cast<int>(nnz);
    }

    matrix.data.resize(nnz);
    matrix.indices.resize(nnz);
    for (int row = 0; row < K; ++row) {
        const double center = rho * z[static_cast<std::size_t>(row)];
        const double i_center = i_centers == nullptr
            ? (center - z0) * inv_dz
            : (*i_centers)[static_cast<std::size_t>(row)];
        const double lo_value = std::floor(i_center) - band;
        const int lo = lo_value <= 0.0
            ? 0
            : (lo_value >= static_cast<double>(K)
                ? K
                : static_cast<int>(lo_value));
        const int begin =
            matrix.indptr[static_cast<std::size_t>(row)];
        const int end =
            matrix.indptr[static_cast<std::size_t>(row) + 1];

        for (int offset = begin; offset < end; ++offset) {
            const int col = lo + offset - begin;
            const std::size_t idx = static_cast<std::size_t>(col);
            const double scaled_diff = (z[idx] - center) / sigma_cond;
            matrix.indices[static_cast<std::size_t>(offset)] = col;
            matrix.data[static_cast<std::size_t>(offset)] =
                coeff * std::exp(-0.5 * scaled_diff * scaled_diff)
                * trap_w[idx];
        }
    }
    return true;
}

void sparse_matvec(
    const SparseTransitionMatrix& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < K; ++row) {
        double value = 0.0;
        const int begin = matrix.indptr[static_cast<std::size_t>(row)];
        const int end = matrix.indptr[static_cast<std::size_t>(row) + 1];
        for (int offset = begin; offset < end; ++offset) {
            const std::size_t idx = static_cast<std::size_t>(offset);
            value += matrix.data[idx]
                * v[static_cast<std::size_t>(matrix.indices[idx])];
        }
        out[static_cast<std::size_t>(row)] = value;
    }
}

void sparse_transpose_matvec(
    const SparseTransitionMatrix& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < K; ++row) {
        const double source = v[static_cast<std::size_t>(row)];
        const int begin = matrix.indptr[static_cast<std::size_t>(row)];
        const int end = matrix.indptr[static_cast<std::size_t>(row) + 1];
        for (int offset = begin; offset < end; ++offset) {
            const std::size_t idx = static_cast<std::size_t>(offset);
            out[static_cast<std::size_t>(matrix.indices[idx])] +=
                matrix.data[idx] * source;
        }
    }
}

int matrix_transition_band(const OuGrid& grid) {
    if (grid.K < 2
        || !std::isfinite(grid.r_kernel_grid)
        || grid.r_kernel_grid <= 0.0) {
        return -1;
    }
    const double band_value = std::ceil(5.0 * grid.r_kernel_grid);
    if (!std::isfinite(band_value)
        || band_value < 0.0
        || band_value > static_cast<double>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(band_value);
}

bool build_matrix_transition_operator(
    const OuGrid& grid,
    scar::OuGridMethod method,
    MatrixTransitionOperator& op) {

    op = {};
    const int band = matrix_transition_band(grid);
    if (band < 0) {
        return false;
    }
    const bool sparse =
        method == scar::OuGridMethod::Sparse
        || (method == scar::OuGridMethod::Auto
            && (static_cast<std::size_t>(grid.K) > kMaxDenseGridSize
                || band < grid.K / 4));
    if (method == scar::OuGridMethod::Dense
        && static_cast<std::size_t>(grid.K) > kMaxDenseGridSize) {
        return false;
    }

    op.K = grid.K;
    op.sparse = sparse;
    if (sparse) {
        std::vector<double> i_centers(
            static_cast<std::size_t>(grid.K), 0.0);
        const double midpoint =
            0.5 * static_cast<double>(grid.K - 1);
        for (int row = 0; row < grid.K; ++row) {
            i_centers[static_cast<std::size_t>(row)] =
                grid.rho * static_cast<double>(row)
                + (1.0 - grid.rho) * midpoint;
        }
        return build_sparse_transition_matrix(
            grid.z,
            grid.rho,
            grid.sigma_cond,
            grid.trap_w,
            grid.K,
            band,
            op.csr,
            &i_centers);
    }
    return build_dense_transition_matrix(grid, op.dense);
}

void matrix_matvec(
    const MatrixTransitionOperator& op,
    const std::vector<double>& v,
    std::vector<double>& out) {

    if (!op.sparse) {
        dense_matvec(op.dense, op.K, v, out);
        return;
    }
    sparse_matvec(op.csr, op.K, v, out);
}

void matrix_transpose_matvec(
    const MatrixTransitionOperator& op,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    if (op.sparse) {
        sparse_transpose_matvec(op.csr, op.K, v, out);
        return;
    }
    for (int row = 0; row < op.K; ++row) {
        const double source = v[static_cast<std::size_t>(row)];
        const std::size_t row_offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(op.K);
        for (int col = 0; col < op.K; ++col) {
            out[static_cast<std::size_t>(col)] +=
                op.dense[row_offset + static_cast<std::size_t>(col)]
                * source;
        }
    }
}

void matrix_predict_matvec(
    const MatrixTransitionOperator& op,
    const OuGrid& grid,
    const std::vector<double>& source,
    std::vector<double>& out_density) {

    matrix_transpose_matvec(op, source, out_density);
    for (int col = 0; col < grid.K; ++col) {
        out_density[static_cast<std::size_t>(col)] /=
            grid.trap_w[static_cast<std::size_t>(col)];
    }
}

void dense_matvec(
    const std::vector<double>& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out) {

    scar_internal::linalg::row_major_matvec(
        matrix.data(),
        static_cast<std::size_t>(K),
        static_cast<std::size_t>(K),
        v.data(),
        out.data());
}

void dense_predict_matvec(
    const std::vector<double>& matrix,
    const OuGrid& grid,
    const std::vector<double>& source,
    std::vector<double>& out_density) {

    std::fill(out_density.begin(), out_density.end(), 0.0);
    for (int row = 0; row < grid.K; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(grid.K);
        const double source_value = source[static_cast<std::size_t>(row)];
        for (int col = 0; col < grid.K; ++col) {
            out_density[static_cast<std::size_t>(col)] +=
                matrix[row_offset + static_cast<std::size_t>(col)] * source_value;
        }
    }
    for (int col = 0; col < grid.K; ++col) {
        out_density[static_cast<std::size_t>(col)] /=
            grid.trap_w[static_cast<std::size_t>(col)];
    }
}

bool matrix_backward_loglik(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double& loglik) {

    std::vector<double> msg(static_cast<std::size_t>(grid.K), 1.0);
    std::vector<double> v(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> next_msg(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    copula_prepare_grid_transform(copula, grid.x_grid, r_grid, dpsi_grid);

    double log_scale = 0.0;
    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        double emission_log_scale = 0.0;
        copula_pdf_row_precomputed_flat(
            copula,
            u,
            t,
            r_grid,
            fi_row.data(),
            &emission_log_scale);
        log_scale += emission_log_scale;
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            v[idx] = fi_row[idx] * msg[idx];
        }

        matrix_matvec(op, v, next_msg);

        double scale = 0.0;
        for (double value : next_msg) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return false;
        }
        for (int j = 0; j < grid.K; ++j) {
            msg[static_cast<std::size_t>(j)] =
                next_msg[static_cast<std::size_t>(j)] / scale;
        }
        log_scale += std::log(scale);
    }

    double emission_log_scale = 0.0;
    copula_pdf_row_precomputed_flat(
        copula,
        u,
        0,
        r_grid,
        fi_row.data(),
        &emission_log_scale);
    log_scale += emission_log_scale;
    double result = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        result += fi_row[idx] * grid.p0[idx] * msg[idx] * grid.trap_w[idx];
    }
    if (!std::isfinite(result) || result <= 0.0) {
        return false;
    }

    loglik = std::log(result) + log_scale;
    return true;
}

bool matrix_forward_predictive_mean(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double* out) {

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }
    std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);

    auto advance_matrix = [&](const std::vector<double>& phi,
                              const std::vector<double>& fi_row,
                              std::vector<double>& phi_next) -> bool {
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = fi_row[idx] * phi[idx] * grid.trap_w[idx];
        }
        matrix_predict_matvec(op, grid, source, phi_next);

        double scale = 0.0;
        for (double value : phi_next) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return false;
        }
        for (double& value : phi_next) {
            value /= scale;
        }
        return true;
    };

    auto on_row = [&](std::int64_t t,
                      const std::vector<double>& weights,
                      const std::vector<double>& /*fi_row*/) {
        double mean = 0.0;
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            mean += weights[idx] * r_grid[idx];
        }
        out[t] = mean;
    };

    return forward_filter_grid(copula, grid, u, n_obs, advance_matrix, on_row);
}

bool matrix_forward_mixture_h(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const MatrixTransitionOperator& op,
    const double* u,
    std::int64_t n_obs,
    double* out,
    double* out_reverse) {

    if (copula.family == scar::CopulaFamily::Student) {
        return false;
    }

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }
    std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
    const std::size_t n_obs_size = static_cast<std::size_t>(n_obs);
    const bool use_gaussian_quantiles =
        copula.family == scar::CopulaFamily::Gaussian
        && copula.rotation == scar::Rotation::R0
        && copula.gaussian_z1_cache.size() == n_obs_size
        && copula.gaussian_z2_cache.size() == n_obs_size;
    const scar::CopulaSpec transposed_copula =
        transposed_copula_spec(copula);

    auto advance_matrix = [&](const std::vector<double>& phi,
                              const std::vector<double>& fi_row,
                              std::vector<double>& phi_next) -> bool {
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = fi_row[idx] * phi[idx] * grid.trap_w[idx];
        }
        matrix_predict_matvec(op, grid, source, phi_next);

        double scale = 0.0;
        for (double value : phi_next) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return false;
        }
        for (double& value : phi_next) {
            value /= scale;
        }
        return true;
    };

    auto on_row = [&](std::int64_t t,
                      const std::vector<double>& weights,
                      const std::vector<double>& /*fi_row*/) {
        double h_mix = 0.0;
        double h_mix_reverse = 0.0;
        const double u2 = u[2 * t + 1];
        const double u1 = u[2 * t];
        if (use_gaussian_quantiles) {
            const std::size_t row = static_cast<std::size_t>(t);
            const double z1 = copula.gaussian_z1_cache[row];
            const double z2 = copula.gaussian_z2_cache[row];
            for (int j = 0; j < grid.K; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                h_mix += weights[idx]
                    * gaussian_h_from_quantiles(z2, z1, r_grid[idx]);
                if (out_reverse != nullptr) {
                    h_mix_reverse += weights[idx]
                        * gaussian_h_from_quantiles(z1, z2, r_grid[idx]);
                }
            }
        } else {
            for (int j = 0; j < grid.K; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                h_mix += weights[idx]
                    * copula_h_rotated(
                        transposed_copula, u2, u1, r_grid[idx]);
                if (out_reverse != nullptr) {
                    h_mix_reverse += weights[idx]
                        * copula_h_rotated(copula, u1, u2, r_grid[idx]);
                }
            }
        }
        out[t] = std::min(std::max(h_mix, kHEps), 1.0 - kHEps);
        if (out_reverse != nullptr) {
            out_reverse[t] = std::min(
                std::max(h_mix_reverse, kHEps), 1.0 - kHEps);
        }
    };

    return forward_filter_grid(copula, grid, u, n_obs, advance_matrix, on_row);
}

bool local_forward_predictive_mean(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const double* u,
    std::int64_t n_obs,
    double* out) {

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }
    std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);

    auto advance_local = [&](const std::vector<double>& phi,
                             const std::vector<double>& fi_row,
                             std::vector<double>& phi_next) -> bool {
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = fi_row[idx] * phi[idx] * grid.trap_w[idx];
        }
        local_gh_predict_matvec(
            grid.z,
            grid.trap_w,
            grid.rho,
            grid.sigma_cond,
            gh_nodes,
            gh_weights,
            source,
            phi_next);

        double scale = 0.0;
        for (double value : phi_next) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return false;
        }
        for (double& value : phi_next) {
            value /= scale;
        }
        return true;
    };

    auto on_row = [&](std::int64_t t,
                      const std::vector<double>& weights,
                      const std::vector<double>& /*fi_row*/) {
        double mean = 0.0;
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            mean += weights[idx] * r_grid[idx];
        }
        out[t] = mean;
    };

    return forward_filter_grid(copula, grid, u, n_obs, advance_local, on_row);
}

bool local_forward_mixture_h(
    const scar::CopulaSpec& copula,
    const OuGrid& grid,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const double* u,
    std::int64_t n_obs,
    double* out,
    double* out_reverse) {

    if (copula.family == scar::CopulaFamily::Student) {
        return false;
    }

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }
    std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
    const std::size_t n_obs_size = static_cast<std::size_t>(n_obs);
    const bool use_gaussian_quantiles =
        copula.family == scar::CopulaFamily::Gaussian
        && copula.rotation == scar::Rotation::R0
        && copula.gaussian_z1_cache.size() == n_obs_size
        && copula.gaussian_z2_cache.size() == n_obs_size;
    const scar::CopulaSpec transposed_copula =
        transposed_copula_spec(copula);

    auto advance_local = [&](const std::vector<double>& phi,
                             const std::vector<double>& fi_row,
                             std::vector<double>& phi_next) -> bool {
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = fi_row[idx] * phi[idx] * grid.trap_w[idx];
        }
        local_gh_predict_matvec(
            grid.z,
            grid.trap_w,
            grid.rho,
            grid.sigma_cond,
            gh_nodes,
            gh_weights,
            source,
            phi_next);

        double scale = 0.0;
        for (double value : phi_next) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return false;
        }
        for (double& value : phi_next) {
            value /= scale;
        }
        return true;
    };

    auto on_row = [&](std::int64_t t,
                      const std::vector<double>& weights,
                      const std::vector<double>& /*fi_row*/) {
        double h_mix = 0.0;
        double h_mix_reverse = 0.0;
        const double u2 = u[2 * t + 1];
        const double u1 = u[2 * t];
        if (use_gaussian_quantiles) {
            const std::size_t row = static_cast<std::size_t>(t);
            const double z1 = copula.gaussian_z1_cache[row];
            const double z2 = copula.gaussian_z2_cache[row];
            for (int j = 0; j < grid.K; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                h_mix += weights[idx]
                    * gaussian_h_from_quantiles(z2, z1, r_grid[idx]);
                if (out_reverse != nullptr) {
                    h_mix_reverse += weights[idx]
                        * gaussian_h_from_quantiles(z1, z2, r_grid[idx]);
                }
            }
        } else {
            for (int j = 0; j < grid.K; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                h_mix += weights[idx]
                    * copula_h_rotated(
                        transposed_copula, u2, u1, r_grid[idx]);
                if (out_reverse != nullptr) {
                    h_mix_reverse += weights[idx]
                        * copula_h_rotated(copula, u1, u2, r_grid[idx]);
                }
            }
        }
        out[t] = std::min(std::max(h_mix, kHEps), 1.0 - kHEps);
        if (out_reverse != nullptr) {
            out_reverse[t] = std::min(
                std::max(h_mix_reverse, kHEps), 1.0 - kHEps);
        }
    };

    return forward_filter_grid(copula, grid, u, n_obs, advance_local, on_row);
}

}  // namespace scar_internal
