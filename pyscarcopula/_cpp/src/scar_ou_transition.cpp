#include "scar_internal.hpp"

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

void dense_matvec(
    const std::vector<double>& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < K; ++row) {
        double acc = 0.0;
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(K);
        for (int col = 0; col < K; ++col) {
            acc += matrix[row_offset + static_cast<std::size_t>(col)]
                * v[static_cast<std::size_t>(col)];
        }
        out[static_cast<std::size_t>(row)] = acc;
    }
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
    const std::vector<double>& matrix,
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
        copula_pdf_row_precomputed_flat(
            copula,
            u,
            t,
            r_grid,
            fi_row.data());
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            v[idx] = fi_row[idx] * msg[idx];
        }

        dense_matvec(matrix, grid.K, v, next_msg);

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

    copula_pdf_row_precomputed_flat(
        copula,
        u,
        0,
        r_grid,
        fi_row.data());
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
    const std::vector<double>& matrix,
    const double* u,
    std::int64_t n_obs,
    double* out) {

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }

    auto advance_matrix = [&](const std::vector<double>& phi,
                              const std::vector<double>& fi_row,
                              std::vector<double>& phi_next) -> bool {
        std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = fi_row[idx] * phi[idx] * grid.trap_w[idx];
        }
        dense_predict_matvec(matrix, grid, source, phi_next);

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
    const std::vector<double>& matrix,
    const double* u,
    std::int64_t n_obs,
    double* out) {

    if (copula.family == scar::CopulaFamily::Student) {
        return false;
    }

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }

    auto advance_matrix = [&](const std::vector<double>& phi,
                              const std::vector<double>& fi_row,
                              std::vector<double>& phi_next) -> bool {
        std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = fi_row[idx] * phi[idx] * grid.trap_w[idx];
        }
        dense_predict_matvec(matrix, grid, source, phi_next);

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
        const double u2 = u[2 * t + 1];
        const double u1 = u[2 * t];
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            h_mix += weights[idx] * copula_h_rotated(copula, u2, u1, r_grid[idx]);
        }
        out[t] = std::min(std::max(h_mix, kHEps), 1.0 - kHEps);
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

    auto advance_local = [&](const std::vector<double>& phi,
                             const std::vector<double>& fi_row,
                             std::vector<double>& phi_next) -> bool {
        std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
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
    double* out) {

    if (copula.family == scar::CopulaFamily::Student) {
        return false;
    }

    std::vector<double> r_grid(static_cast<std::size_t>(grid.K), 0.0);
    for (int j = 0; j < grid.K; ++j) {
        r_grid[static_cast<std::size_t>(j)] =
            copula_transform(copula, grid.x_grid[static_cast<std::size_t>(j)]);
    }

    auto advance_local = [&](const std::vector<double>& phi,
                             const std::vector<double>& fi_row,
                             std::vector<double>& phi_next) -> bool {
        std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
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
        const double u2 = u[2 * t + 1];
        const double u1 = u[2 * t];
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            h_mix += weights[idx] * copula_h_rotated(copula, u2, u1, r_grid[idx]);
        }
        out[t] = std::min(std::max(h_mix, kHEps), 1.0 - kHEps);
    };

    return forward_filter_grid(copula, grid, u, n_obs, advance_local, on_row);
}

}  // namespace scar_internal
