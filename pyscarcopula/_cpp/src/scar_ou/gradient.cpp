#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scar {
using namespace evaluator_detail;

namespace {

struct GridGradientOperators {
    int K = 0;
    int width = 0;
    bool local = false;
    std::vector<double> dense;
    std::vector<double> dense_grad;
    std::vector<int> cols;
    std::vector<double> vals;
    std::vector<double> grad_vals;
};

void dense_grid_matvec(
    const std::vector<double>& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < K; ++row) {
        double acc = 0.0;
        const std::size_t offset =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(K);
        for (int col = 0; col < K; ++col) {
            acc += matrix[offset + static_cast<std::size_t>(col)]
                * v[static_cast<std::size_t>(col)];
        }
        out[static_cast<std::size_t>(row)] = acc;
    }
}

void local_grid_matvec(
    const GridGradientOperators& op,
    const std::vector<double>& values,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < op.K; ++row) {
        double acc = 0.0;
        const std::size_t offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(op.width);
        for (int j = 0; j < op.width; ++j) {
            const std::size_t idx = offset + static_cast<std::size_t>(j);
            acc += values[idx] * v[static_cast<std::size_t>(op.cols[idx])];
        }
        out[static_cast<std::size_t>(row)] = acc;
    }
}

void operator_matvec(
    const GridGradientOperators& op,
    bool gradient,
    const std::vector<double>& v,
    std::vector<double>& out) {

    if (op.local) {
        local_grid_matvec(op, gradient ? op.grad_vals : op.vals, v, out);
    } else {
        dense_grid_matvec(gradient ? op.dense_grad : op.dense, op.K, v, out);
    }
}

void operator_transpose_matvec(
    const GridGradientOperators& op,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    if (op.local) {
        for (int row = 0; row < op.K; ++row) {
            const double source = v[static_cast<std::size_t>(row)];
            const std::size_t offset =
                static_cast<std::size_t>(row)
                * static_cast<std::size_t>(op.width);
            for (int j = 0; j < op.width; ++j) {
                const std::size_t idx = offset + static_cast<std::size_t>(j);
                out[static_cast<std::size_t>(op.cols[idx])] +=
                    op.vals[idx] * source;
            }
        }
        return;
    }

    for (int row = 0; row < op.K; ++row) {
        const double source = v[static_cast<std::size_t>(row)];
        const std::size_t offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(op.K);
        for (int col = 0; col < op.K; ++col) {
            out[static_cast<std::size_t>(col)] +=
                op.dense[offset + static_cast<std::size_t>(col)] * source;
        }
    }
}

bool build_dense_grid_gradient_operator(
    const std::vector<double>& xi,
    const std::vector<double>& base_w,
    double rho,
    GridGradientOperators& op) {

    const int K = static_cast<int>(xi.size());
    const double omr2 = 1.0 - rho * rho;
    if (K < 2 || omr2 <= 0.0) {
        return false;
    }
    const double coeff = 1.0 / (std::sqrt(omr2) * std::sqrt(2.0 * scar_internal::kPi));
    op = GridGradientOperators{};
    op.K = K;
    op.local = false;
    std::size_t K_size = 0;
    std::size_t matrix_size = 0;
    if (!scar_internal::checked_positive_int_size(
            K, scar_internal::kMaxDenseGridSize, K_size)
        || !scar_internal::checked_size_mul(K_size, K_size, matrix_size)) {
        return false;
    }
    op.dense.assign(matrix_size, 0.0);
    op.dense_grad.assign(matrix_size, 0.0);

    for (int row = 0; row < K; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * K_size;
        for (int col = 0; col < K; ++col) {
            const double q = xi[static_cast<std::size_t>(col)]
                - rho * xi[static_cast<std::size_t>(row)];
            const double tw = coeff
                * std::exp(-0.5 * q * q / omr2)
                * base_w[static_cast<std::size_t>(col)];
            const double dlog = rho / omr2
                + q * xi[static_cast<std::size_t>(row)] / omr2
                - rho * q * q / (omr2 * omr2);
            const std::size_t idx =
                row_offset + static_cast<std::size_t>(col);
            op.dense[idx] = tw;
            op.dense_grad[idx] = dlog * tw;
        }
    }
    return true;
}

bool build_local_grid_gradient_operator(
    const std::vector<double>& xi,
    double rho,
    int gh_order,
    GridGradientOperators& op) {

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::physicists_hermite_normal_rule(
            gh_order, gh_nodes, gh_weights)) {
        return false;
    }
    const int K = static_cast<int>(xi.size());
    const int q_count = static_cast<int>(gh_nodes.size());
    if (q_count <= 0 || q_count > INT_MAX / 2) {
        return false;
    }
    const int width = q_count * 2;
    const double s2 = 1.0 - rho * rho;
    if (K < 2 || width <= 0 || s2 <= 0.0) {
        return false;
    }
    const double s = std::sqrt(s2);
    const double xi0 = xi.front();
    const double xi_last = xi.back();
    const double dxi = xi[1] - xi[0];

    op = GridGradientOperators{};
    op.K = K;
    op.width = width;
    op.local = true;
    std::size_t K_size = 0;
    std::size_t width_size = 0;
    std::size_t operator_size = 0;
    if (!scar_internal::checked_positive_int_size(
            K, scar_internal::kMaxGridSize, K_size)
        || !scar_internal::checked_positive_int_size(
            width,
            2 * scar_internal::kMaxSpectralOrder,
            width_size)
        || !scar_internal::checked_size_mul(
            K_size, width_size, operator_size)) {
        return false;
    }
    op.cols.assign(operator_size, 0);
    op.vals.assign(operator_size, 0.0);
    op.grad_vals.assign(operator_size, 0.0);

    for (int row = 0; row < K; ++row) {
        const double center = rho * xi[static_cast<std::size_t>(row)];
        const double dcenter_drho = xi[static_cast<std::size_t>(row)];
        for (int q = 0; q < q_count; ++q) {
            const double node = gh_nodes[static_cast<std::size_t>(q)];
            const double weight = gh_weights[static_cast<std::size_t>(q)];
            const double offset = std::sqrt(2.0) * s * node;
            const double doffset_drho = -std::sqrt(2.0) * rho / s * node;
            const double y = center + offset;
            const std::size_t pos =
                static_cast<std::size_t>(row) * width_size
                + 2 * static_cast<std::size_t>(q);

            if (y <= xi0) {
                op.cols[pos] = 0;
                op.cols[pos + 1] = 0;
                op.vals[pos] = weight;
                continue;
            }
            if (y >= xi_last) {
                op.cols[pos] = K - 1;
                op.cols[pos + 1] = K - 1;
                op.vals[pos] = weight;
                continue;
            }

            int left = static_cast<int>(std::floor((y - xi0) / dxi));
            if (left >= K - 1) {
                op.cols[pos] = K - 1;
                op.cols[pos + 1] = K - 1;
                op.vals[pos] = weight;
                continue;
            }

            const double lam = (y - xi[static_cast<std::size_t>(left)]) / dxi;
            const double dlam_drho = (dcenter_drho + doffset_drho) / dxi;
            op.cols[pos] = left;
            op.cols[pos + 1] = left + 1;
            op.vals[pos] = weight * (1.0 - lam);
            op.vals[pos + 1] = weight * lam;
            op.grad_vals[pos] = -weight * dlam_drho;
            op.grad_vals[pos + 1] = weight * dlam_drho;
        }
    }
    return true;
}

GradLogLikResult grid_neg_loglik_with_grad(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend backend,
    bool correlation_gradient = false) {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_grad(SCAR_INVALID_TRANSFORM, backend);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, backend);
    }
    if (n_obs < 2 || config.K < 2 || config.grid_range <= 0.0
        || config.pts_per_sigma <= 0) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (!valid_grid_config(config, backend)
        || !valid_observation_grid_size(u.size(), config.K)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (backend == OuBackend::LocalGh && config.gh_order <= 0) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (backend == OuBackend::LocalGh
        && static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (correlation_gradient && copula.family != CopulaFamily::Student) {
        return invalid_grad(SCAR_INVALID_FAMILY, backend);
    }

    const double dt = 1.0 / static_cast<double>(n_obs - 1);
    const double rho = std::exp(-params.kappa * dt);
    const double sigma = std::sqrt(0.5 * params.nu * params.nu / params.kappa);
    const double sigma_cond = sigma * std::sqrt(1.0 - rho * rho);
    if (!std::isfinite(sigma) || !std::isfinite(sigma_cond)
        || sigma <= 0.0 || sigma_cond <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
    }

    int K_adaptive = config.K;
    if (config.adaptive) {
        const double dz_target =
            sigma_cond / static_cast<double>(config.pts_per_sigma);
        const double K_min_value =
            std::ceil(2.0 * config.grid_range * sigma / dz_target) + 1.0;
        if (!std::isfinite(K_min_value) || K_min_value < 2.0) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
        }
        if (K_min_value > static_cast<double>(scar_internal::kMaxGridSize)
            || K_min_value > static_cast<double>(INT_MAX)) {
            if (config.max_K <= 0) {
                return invalid_grad(SCAR_INVALID_SIZE, backend);
            }
            K_adaptive =
                config.max_K == INT_MAX ? INT_MAX : config.max_K + 1;
        } else {
            const int K_min = static_cast<int>(K_min_value);
            K_adaptive = std::max(config.K, K_min);
        }
    }
    int K_eff = K_adaptive;
    if (config.max_K > 0) {
        K_eff = std::min(K_adaptive, config.max_K);
        K_eff = std::max(K_eff, std::min(config.K, config.max_K));
    }
    if (K_eff < 2) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    const std::size_t K_eff_size = static_cast<std::size_t>(K_eff);
    if (K_eff_size > scar_internal::kMaxGridSize
        || (backend == OuBackend::Matrix
            && K_eff_size > scar_internal::kMaxDenseGridSize)
        || !valid_observation_grid_size(u.size(), K_eff)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }

    const double dxi =
        2.0 * config.grid_range / static_cast<double>(K_eff - 1);
    const double r_kernel_grid = std::sqrt(1.0 - rho * rho) / dxi;
    const bool adaptive_was_capped = K_eff < K_adaptive;
    if (backend == OuBackend::Matrix
        && (adaptive_was_capped || r_kernel_grid <= config.r_gh)) {
        // Explicit matrix mode is still valid; keep it explicit like Python.
    }

    std::vector<double> xi(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> base_w(static_cast<std::size_t>(K_eff), dxi);
    std::vector<double> pw_const(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> x_grid(static_cast<std::size_t>(K_eff), 0.0);
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        xi[idx] = -config.grid_range + dxi * static_cast<double>(j);
        x_grid[idx] = params.mu + sigma * xi[idx];
    }
    base_w.front() *= 0.5;
    base_w.back() *= 0.5;
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        pw_const[idx] = std::exp(-0.5 * xi[idx] * xi[idx])
            / std::sqrt(2.0 * scar_internal::kPi)
            * base_w[idx];
    }

    GridGradientOperators op;
    const bool built = backend == OuBackend::LocalGh
        ? build_local_grid_gradient_operator(xi, rho, config.gh_order, op)
        : build_dense_grid_gradient_operator(xi, base_w, rho, op);
    if (!built) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
    }

    const double* observation_values = observation_data(copula, u);
    const std::size_t K_size = static_cast<std::size_t>(K_eff);
    std::size_t nK = 0;
    if (!scar_internal::checked_size_mul(u.size(), K_size, nK)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    std::vector<double> fi(nK, 0.0);
    std::vector<double> dfi_dx(nK, 0.0);
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, x_grid, r_grid, dpsi_grid);
    scar_internal::copula_pdf_and_grad_grid_precomputed(
        copula,
        observation_values,
        n_obs,
        r_grid,
        dpsi_grid,
        fi,
        dfi_dx);

    std::vector<double> beta(nK, 0.0);
    std::vector<double> c_vals(static_cast<std::size_t>(n_obs - 1), 0.0);
    for (int j = 0; j < K_eff; ++j) {
        beta[(u.size() - 1) * K_size + static_cast<std::size_t>(j)] = 1.0;
    }
    std::vector<double> target(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> next(static_cast<std::size_t>(K_eff), 0.0);
    double cumul_logc = 0.0;
    for (std::int64_t t = n_obs - 2; t >= 0; --t) {
        const std::size_t next_row =
            static_cast<std::size_t>(t + 1) * K_size;
        for (int j = 0; j < K_eff; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            target[idx] = fi[next_row + idx] * beta[next_row + idx];
        }
        operator_matvec(op, false, target, next);
        double scale = 0.0;
        for (double value : next) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
        }
        c_vals[static_cast<std::size_t>(t)] = scale;
        cumul_logc += std::log(scale);
        const std::size_t row = static_cast<std::size_t>(t) * K_size;
        for (int j = 0; j < K_eff; ++j) {
            beta[row + static_cast<std::size_t>(j)] =
                next[static_cast<std::size_t>(j)] / scale;
        }
    }

    double Z0 = 0.0;
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        Z0 += fi[idx] * pw_const[idx] * beta[idx];
    }
    if (!std::isfinite(Z0) || Z0 <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
    }

    const double drho_dkappa = -dt * rho;
    const double dsigma_dkappa = -0.5 * sigma / params.kappa;
    const double dsigma_dnu = sigma / params.nu;
    std::size_t triple_K = 0;
    if (!scar_internal::checked_size_mul(3, K_size, triple_K)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    std::vector<double> dx_dalpha(triple_K, 0.0);
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        dx_dalpha[idx] = dsigma_dkappa * xi[idx];
        dx_dalpha[static_cast<std::size_t>(K_eff) + idx] = 1.0;
        dx_dalpha[2 * K_size + idx] =
            dsigma_dnu * xi[idx];
    }

    std::vector<double> d_beta(triple_K, 0.0);
    std::vector<double> new_d_beta(triple_K, 0.0);
    std::vector<double> d_target(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> contrib(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> transition_grad(static_cast<std::size_t>(K_eff), 0.0);
    for (std::int64_t t = n_obs - 2; t >= 0; --t) {
        const std::size_t next_row =
            static_cast<std::size_t>(t + 1) * K_size;
        for (int j = 0; j < K_eff; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            target[idx] = fi[next_row + idx] * beta[next_row + idx];
        }
        operator_matvec(op, true, target, transition_grad);
        const double inv_c = 1.0 / c_vals[static_cast<std::size_t>(t)];

        for (int p = 0; p < 3; ++p) {
            const std::size_t p_offset =
                static_cast<std::size_t>(p) * K_size;
            for (int j = 0; j < K_eff; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                const double dfi =
                    dfi_dx[next_row + idx] * dx_dalpha[p_offset + idx];
                d_target[idx] =
                    dfi * beta[next_row + idx]
                    + fi[next_row + idx] * d_beta[p_offset + idx];
            }
            operator_matvec(op, false, d_target, contrib);
            if (p == 0) {
                for (int j = 0; j < K_eff; ++j) {
                    const std::size_t idx = static_cast<std::size_t>(j);
                    contrib[idx] += transition_grad[idx] * drho_dkappa;
                }
            }
            for (int j = 0; j < K_eff; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                new_d_beta[p_offset + idx] = contrib[idx] * inv_c;
            }
        }
        d_beta.swap(new_d_beta);
    }

    double grad[3] = {0.0, 0.0, 0.0};
    for (int p = 0; p < 3; ++p) {
        const std::size_t p_offset =
            static_cast<std::size_t>(p) * K_size;
        double num = 0.0;
        for (int j = 0; j < K_eff; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            const double dfi0 = dfi_dx[idx] * dx_dalpha[p_offset + idx];
            num += (dfi0 * beta[idx] + fi[idx] * d_beta[p_offset + idx])
                * pw_const[idx];
        }
        grad[p] = num / Z0;
    }

    std::vector<double> corr_grad;
    if (correlation_gradient) {
        std::vector<double> precision;
        if (!scar_internal::student_precision_matrix(copula, precision)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        std::size_t dim_square = 0;
        if (!scar_internal::valid_student_dimension(
                copula.dim, dim_square)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        const std::size_t dim_size =
            static_cast<std::size_t>(copula.dim);
        std::size_t n_corr = 0;
        if (!scar_internal::valid_student_correlation_count(
                copula.dim, n_corr)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        std::size_t score_elements = 0;
        if (!scar_internal::checked_size_mul(
                K_size, n_corr, score_elements)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        corr_grad.assign(n_corr, 0.0);
        std::vector<double> scores(score_elements, 0.0);
        std::vector<double> alpha = pw_const;
        std::vector<double> alpha_source(
            static_cast<std::size_t>(K_eff), 0.0);
        std::vector<double> alpha_next(
            static_cast<std::size_t>(K_eff), 0.0);

        for (std::int64_t t = 0; t < n_obs; ++t) {
            const std::size_t row_offset =
                static_cast<std::size_t>(t) * K_size;
            const double* row =
                observation_values
                + static_cast<std::size_t>(t) * dim_size;
            if (!scar_internal::student_corr_score_row(
                    copula,
                    row,
                    t,
                    r_grid,
                    precision,
                    scores.data())) {
                return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
            }

            double posterior_total = 0.0;
            for (int j = 0; j < K_eff; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                posterior_total +=
                    alpha[idx] * fi[row_offset + idx]
                    * beta[row_offset + idx];
            }
            if (!std::isfinite(posterior_total) || posterior_total <= 0.0) {
                return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
            }
            for (int j = 0; j < K_eff; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j);
                const double posterior =
                    alpha[idx] * fi[row_offset + idx]
                    * beta[row_offset + idx] / posterior_total;
                const std::size_t score_offset = idx * n_corr;
                for (std::size_t p = 0; p < n_corr; ++p) {
                    corr_grad[p] += posterior * scores[score_offset + p];
                }
            }

            if (t < n_obs - 1) {
                for (int j = 0; j < K_eff; ++j) {
                    const std::size_t idx = static_cast<std::size_t>(j);
                    alpha_source[idx] = alpha[idx] * fi[row_offset + idx];
                }
                operator_transpose_matvec(op, alpha_source, alpha_next);
                double alpha_scale = 0.0;
                for (double value : alpha_next) {
                    alpha_scale = std::max(alpha_scale, std::abs(value));
                }
                if (!std::isfinite(alpha_scale) || alpha_scale <= 0.0) {
                    return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
                }
                for (double& value : alpha_next) {
                    value /= alpha_scale;
                }
                alpha.swap(alpha_next);
            }
        }
        for (double value : corr_grad) {
            if (!std::isfinite(value)) {
                return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
            }
        }
    }

    GradLogLikResult out;
    out.neg_log_likelihood = -(std::log(Z0) + cumul_logc);
    out.neg_gradient = {-grad[0], -grad[1], -grad[2]};
    out.neg_corr_gradient.resize(corr_grad.size());
    for (std::size_t i = 0; i < corr_grad.size(); ++i) {
        out.neg_corr_gradient[i] = -corr_grad[i];
    }
    out.backend = backend;
    out.status = SCAR_OK;
    return out;
}

GradLogLikResult spectral_neg_loglik_with_grad(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& raw_config,
    bool correlation_gradient) {

    const OuNumericalConfig config = with_default_quad_order(raw_config);
    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_grad(SCAR_INVALID_TRANSFORM, OuBackend::Spectral);
    }
    if (correlation_gradient && copula.family != CopulaFamily::Student) {
        return invalid_grad(SCAR_INVALID_FAMILY, OuBackend::Spectral);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    std::size_t spectral_elements = 0;
    if (n_obs <= 0
        || !scar_internal::valid_spectral_dimensions(
            config.spectral_quad_order,
            config.spectral_basis_order,
            spectral_elements)) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }

    const double sigma = params.nu / std::sqrt(2.0 * params.kappa);
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    std::vector<double> z;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> weighted_basis;
    if (!scar_internal::standard_normal_hermite_rule_with_weighted_basis(
            config.spectral_quad_order,
            config.spectral_basis_order,
            z,
            weights,
            basis,
            weighted_basis)) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    const int quad_order = config.spectral_quad_order;
    const int basis_order = config.spectral_basis_order;
    const double* observation_values = observation_data(copula, u);
    const double dt = n_obs > 1 ? 1.0 / static_cast<double>(n_obs - 1) : 1.0;
    const double rho = std::exp(-params.kappa * dt);

    std::vector<double> powers(static_cast<std::size_t>(basis_order), 1.0);
    std::vector<double> dpowers_dkappa(static_cast<std::size_t>(basis_order), 0.0);
    for (int n = 1; n < basis_order; ++n) {
        const std::size_t idx = static_cast<std::size_t>(n);
        powers[idx] = powers[idx - 1] * rho;
        dpowers_dkappa[idx] = -dt * static_cast<double>(n) * powers[idx];
    }

    std::vector<double> x_grid(static_cast<std::size_t>(quad_order), 0.0);
    std::size_t triple_quad = 0;
    std::size_t triple_basis = 0;
    if (!scar_internal::checked_size_mul(
            3, static_cast<std::size_t>(quad_order), triple_quad)
        || !scar_internal::checked_size_mul(
            3, static_cast<std::size_t>(basis_order), triple_basis)) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    std::vector<double> dx_dalpha(triple_quad, 0.0);
    for (int q = 0; q < quad_order; ++q) {
        const std::size_t idx = static_cast<std::size_t>(q);
        x_grid[idx] = params.mu + sigma * z[idx];
        dx_dalpha[idx] = -0.5 * sigma / params.kappa * z[idx];
        dx_dalpha[static_cast<std::size_t>(quad_order) + idx] = 1.0;
        dx_dalpha[
            2 * static_cast<std::size_t>(quad_order) + idx] =
            sigma / params.nu * z[idx];
    }
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, x_grid, r_grid, dpsi_grid);

    std::size_t n_corr = 0;
    std::vector<double> precision;
    if (correlation_gradient
        && (!scar_internal::valid_student_correlation_count(
                copula.dim, n_corr)
            || !scar_internal::student_precision_matrix(
                copula, precision))) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    std::size_t corr_basis_elements = 0;
    std::size_t score_elements = 0;
    if (correlation_gradient
        && (!scar_internal::checked_size_mul(
                n_corr,
                static_cast<std::size_t>(basis_order),
                corr_basis_elements)
            || !scar_internal::checked_size_mul(
                static_cast<std::size_t>(quad_order),
                n_corr,
                score_elements))) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }

    std::vector<double> coeff(static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> dcoeff(triple_basis, 0.0);
    std::vector<double> projected(static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> dprojected(triple_basis, 0.0);
    std::vector<double> raw(static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> draw(triple_basis, 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(quad_order), 0.0);
    std::vector<double> dfi_dx_row(static_cast<std::size_t>(quad_order), 0.0);
    std::vector<double> corr_coeff(corr_basis_elements, 0.0);
    std::vector<double> corr_projected(corr_basis_elements, 0.0);
    std::vector<double> corr_raw(corr_basis_elements, 0.0);
    std::vector<double> corr_value_projected(
        static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> scores(score_elements, 0.0);
    std::vector<double> corr_dlog_scale(n_corr, 0.0);

    coeff[0] = 1.0;
    double log_scale = 0.0;
    double dlog_scale[3] = {0.0, 0.0, 0.0};

    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        scar_internal::copula_pdf_and_grad_row_precomputed_flat(
            copula,
            observation_values,
            t,
            r_grid,
            dpsi_grid,
            fi_row.data(),
            dfi_dx_row.data());

        scar_internal::project_multiply_with_grad(
            coeff,
            dcoeff,
            fi_row,
            dfi_dx_row,
            dx_dalpha,
            basis,
            weighted_basis,
            quad_order,
            basis_order,
            projected,
            dprojected);
        if (correlation_gradient) {
            const double* row =
                observation_values
                + static_cast<std::size_t>(t)
                    * static_cast<std::size_t>(copula.dim);
            if (!scar_internal::student_corr_score_row(
                    copula,
                    row,
                    t,
                    r_grid,
                    precision,
                    scores.data())) {
                return invalid_grad(
                    SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
            }
            scar_internal::project_multiply_with_score_grad(
                coeff,
                corr_coeff,
                fi_row,
                scores,
                basis,
                weighted_basis,
                quad_order,
                basis_order,
                static_cast<int>(n_corr),
                corr_value_projected,
                corr_projected);
        }

        double scale = 0.0;
        int scale_idx = 0;
        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            raw[idx] = powers[idx] * projected[idx];
            for (int p = 0; p < 3; ++p) {
                draw[
                    static_cast<std::size_t>(p)
                        * static_cast<std::size_t>(basis_order)
                    + static_cast<std::size_t>(n)] =
                    powers[idx]
                    * dprojected[
                        static_cast<std::size_t>(p)
                            * static_cast<std::size_t>(basis_order)
                        + static_cast<std::size_t>(n)];
            }
            draw[idx] += dpowers_dkappa[idx] * projected[idx];
            if (std::abs(raw[idx]) > scale) {
                scale = std::abs(raw[idx]);
                scale_idx = n;
            }
        }
        if (correlation_gradient) {
            for (std::size_t p = 0; p < n_corr; ++p) {
                const std::size_t param_base =
                    p * static_cast<std::size_t>(basis_order);
                for (int n = 0; n < basis_order; ++n) {
                    const std::size_t idx = static_cast<std::size_t>(n);
                    corr_raw[param_base + idx] =
                        powers[idx] * corr_projected[param_base + idx];
                }
            }
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
        }

        const double sign = raw[static_cast<std::size_t>(scale_idx)] >= 0.0
            ? 1.0
            : -1.0;
        double dscale[3] = {0.0, 0.0, 0.0};
        for (int p = 0; p < 3; ++p) {
            dscale[p] = sign
                * draw[
                    static_cast<std::size_t>(p)
                        * static_cast<std::size_t>(basis_order)
                    + static_cast<std::size_t>(scale_idx)];
        }

        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            coeff[idx] = raw[idx] / scale;
            for (int p = 0; p < 3; ++p) {
                const std::size_t didx =
                    static_cast<std::size_t>(p)
                        * static_cast<std::size_t>(basis_order)
                    + static_cast<std::size_t>(n);
                dcoeff[didx] = (draw[didx] * scale - raw[idx] * dscale[p])
                    / (scale * scale);
            }
        }
        log_scale += std::log(scale);
        for (int p = 0; p < 3; ++p) {
            dlog_scale[p] += dscale[p] / scale;
        }
        if (correlation_gradient) {
            for (std::size_t p = 0; p < n_corr; ++p) {
                const std::size_t param_base =
                    p * static_cast<std::size_t>(basis_order);
                const double corr_scale_derivative =
                    sign * corr_raw[
                        param_base
                        + static_cast<std::size_t>(scale_idx)];
                for (int n = 0; n < basis_order; ++n) {
                    const std::size_t idx = static_cast<std::size_t>(n);
                    corr_coeff[param_base + idx] = (
                        corr_raw[param_base + idx] * scale
                        - raw[idx] * corr_scale_derivative
                    ) / (scale * scale);
                }
                corr_dlog_scale[p] += corr_scale_derivative / scale;
            }
        }
    }

    scar_internal::copula_pdf_and_grad_row_precomputed_flat(
        copula,
        observation_values,
        0,
        r_grid,
        dpsi_grid,
        fi_row.data(),
        dfi_dx_row.data());
    scar_internal::project_multiply_with_grad(
        coeff,
        dcoeff,
        fi_row,
        dfi_dx_row,
        dx_dalpha,
        basis,
        weighted_basis,
        quad_order,
        basis_order,
        projected,
        dprojected);
    if (correlation_gradient) {
        if (!scar_internal::student_corr_score_row(
                copula,
                observation_values,
                0,
                r_grid,
                precision,
                scores.data())) {
            return invalid_grad(
                SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
        }
        scar_internal::project_multiply_with_score_grad(
            coeff,
            corr_coeff,
            fi_row,
            scores,
            basis,
            weighted_basis,
            quad_order,
            basis_order,
            static_cast<int>(n_corr),
            corr_value_projected,
            corr_projected);
    }

    const double likelihood_scaled = projected[0];
    if (!std::isfinite(likelihood_scaled) || likelihood_scaled <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    GradLogLikResult out;
    out.neg_log_likelihood = -(std::log(likelihood_scaled) + log_scale);
    out.neg_gradient.assign(3, 0.0);
    for (int p = 0; p < 3; ++p) {
        const double grad =
            dprojected[
                static_cast<std::size_t>(p)
                    * static_cast<std::size_t>(basis_order)]
            / likelihood_scaled
            + dlog_scale[p];
        out.neg_gradient[static_cast<std::size_t>(p)] = -grad;
    }
    out.neg_corr_gradient.assign(n_corr, 0.0);
    for (std::size_t p = 0; p < n_corr; ++p) {
        const double grad =
            corr_projected[
                p * static_cast<std::size_t>(basis_order)]
            / likelihood_scaled
            + corr_dlog_scale[p];
        out.neg_corr_gradient[p] = -grad;
    }
    out.backend = OuBackend::Spectral;
    out.status = SCAR_OK;
    return out;
}

}  // namespace

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return spectral_neg_loglik_with_grad(
        params, copula, u, config, false);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return spectral_neg_loglik_with_grad(
        params, copula, u, config, true);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::LocalGh);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::Matrix);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::LocalGh, true);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::Matrix, true);
}

}  // namespace scar
