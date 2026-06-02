#include "scar/ou.hpp"

#include "scar_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

std::vector<double> flatten_observations(const Observations& u) {
    std::vector<double> out(u.size() * 2, 0.0);
    for (std::size_t i = 0; i < u.size(); ++i) {
        out[2 * i] = u[i][0];
        out[2 * i + 1] = u[i][1];
    }
    return out;
}

bool supported_ou_copula(const CopulaSpec& copula) {
    return scar_internal::copula_is_supported_for_ou(copula);
}

bool valid_ou_params(const OuParams& params) {
    return std::isfinite(params.kappa)
        && std::isfinite(params.mu)
        && std::isfinite(params.nu)
        && params.kappa > 0.0
        && params.nu > 0.0;
}

bool finite_config_doubles(const OuNumericalConfig& config) {
    return std::isfinite(config.grid_range)
        && std::isfinite(config.r_gh)
        && std::isfinite(config.auto_small_kdt);
}

bool recoverable_numerical_status(int status) {
    return status == SCAR_NUMERICAL_FAILURE;
}

bool auto_loglik_accepted(const LogLikResult& result) {
    return result.status == SCAR_OK
        && std::isfinite(result.log_likelihood)
        && result.log_likelihood > -1e9;
}

bool auto_grad_accepted(const GradLogLikResult& result) {
    return result.status == SCAR_OK
        && std::isfinite(result.neg_log_likelihood)
        && result.neg_log_likelihood < 1e9;
}

LogLikResult invalid_loglik(int status, OuBackend backend) {
    return {
        -std::numeric_limits<double>::infinity(),
        backend,
        status,
    };
}

GradLogLikResult invalid_grad(int status, OuBackend backend) {
    return {
        1e10,
        std::vector<double>{0.0, 0.0, 0.0},
        backend,
        status,
    };
}

OuNumericalConfig with_default_quad_order(OuNumericalConfig config) {
    if (config.spectral_quad_order <= 0) {
        config.spectral_quad_order =
            std::max(2 * config.spectral_basis_order + 16, 48);
    }
    return config;
}

int matrix_grid_fallback_reason(
    const OuParams& params,
    const Observations& u,
    const OuNumericalConfig& config) {

    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            static_cast<std::int64_t>(u.size()),
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return SCAR_FALLBACK_FAILED;
    }
    return grid.adaptive_was_capped
        ? SCAR_FALLBACK_CAPPED
        : SCAR_FALLBACK_NONE;
}

void set_auto_fallback(
    LogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason = SCAR_FALLBACK_NONE) {

    result.fallback_chain = chain;
    result.fallback_from = chain.empty()
        ? -1
        : static_cast<int>(chain.back());
    result.matrix_fallback_reason = matrix_reason;
}

void set_auto_fallback(
    GradLogLikResult& result,
    const std::vector<OuBackend>& chain,
    int matrix_reason = SCAR_FALLBACK_NONE) {

    result.fallback_chain = chain;
    result.fallback_from = chain.empty()
        ? -1
        : static_cast<int>(chain.back());
    result.matrix_fallback_reason = matrix_reason;
}

void normalize_by_max(std::vector<double>& values) {
    double scale = 0.0;
    for (double value : values) {
        scale = std::max(scale, std::abs(value));
    }
    if (std::isfinite(scale) && scale > 0.0) {
        for (double& value : values) {
            value /= scale;
        }
    }
}

void normalize_state_prob(
    const scar_internal::OuGrid& grid,
    const std::vector<double>& phi,
    std::vector<double>& prob) {

    prob.assign(static_cast<std::size_t>(grid.K), 1.0 / static_cast<double>(grid.K));
    double total = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        prob[idx] = phi[idx] * grid.trap_w[idx];
        total += prob[idx];
    }
    if (std::isfinite(total) && total > 0.0) {
        for (double& value : prob) {
            value /= total;
        }
    } else {
        std::fill(prob.begin(), prob.end(), 1.0 / static_cast<double>(grid.K));
    }
}

StateDistribution invalid_state_distribution(int status, OuBackend backend) {
    return {{}, {}, backend, status};
}

template <typename AdvanceDensity>
StateDistribution state_distribution_impl(
    const CopulaSpec& copula,
    const scar_internal::OuGrid& grid,
    const std::vector<double>& flat_u,
    std::int64_t n_obs,
    bool horizon_next,
    OuBackend backend,
    AdvanceDensity advance_density) {

    std::vector<double> phi = grid.p0;
    std::vector<double> source(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> next_phi(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(grid.K), 0.0);

    auto advance = [&]() {
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            source[idx] = phi[idx] * grid.trap_w[idx];
        }
        advance_density(source, next_phi);
        phi.swap(next_phi);
        normalize_by_max(phi);
    };

    for (std::int64_t t = 0; t < n_obs; ++t) {
        scar_internal::copula_fi_row_on_grid(
            copula, flat_u.data(), t, grid.x_grid, fi_row);
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            phi[idx] *= fi_row[idx];
        }
        if (t < n_obs - 1) {
            advance();
        }
        normalize_by_max(phi);
    }

    if (horizon_next) {
        advance();
    }

    StateDistribution out;
    out.z_grid = grid.x_grid;
    normalize_state_prob(grid, phi, out.prob);
    out.backend = backend;
    out.status = SCAR_OK;
    return out;
}

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
        const std::size_t offset = static_cast<std::size_t>(row * K);
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
        const std::size_t offset = static_cast<std::size_t>(row * op.width);
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
    op.dense.assign(static_cast<std::size_t>(K * K), 0.0);
    op.dense_grad.assign(static_cast<std::size_t>(K * K), 0.0);

    for (int row = 0; row < K; ++row) {
        for (int col = 0; col < K; ++col) {
            const double q = xi[static_cast<std::size_t>(col)]
                - rho * xi[static_cast<std::size_t>(row)];
            const double tw = coeff
                * std::exp(-0.5 * q * q / omr2)
                * base_w[static_cast<std::size_t>(col)];
            const double dlog = rho / omr2
                + q * xi[static_cast<std::size_t>(row)] / omr2
                - rho * q * q / (omr2 * omr2);
            const std::size_t idx = static_cast<std::size_t>(row * K + col);
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
    op.cols.assign(static_cast<std::size_t>(K * width), 0);
    op.vals.assign(static_cast<std::size_t>(K * width), 0.0);
    op.grad_vals.assign(static_cast<std::size_t>(K * width), 0.0);

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
                static_cast<std::size_t>(row * width + 2 * q);

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
    const Observations& u,
    const OuNumericalConfig& config,
    OuBackend backend) {

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
    if (backend == OuBackend::LocalGh && config.gh_order <= 0) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
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
        const int K_min = static_cast<int>(
            std::ceil(2.0 * config.grid_range * sigma / dz_target)) + 1;
        K_adaptive = std::max(config.K, K_min);
    }
    int K_eff = K_adaptive;
    if (config.max_K > 0) {
        K_eff = std::min(K_adaptive, config.max_K);
        K_eff = std::max(K_eff, std::min(config.K, config.max_K));
    }
    if (K_eff < 2) {
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

    const std::vector<double> flat_u = flatten_observations(u);
    const std::size_t nK = static_cast<std::size_t>(n_obs * K_eff);
    std::vector<double> fi(nK, 0.0);
    std::vector<double> dfi_dx(nK, 0.0);
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, x_grid, r_grid, dpsi_grid);
    scar_internal::copula_pdf_and_grad_grid_precomputed(
        copula,
        flat_u.data(),
        n_obs,
        r_grid,
        dpsi_grid,
        fi,
        dfi_dx);

    std::vector<double> beta(nK, 0.0);
    std::vector<double> c_vals(static_cast<std::size_t>(n_obs - 1), 0.0);
    for (int j = 0; j < K_eff; ++j) {
        beta[static_cast<std::size_t>((n_obs - 1) * K_eff + j)] = 1.0;
    }
    std::vector<double> target(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> next(static_cast<std::size_t>(K_eff), 0.0);
    double cumul_logc = 0.0;
    for (std::int64_t t = n_obs - 2; t >= 0; --t) {
        const std::size_t next_row = static_cast<std::size_t>((t + 1) * K_eff);
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
        const std::size_t row = static_cast<std::size_t>(t * K_eff);
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
    std::vector<double> dx_dalpha(static_cast<std::size_t>(3 * K_eff), 0.0);
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        dx_dalpha[idx] = dsigma_dkappa * xi[idx];
        dx_dalpha[static_cast<std::size_t>(K_eff) + idx] = 1.0;
        dx_dalpha[static_cast<std::size_t>(2 * K_eff) + idx] =
            dsigma_dnu * xi[idx];
    }

    std::vector<double> d_beta(static_cast<std::size_t>(3 * K_eff), 0.0);
    std::vector<double> new_d_beta(static_cast<std::size_t>(3 * K_eff), 0.0);
    std::vector<double> d_target(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> contrib(static_cast<std::size_t>(K_eff), 0.0);
    std::vector<double> transition_grad(static_cast<std::size_t>(K_eff), 0.0);
    for (std::int64_t t = n_obs - 2; t >= 0; --t) {
        const std::size_t next_row = static_cast<std::size_t>((t + 1) * K_eff);
        for (int j = 0; j < K_eff; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            target[idx] = fi[next_row + idx] * beta[next_row + idx];
        }
        operator_matvec(op, true, target, transition_grad);
        const double inv_c = 1.0 / c_vals[static_cast<std::size_t>(t)];

        for (int p = 0; p < 3; ++p) {
            const std::size_t p_offset = static_cast<std::size_t>(p * K_eff);
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
        const std::size_t p_offset = static_cast<std::size_t>(p * K_eff);
        double num = 0.0;
        for (int j = 0; j < K_eff; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            const double dfi0 = dfi_dx[idx] * dx_dalpha[p_offset + idx];
            num += (dfi0 * beta[idx] + fi[idx] * d_beta[p_offset + idx])
                * pw_const[idx];
        }
        grad[p] = num / Z0;
    }

    GradLogLikResult out;
    out.neg_log_likelihood = -(std::log(Z0) + cumul_logc);
    out.neg_gradient = {-grad[0], -grad[1], -grad[2]};
    out.backend = backend;
    out.status = SCAR_OK;
    return out;
}

}  // namespace

LogLikResult ScarOuEvaluator::loglik_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& raw_config) const {

    const OuNumericalConfig config = with_default_quad_order(raw_config);
    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_loglik(SCAR_INVALID_TRANSFORM, OuBackend::Spectral);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (n_obs <= 0
        || config.spectral_basis_order <= 0
        || config.spectral_quad_order < config.spectral_basis_order) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }

    const double sigma = params.nu / std::sqrt(2.0 * params.kappa);
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
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
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    const std::vector<double> flat_u = flatten_observations(u);
    const double dt = n_obs > 1 ? 1.0 / static_cast<double>(n_obs - 1) : 1.0;
    const double rho = std::exp(-params.kappa * dt);

    std::vector<double> powers(
        static_cast<std::size_t>(config.spectral_basis_order), 1.0);
    for (int n = 1; n < config.spectral_basis_order; ++n) {
        powers[static_cast<std::size_t>(n)] =
            powers[static_cast<std::size_t>(n - 1)] * rho;
    }

    std::vector<double> x_grid(static_cast<std::size_t>(config.spectral_quad_order));
    for (int q = 0; q < config.spectral_quad_order; ++q) {
        x_grid[static_cast<std::size_t>(q)] =
            params.mu + sigma * z[static_cast<std::size_t>(q)];
    }
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, x_grid, r_grid, dpsi_grid);

    std::vector<double> coeff(
        static_cast<std::size_t>(config.spectral_basis_order), 0.0);
    std::vector<double> projected(
        static_cast<std::size_t>(config.spectral_basis_order), 0.0);
    std::vector<double> fi_row(
        static_cast<std::size_t>(config.spectral_quad_order), 0.0);
    coeff[0] = 1.0;
    double log_scale = 0.0;

    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        scar_internal::copula_pdf_row_precomputed(
            copula,
            flat_u[2 * t],
            flat_u[2 * t + 1],
            r_grid,
            fi_row.data());

        scar_internal::project_multiply(
            coeff,
            fi_row,
            basis,
            weighted_basis,
            config.spectral_quad_order,
            config.spectral_basis_order,
            projected);

        double scale = 0.0;
        for (int n = 0; n < config.spectral_basis_order; ++n) {
            coeff[static_cast<std::size_t>(n)] =
                powers[static_cast<std::size_t>(n)]
                * projected[static_cast<std::size_t>(n)];
            scale = std::max(scale, std::abs(coeff[static_cast<std::size_t>(n)]));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
        }
        for (double& value : coeff) {
            value /= scale;
        }
        log_scale += std::log(scale);
    }

    scar_internal::copula_pdf_row_precomputed(
        copula,
        flat_u[0],
        flat_u[1],
        r_grid,
        fi_row.data());
    scar_internal::project_multiply(
        coeff,
        fi_row,
        basis,
        weighted_basis,
        config.spectral_quad_order,
        config.spectral_basis_order,
        projected);

    const double likelihood_scaled = projected[0];
    if (!std::isfinite(likelihood_scaled) || likelihood_scaled <= 0.0) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }
    return {std::log(likelihood_scaled) + log_scale, OuBackend::Spectral, SCAR_OK};
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& raw_config) const {

    const OuNumericalConfig config = with_default_quad_order(raw_config);
    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_grad(SCAR_INVALID_TRANSFORM, OuBackend::Spectral);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (n_obs <= 0
        || config.spectral_basis_order <= 0
        || config.spectral_quad_order < config.spectral_basis_order) {
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
    const std::vector<double> flat_u = flatten_observations(u);
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
    std::vector<double> dx_dalpha(static_cast<std::size_t>(3 * quad_order), 0.0);
    for (int q = 0; q < quad_order; ++q) {
        const std::size_t idx = static_cast<std::size_t>(q);
        x_grid[idx] = params.mu + sigma * z[idx];
        dx_dalpha[idx] = -0.5 * sigma / params.kappa * z[idx];
        dx_dalpha[static_cast<std::size_t>(quad_order) + idx] = 1.0;
        dx_dalpha[static_cast<std::size_t>(2 * quad_order) + idx] =
            sigma / params.nu * z[idx];
    }
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, x_grid, r_grid, dpsi_grid);

    std::vector<double> coeff(static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> dcoeff(static_cast<std::size_t>(3 * basis_order), 0.0);
    std::vector<double> projected(static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> dprojected(static_cast<std::size_t>(3 * basis_order), 0.0);
    std::vector<double> raw(static_cast<std::size_t>(basis_order), 0.0);
    std::vector<double> draw(static_cast<std::size_t>(3 * basis_order), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(quad_order), 0.0);
    std::vector<double> dfi_dx_row(static_cast<std::size_t>(quad_order), 0.0);

    coeff[0] = 1.0;
    double log_scale = 0.0;
    double dlog_scale[3] = {0.0, 0.0, 0.0};

    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        scar_internal::copula_pdf_and_grad_row_precomputed(
            copula,
            flat_u[2 * t],
            flat_u[2 * t + 1],
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

        double scale = 0.0;
        int scale_idx = 0;
        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            raw[idx] = powers[idx] * projected[idx];
            for (int p = 0; p < 3; ++p) {
                draw[static_cast<std::size_t>(p * basis_order + n)] =
                    powers[idx]
                    * dprojected[static_cast<std::size_t>(p * basis_order + n)];
            }
            draw[idx] += dpowers_dkappa[idx] * projected[idx];
            if (std::abs(raw[idx]) > scale) {
                scale = std::abs(raw[idx]);
                scale_idx = n;
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
                * draw[static_cast<std::size_t>(p * basis_order + scale_idx)];
        }

        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            coeff[idx] = raw[idx] / scale;
            for (int p = 0; p < 3; ++p) {
                const std::size_t didx =
                    static_cast<std::size_t>(p * basis_order + n);
                dcoeff[didx] = (draw[didx] * scale - raw[idx] * dscale[p])
                    / (scale * scale);
            }
        }
        log_scale += std::log(scale);
        for (int p = 0; p < 3; ++p) {
            dlog_scale[p] += dscale[p] / scale;
        }
    }

    scar_internal::copula_pdf_and_grad_row_precomputed(
        copula,
        flat_u[0],
        flat_u[1],
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

    const double likelihood_scaled = projected[0];
    if (!std::isfinite(likelihood_scaled) || likelihood_scaled <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    GradLogLikResult out;
    out.neg_log_likelihood = -(std::log(likelihood_scaled) + log_scale);
    out.neg_gradient.assign(3, 0.0);
    for (int p = 0; p < 3; ++p) {
        const double grad =
            dprojected[static_cast<std::size_t>(p * basis_order)]
            / likelihood_scaled
            + dlog_scale[p];
        out.neg_gradient[static_cast<std::size_t>(p)] = -grad;
    }
    out.backend = OuBackend::Spectral;
    out.status = SCAR_OK;
    return out;
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::LocalGh);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::Matrix);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config) const {

    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (u.empty() || config.auto_small_kdt <= 0.0) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const double kdt = u.size() <= 1
        ? params.kappa
        : params.kappa / static_cast<double>(u.size() - 1);
    if (kdt < config.auto_small_kdt) {
        return neg_loglik_with_grad_local_gh(params, copula, u, config);
    }
    GradLogLikResult result =
        neg_loglik_with_grad_spectral(params, copula, u, config);
    if (auto_grad_accepted(result)) {
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    result = neg_loglik_with_grad_matrix(params, copula, u, config);
    const int matrix_reason = auto_grad_accepted(result)
        ? matrix_grid_fallback_reason(params, u, config)
        : SCAR_FALLBACK_FAILED;
    if (auto_grad_accepted(result)
        && matrix_reason == SCAR_FALLBACK_NONE) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                          matrix_reason);
        return result;
    }
    result = neg_loglik_with_grad_local_gh(params, copula, u, config);
    set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                      matrix_reason);
    return result;
}

LogLikResult ScarOuEvaluator::loglik_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_loglik(SCAR_INVALID_TRANSFORM, OuBackend::LocalGh);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::LocalGh);
    }
    if (n_obs < 2 || config.K < 2 || config.grid_range <= 0.0
        || config.pts_per_sigma <= 0 || config.gh_order <= 0) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    if (config.max_K > 0 && config.max_K < 2) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }
    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
    }

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
    }

    const std::vector<double> flat_u = flatten_observations(u);
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    scar_internal::copula_prepare_grid_transform(
        copula, grid.x_grid, r_grid, dpsi_grid);
    std::vector<double> msg(static_cast<std::size_t>(grid.K), 1.0);
    std::vector<double> v(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> next_msg(static_cast<std::size_t>(grid.K), 0.0);
    std::vector<double> fi_row(static_cast<std::size_t>(grid.K), 0.0);
    double log_scale = 0.0;
    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        scar_internal::copula_pdf_row_precomputed(
            copula,
            flat_u[2 * t],
            flat_u[2 * t + 1],
            r_grid,
            fi_row.data());
        for (int j = 0; j < grid.K; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j);
            v[idx] = fi_row[idx] * msg[idx];
        }

        scar_internal::local_gh_matvec(
            grid.z,
            grid.rho,
            grid.sigma_cond,
            gh_nodes,
            gh_weights,
            v,
            next_msg);

        double scale = 0.0;
        for (double value : next_msg) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
        }
        for (int j = 0; j < grid.K; ++j) {
            msg[static_cast<std::size_t>(j)] =
                next_msg[static_cast<std::size_t>(j)] / scale;
        }
        log_scale += std::log(scale);
    }

    scar_internal::copula_pdf_row_precomputed(
        copula,
        flat_u[0],
        flat_u[1],
        r_grid,
        fi_row.data());
    double result = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        result += fi_row[idx] * grid.p0[idx] * msg[idx] * grid.trap_w[idx];
    }
    if (!std::isfinite(result) || result <= 0.0) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::LocalGh);
    }
    return {std::log(result) + log_scale, OuBackend::LocalGh, SCAR_OK};
}

LogLikResult ScarOuEvaluator::loglik_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_loglik(SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (n_obs < 2 || config.K < 2 || config.grid_range <= 0.0
        || config.pts_per_sigma <= 0) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Matrix);
    }

    std::vector<double> matrix;
    scar_internal::build_dense_transition_matrix(grid, matrix);
    double value = -std::numeric_limits<double>::infinity();
    const std::vector<double> flat_u = flatten_observations(u);
    if (!scar_internal::matrix_backward_loglik(
            copula,
            grid,
            matrix,
            flat_u.data(),
            n_obs,
            value)) {
        return invalid_loglik(SCAR_NUMERICAL_FAILURE, OuBackend::Matrix);
    }
    return {value, OuBackend::Matrix, SCAR_OK};
}

LogLikResult ScarOuEvaluator::loglik_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config) const {

    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_loglik(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    if (u.empty() || config.auto_small_kdt <= 0.0) {
        return invalid_loglik(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const double kdt = u.size() <= 1
        ? params.kappa
        : params.kappa / static_cast<double>(u.size() - 1);
    if (kdt < config.auto_small_kdt) {
        return loglik_local_gh(params, copula, u, config);
    }

    LogLikResult result = loglik_spectral(params, copula, u, config);
    if (auto_loglik_accepted(result)) {
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    result = loglik_matrix(params, copula, u, config);
    const int matrix_reason = auto_loglik_accepted(result)
        ? matrix_grid_fallback_reason(params, u, config)
        : SCAR_FALLBACK_FAILED;
    if (auto_loglik_accepted(result)
        && matrix_reason == SCAR_FALLBACK_NONE) {
        set_auto_fallback(result, {OuBackend::Spectral});
        return result;
    }
    if (result.status != SCAR_OK
        && !recoverable_numerical_status(result.status)) {
        set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                          matrix_reason);
        return result;
    }
    result = loglik_local_gh(params, copula, u, config);
    set_auto_fallback(result, {OuBackend::Spectral, OuBackend::Matrix},
                      matrix_reason);
    return result;
}

std::vector<double> ScarOuEvaluator::predictive_mean_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const std::vector<double> flat_u = flatten_observations(u);
    if (!scar_internal::local_forward_predictive_mean(
            copula,
            grid, gh_nodes, gh_weights, flat_u.data(),
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_INVALID_SIZE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::predictive_mean_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    std::vector<double> matrix;
    scar_internal::build_dense_transition_matrix(grid, matrix);
    const std::vector<double> flat_u = flatten_observations(u);
    if (!scar_internal::matrix_forward_predictive_mean(
            copula, grid, matrix, flat_u.data(), static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_INVALID_SIZE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::predictive_mean_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    scar_internal::OuGrid grid;
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(u.size(), 0.0);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return predictive_mean_matrix(params, copula, u, config, status);
    }
    return predictive_mean_local_gh(params, copula, u, config, status);
}

std::vector<double> ScarOuEvaluator::mixture_h_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    const std::vector<double> flat_u = flatten_observations(u);
    if (!scar_internal::local_forward_mixture_h(
            copula,
            grid, gh_nodes, gh_weights, flat_u.data(),
            static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_INVALID_SIZE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::mixture_h_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    int& status) const {

    status = SCAR_OK;
    std::vector<double> out(u.size(), 0.0);
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return out;
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return out;
    }
    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return out;
    }
    std::vector<double> matrix;
    scar_internal::build_dense_transition_matrix(grid, matrix);
    const std::vector<double> flat_u = flatten_observations(u);
    if (!scar_internal::matrix_forward_mixture_h(
            copula, grid, matrix, flat_u.data(), static_cast<std::int64_t>(u.size()),
            out.data())) {
        status = SCAR_INVALID_SIZE;
    }
    return out;
}

std::vector<double> ScarOuEvaluator::mixture_h_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    OuBackend& backend,
    int& status) const {

    scar_internal::OuGrid grid;
    if (!supported_ou_copula(copula)) {
        status = SCAR_INVALID_TRANSFORM;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        status = SCAR_INVALID_PARAMETER;
        return std::vector<double>(u.size(), 0.0);
    }
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        status = SCAR_INVALID_SIZE;
        return std::vector<double>(u.size(), 0.0);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    backend = selected == 0 ? OuBackend::Matrix : OuBackend::LocalGh;
    if (backend == OuBackend::Matrix) {
        return mixture_h_matrix(params, copula, u, config, status);
    }
    return mixture_h_local_gh(params, copula, u, config, status);
}

StateDistribution ScarOuEvaluator::state_distribution_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_state_distribution(SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_state_distribution(SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (n_obs < 2) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    scar_internal::OuGrid grid;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }

    std::vector<double> matrix;
    scar_internal::build_dense_transition_matrix(grid, matrix);
    const std::vector<double> flat_u = flatten_observations(u);
    auto advance = [&](const std::vector<double>& source,
                       std::vector<double>& next_phi) {
        scar_internal::dense_predict_matvec(matrix, grid, source, next_phi);
    };
    return state_distribution_impl(
        copula, grid, flat_u, n_obs, horizon_next, OuBackend::Matrix, advance);
}

StateDistribution ScarOuEvaluator::state_distribution_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    if (!supported_ou_copula(copula)) {
        return invalid_state_distribution(SCAR_INVALID_TRANSFORM, OuBackend::LocalGh);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_state_distribution(SCAR_INVALID_PARAMETER, OuBackend::LocalGh);
    }
    if (n_obs < 2) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }

    scar_internal::OuGrid grid;
    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::build_ou_grid(
            params.kappa,
            params.mu,
            params.nu,
            n_obs,
            config.K,
            config.grid_range,
            config.adaptive,
            config.pts_per_sigma,
            config.max_K,
            grid)
        || !scar_internal::physicists_hermite_normal_rule(
            config.gh_order, gh_nodes, gh_weights)) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::LocalGh);
    }

    const std::vector<double> flat_u = flatten_observations(u);
    auto advance = [&](const std::vector<double>& source,
                       std::vector<double>& next_phi) {
        scar_internal::local_gh_predict_matvec(
            grid.z,
            grid.trap_w,
            grid.rho,
            grid.sigma_cond,
            gh_nodes,
            gh_weights,
            source,
            next_phi);
    };
    return state_distribution_impl(
        copula, grid, flat_u, n_obs, horizon_next, OuBackend::LocalGh, advance);
}

StateDistribution ScarOuEvaluator::state_distribution_auto(
    const OuParams& params,
    const CopulaSpec& copula,
    const Observations& u,
    const OuNumericalConfig& config,
    bool horizon_next) const {

    scar_internal::OuGrid grid;
    if (!supported_ou_copula(copula)) {
        return invalid_state_distribution(SCAR_INVALID_TRANSFORM, OuBackend::Matrix);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_state_distribution(SCAR_INVALID_PARAMETER, OuBackend::Matrix);
    }
    if (!scar_internal::build_ou_grid(
            params.kappa, params.mu, params.nu, static_cast<std::int64_t>(u.size()),
            config.K, config.grid_range, config.adaptive,
            config.pts_per_sigma, config.max_K, grid)) {
        return invalid_state_distribution(SCAR_INVALID_SIZE, OuBackend::Matrix);
    }
    const int selected = scar_internal::select_grid_transition_backend(
        grid, config.r_gh);
    if (selected == 0) {
        return state_distribution_matrix(params, copula, u, config, horizon_next);
    }
    return state_distribution_local_gh(params, copula, u, config, horizon_next);
}

}  // namespace scar

