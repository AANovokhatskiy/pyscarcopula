#include "scar_internal.hpp"

namespace scar_internal {

bool build_ou_grid(
    double kappa,
    double mu,
    double nu,
    std::int64_t n_obs,
    int K,
    double grid_range,
    bool adaptive,
    int pts_per_sigma,
    int max_K,
    OuGrid& grid) {

    if (n_obs < 2 || K < 2 || grid_range <= 0.0 || pts_per_sigma <= 0) {
        return false;
    }
    if (max_K > 0 && max_K < 2) {
        return false;
    }
    if (!std::isfinite(kappa) || !std::isfinite(mu) || !std::isfinite(nu)
        || !std::isfinite(grid_range) || kappa <= 0.0 || nu <= 0.0) {
        return false;
    }

    const double dt = 1.0 / static_cast<double>(n_obs - 1);
    const double rho = std::exp(-kappa * dt);
    const double sigma = std::sqrt(0.5 * nu * nu / kappa);
    const double sigma_cond = sigma * std::sqrt(1.0 - rho * rho);
    if (!std::isfinite(sigma) || !std::isfinite(sigma_cond)
        || sigma <= 0.0 || sigma_cond <= 0.0) {
        return false;
    }

    int K_adaptive = K;
    if (adaptive) {
        const double dz_target = sigma_cond / static_cast<double>(pts_per_sigma);
        const int K_min = static_cast<int>(
            std::ceil(2.0 * grid_range * sigma / dz_target)) + 1;
        K_adaptive = std::max(K, K_min);
    }

    int K_eff = K_adaptive;
    if (max_K > 0) {
        K_eff = std::min(K_adaptive, max_K);
        K_eff = std::max(K_eff, std::min(K, max_K));
    }
    if (K_eff < 2) {
        return false;
    }

    grid = OuGrid{};
    grid.K = K_eff;
    grid.rho = rho;
    grid.sigma = sigma;
    grid.sigma_cond = sigma_cond;
    grid.adaptive_was_capped = K_eff < K_adaptive;
    grid.z.assign(static_cast<std::size_t>(K_eff), 0.0);
    grid.x_grid.assign(static_cast<std::size_t>(K_eff), 0.0);
    grid.trap_w.assign(static_cast<std::size_t>(K_eff), 0.0);
    grid.p0.assign(static_cast<std::size_t>(K_eff), 0.0);

    const double z_min = -grid_range * sigma;
    const double z_max = grid_range * sigma;
    const double dz = (z_max - z_min) / static_cast<double>(K_eff - 1);
    grid.dz = dz;
    grid.r_kernel_grid = sigma_cond / dz;

    for (int j = 0; j < K_eff; ++j) {
        const double zj = z_min + dz * static_cast<double>(j);
        const std::size_t idx = static_cast<std::size_t>(j);
        grid.z[idx] = zj;
        grid.x_grid[idx] = zj + mu;
        grid.trap_w[idx] = dz;
        grid.p0[idx] =
            std::exp(-0.5 * (zj / sigma) * (zj / sigma))
            / (sigma * std::sqrt(2.0 * kPi));
    }
    grid.trap_w.front() *= 0.5;
    grid.trap_w.back() *= 0.5;
    return true;
}

void predictive_weights_from_phi(
    const OuGrid& grid,
    const std::vector<double>& phi,
    std::vector<double>& weights) {

    weights.assign(static_cast<std::size_t>(grid.K), 1.0 / static_cast<double>(grid.K));
    double total = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        weights[idx] = phi[idx] * grid.trap_w[idx];
        total += weights[idx];
    }
    if (total > 0.0 && std::isfinite(total)) {
        for (double& value : weights) {
            value /= total;
        }
    } else {
        std::fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(grid.K));
    }
}

}  // namespace scar_internal

