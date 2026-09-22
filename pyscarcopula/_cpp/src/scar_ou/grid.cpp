#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

namespace scar_internal {

int gaussian_transition_band(int K, double kernel_grid_ratio, double a,
                             double radius, std::int64_t observations) {
    if (K < 2 || !std::isfinite(kernel_grid_ratio) || kernel_grid_ratio <= 0.0
        || !std::isfinite(a) || a <= 0.0 || !std::isfinite(radius)
        || radius <= 0.0 || observations < 2) return -1;
    const double variance = -std::expm1(-2.0 * a);
    const double rho = std::exp(-a);
    const double c = (a * rho) / variance;
    const double log_budget = std::log(kOuTransitionTailBudget)
        - std::log(static_cast<double>(observations - 1));
    // For |y| >= s, |d log p / d log a| <=
    // c [rho(1+y*y) + R sqrt(variance)|y|]. Integrating both tails and
    // using Mills' inequality bounds mass plus absolute tangent by
    // 2 phi(s) [1/s + c {rho(s+2/s) + R sqrt(variance)}].
    // At s>=5 these envelopes decrease. One extra grid cell makes the
    // rectangular lattice tail no larger than that continuous integral.
    // Fixed, bounded search; no emission-dependent tuning or row renormalization.
    for (int step = 0; step <= 140; ++step) {
        const double s = 5.0 + 0.25 * step;
        const double envelope = 1.0/s
            + c * (rho*(s+2.0/s) + radius*std::sqrt(variance));
        const double log_tail = std::log(2.0) - .5*std::log(2.0*kPi)
            - .5*s*s + std::log(envelope);
        if (log_tail <= log_budget) {
            const double band = std::ceil(s * kernel_grid_ratio) + 1.0;
            return band >= K-1 ? K-1 : static_cast<int>(band);
        }
    }
    return K-1;
}

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

    if (n_obs < 2 || K < 2 || grid_range <= 0.0 || pts_per_sigma <= 0
        || static_cast<std::size_t>(K) > kMaxGridSize) {
        return false;
    }
    if (max_K > 0
        && (max_K < 2 || static_cast<std::size_t>(max_K) > kMaxGridSize)) {
        return false;
    }
    if (!std::isfinite(kappa) || !std::isfinite(mu) || !std::isfinite(nu)
        || !std::isfinite(grid_range) || kappa <= 0.0 || nu <= 0.0) {
        return false;
    }

    const double dt = 1.0 / static_cast<double>(n_obs - 1);
    const double rho = std::exp(-kappa * dt);
    const double sigma = std::sqrt(0.5 * nu * nu / kappa);
    const double conditional_variance = -std::expm1(-2.0 * kappa * dt);
    const double sigma_cond = sigma * std::sqrt(conditional_variance);
    if (!std::isfinite(sigma) || !std::isfinite(sigma_cond)
        || sigma <= 0.0 || sigma_cond <= 0.0) {
        return false;
    }

    int K_adaptive = K;
    double K_requested = static_cast<double>(K);
    if (adaptive) {
        const double dz_target = sigma_cond / static_cast<double>(pts_per_sigma);
        const double K_min_value =
            std::ceil(2.0 * grid_range * sigma / dz_target) + 1.0;
        if (!std::isfinite(K_min_value) || K_min_value < 2.0) {
            return false;
        }
        K_requested = std::max(K_requested, K_min_value);
        if (K_min_value > static_cast<double>(kMaxGridSize)
            || K_min_value > static_cast<double>(INT_MAX)) {
            if (max_K <= 0) {
                return false;
            }
            K_adaptive = max_K == INT_MAX ? INT_MAX : max_K + 1;
        } else {
            const int K_min = static_cast<int>(K_min_value);
            K_adaptive = std::max(K, K_min);
        }
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
    grid.a = kappa * dt;
    grid.observations = n_obs;
    grid.K_requested = K_requested;
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
    grid.r_kernel_grid =
        std::sqrt(conditional_variance)
        * static_cast<double>(K_eff - 1)
        / (2.0 * grid_range);

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

bool predictive_weights_from_phi(
    const OuGrid& grid,
    const std::vector<double>& phi,
    std::vector<double>& weights) {

    if (grid.K <= 0
        || phi.size() != static_cast<std::size_t>(grid.K)
        || grid.trap_w.size() != static_cast<std::size_t>(grid.K)) {
        weights.clear();
        return false;
    }
    weights.assign(static_cast<std::size_t>(grid.K), 0.0);
    double scale = 0.0;
    for (double value : phi) {
        if (!std::isfinite(value)) {
            return false;
        }
        scale = std::max(scale, value);
    }
    if (scale <= 0.0) {
        return false;
    }
    const double negative_tolerance = 1e-12 * scale;
    double total = 0.0;
    for (int j = 0; j < grid.K; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        if (!std::isfinite(grid.trap_w[idx]) || grid.trap_w[idx] <= 0.0
            || phi[idx] < -negative_tolerance) {
            return false;
        }
        weights[idx] = std::max(phi[idx], 0.0) * grid.trap_w[idx];
        total += weights[idx];
    }
    if (!std::isfinite(total) || total <= 0.0) {
        return false;
    }
    for (double& value : weights) {
        value /= total;
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

}  // namespace scar_internal
