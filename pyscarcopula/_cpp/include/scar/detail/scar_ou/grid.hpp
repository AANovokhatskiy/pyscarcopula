#pragma once

#include "scar/copula.hpp"

#include <cstdint>
#include <vector>

namespace scar_internal {

struct OuGrid {
    int K = 0;
    double rho = 0.0;
    double sigma = 0.0;
    double sigma_cond = 0.0;
    double dz = 0.0;
    double r_kernel_grid = 0.0;
    bool adaptive_was_capped = false;
    std::vector<double> z;
    std::vector<double> x_grid;
    std::vector<double> trap_w;
    std::vector<double> p0;
};

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
    OuGrid& grid);
bool predictive_weights_from_phi(
    const OuGrid& grid,
    const std::vector<double>& phi,
    std::vector<double>& weights);
void copula_fi_row_on_grid(
    const scar::CopulaSpec& spec,
    const double* u,
    std::int64_t t,
    const std::vector<double>& x_grid,
    std::vector<double>& fi_row);

}  // namespace scar_internal
