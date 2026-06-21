#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar_internal {

struct HermiteRuleCacheInfo {
    std::size_t entries = 0;
    std::size_t bytes = 0;
    std::size_t max_entries = 0;
    std::size_t max_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t insertions = 0;
    std::uint64_t evictions = 0;
    std::uint64_t oversized_skips = 0;
    std::uint64_t duplicate_builds = 0;
};

bool standard_normal_hermite_rule(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis);
bool standard_normal_hermite_rule_with_weighted_basis(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis,
    std::vector<double>& weighted_basis);
HermiteRuleCacheInfo hermite_rule_cache_info();
void clear_hermite_rule_cache();
void set_hermite_rule_cache_limits_for_testing(
    std::size_t max_entries,
    std::size_t max_bytes);
void reset_hermite_rule_cache_limits_for_testing();
bool physicists_hermite_normal_rule(
    int order,
    std::vector<double>& nodes,
    std::vector<double>& weights);
void project_multiply(
    const std::vector<double>& coeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    std::vector<double>& out);
void project_multiply_with_grad(
    const std::vector<double>& coeff,
    const std::vector<double>& dcoeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& dfi_dx_row,
    const std::vector<double>& dx_dalpha,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    std::vector<double>& out,
    std::vector<double>& dout);
void project_multiply_with_score_grad(
    const std::vector<double>& coeff,
    const std::vector<double>& dcoeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& scores,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    int n_params,
    std::vector<double>& out,
    std::vector<double>& dout);
void local_gh_matvec(
    const std::vector<double>& z,
    double rho,
    double sigma_cond,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& v,
    std::vector<double>& out);
void local_gh_predict_matvec(
    const std::vector<double>& z,
    const std::vector<double>& trap_w,
    double rho,
    double sigma_cond,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& source,
    std::vector<double>& out_density);

}  // namespace scar_internal
