#include "scar/detail/scar_ou/quadrature.hpp"

#include <cmath>
#include <vector>

namespace {

struct CacheLimitsGuard {
    ~CacheLimitsGuard() {
        scar_internal::reset_hermite_rule_cache_limits_for_testing();
    }
};

struct Rule {
    std::vector<double> nodes;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> weighted_basis;
};

bool build_rule(int quad_order, int basis_order, Rule& rule) {
    return scar_internal::standard_normal_hermite_rule_with_weighted_basis(
        quad_order, basis_order, rule.nodes, rule.weights, rule.basis,
        rule.weighted_basis);
}

}  // namespace

int run_ou_quadrature_cache_tests() {
    CacheLimitsGuard reset_on_exit;
    // Exercise both Newton and Golub-Welsch node generation. The reference
    // disables all cache storage, so a smaller basis cannot seed the nodes.
    for (int quad_order : {48, 256}) {
        scar_internal::set_hermite_rule_cache_limits_for_testing(0, 0);
        Rule cold;
        if (!build_rule(quad_order, 32, cold)) {
            return 1;
        }
        scar_internal::set_hermite_rule_cache_limits_for_testing(1, 200000);
        Rule seed, reused;
        if (!build_rule(quad_order, 8, seed)
            || !build_rule(quad_order, 32, reused)
            || reused.nodes != cold.nodes || reused.weights != cold.weights
            || reused.basis != cold.basis
            || reused.weighted_basis != cold.weighted_basis) {
            return 2;
        }
        const auto info = scar_internal::hermite_rule_cache_info();
        if (info.entries != 1 || info.bytes > info.max_bytes
            || info.insertions != 2 || info.evictions != 1
            || info.hits != 0 || info.misses != 2) {
            return 3;
        }
        // Also exercise the unweighted API, rebuilding a smaller basis from
        // cached nodes and checking exact prefix agreement in each row.
        Rule smaller;
        if (!scar_internal::standard_normal_hermite_rule(
                quad_order, 16, smaller.nodes, smaller.weights, smaller.basis)
            || smaller.nodes != cold.nodes || smaller.weights != cold.weights) {
            return 4;
        }
        for (int q = 0; q < quad_order; ++q) {
            for (int n = 0; n < 16; ++n) {
                if (smaller.basis[q * 16 + n] != cold.basis[q * 32 + n]) {
                    return 5;
                }
            }
        }
        // Independent standard-normal moment identities through degree eight.
        const double moments[] = {1.0, 1.0, 3.0, 15.0, 105.0};
        for (int degree = 0; degree <= 8; degree += 2) {
            double actual = 0.0;
            for (int q = 0; q < quad_order; ++q) {
                actual += reused.weights[q] * std::pow(reused.nodes[q], degree);
            }
            if (std::abs(actual - moments[degree / 2]) > 2e-10) {
                return 6;
            }
        }
    }
    return 0;
}
