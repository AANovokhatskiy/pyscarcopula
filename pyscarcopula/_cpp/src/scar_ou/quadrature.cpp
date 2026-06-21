#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace scar_internal {
namespace {

double pythag(double a, double b) {
    return std::hypot(a, b);
}

bool hermite_rule_invariants_hold(
    const std::vector<double>& z,
    const std::vector<double>& weights,
    const std::vector<double>& basis,
    int quad_order,
    int basis_order,
    double orthogonality_tol = 1e-8) {

    std::size_t basis_elements = 0;
    if (!valid_spectral_dimensions(
            quad_order, basis_order, basis_elements)
        || z.size() != static_cast<std::size_t>(quad_order)
        || weights.size() != static_cast<std::size_t>(quad_order)
        || basis.size() != basis_elements) {
        return false;
    }

    double weight_sum = 0.0;
    for (int q = 0; q < quad_order; ++q) {
        const double node = z[static_cast<std::size_t>(q)];
        const double weight = weights[static_cast<std::size_t>(q)];
        if (!std::isfinite(node) || !std::isfinite(weight) || weight < 0.0) {
            return false;
        }
        weight_sum += weight;
    }
    if (!std::isfinite(weight_sum) || std::abs(weight_sum - 1.0) > 1e-10) {
        return false;
    }

    const double symmetry_tol = 1e-10;
    for (int q = 0; q < quad_order; ++q) {
        const int r = quad_order - 1 - q;
        if (std::abs(z[static_cast<std::size_t>(q)]
                + z[static_cast<std::size_t>(r)]) > symmetry_tol
            || std::abs(weights[static_cast<std::size_t>(q)]
                - weights[static_cast<std::size_t>(r)]) > symmetry_tol) {
            return false;
        }
    }

    double max_orthogonality_error = 0.0;
    for (int a = 0; a < basis_order; ++a) {
        for (int b = 0; b <= a; ++b) {
            double inner = 0.0;
            for (int q = 0; q < quad_order; ++q) {
                const std::size_t base =
                    static_cast<std::size_t>(q)
                    * static_cast<std::size_t>(basis_order);
                inner += weights[static_cast<std::size_t>(q)]
                    * basis[base + static_cast<std::size_t>(a)]
                    * basis[base + static_cast<std::size_t>(b)];
            }
            const double target = a == b ? 1.0 : 0.0;
            max_orthogonality_error = std::max(
                max_orthogonality_error, std::abs(inner - target));
        }
    }
    return max_orthogonality_error <= orthogonality_tol;
}

bool tridiagonal_ql(
    std::vector<double>& d,
    std::vector<double>& e,
    std::vector<double>& z,
    int n) {

    if (n <= 0) {
        return false;
    }
    e[static_cast<std::size_t>(n - 1)] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;
        int m = l;
        do {
            for (m = l; m < n - 1; ++m) {
                const double dd = std::abs(d[static_cast<std::size_t>(m)])
                    + std::abs(d[static_cast<std::size_t>(m + 1)]);
                if (std::abs(e[static_cast<std::size_t>(m)])
                    <= std::numeric_limits<double>::epsilon() * dd) {
                    break;
                }
            }
            if (m == l) {
                break;
            }
            if (++iter > 80) {
                return false;
            }

            double g = (d[static_cast<std::size_t>(l + 1)]
                - d[static_cast<std::size_t>(l)])
                / (2.0 * e[static_cast<std::size_t>(l)]);
            double r = pythag(g, 1.0);
            g = d[static_cast<std::size_t>(m)]
                - d[static_cast<std::size_t>(l)]
                + e[static_cast<std::size_t>(l)]
                / (g + std::copysign(r, g));

            double s = 1.0;
            double c = 1.0;
            double p = 0.0;
            int i = 0;
            bool early_break = false;
            for (i = m - 1; i >= l; --i) {
                const double f = s * e[static_cast<std::size_t>(i)];
                const double b = c * e[static_cast<std::size_t>(i)];
                r = pythag(f, g);
                e[static_cast<std::size_t>(i + 1)] = r;
                if (r == 0.0) {
                    d[static_cast<std::size_t>(i + 1)] -= p;
                    e[static_cast<std::size_t>(m)] = 0.0;
                    early_break = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[static_cast<std::size_t>(i + 1)] - p;
                r = (d[static_cast<std::size_t>(i)] - g) * s + 2.0 * c * b;
                p = s * r;
                d[static_cast<std::size_t>(i + 1)] = g + p;
                g = c * r - b;

                for (int k = 0; k < n; ++k) {
                    const std::size_t row =
                        static_cast<std::size_t>(k)
                        * static_cast<std::size_t>(n);
                    const std::size_t ki1 =
                        row + static_cast<std::size_t>(i + 1);
                    const std::size_t ki =
                        row + static_cast<std::size_t>(i);
                    const double fz = z[ki1];
                    z[ki1] = s * z[ki] + c * fz;
                    z[ki] = c * z[ki] - s * fz;
                }
            }
            if (early_break) {
                continue;
            }
            d[static_cast<std::size_t>(l)] -= p;
            e[static_cast<std::size_t>(l)] = g;
            e[static_cast<std::size_t>(m)] = 0.0;
        } while (m != l);
    }
    return true;
}

bool standard_normal_hermite_rule_golub_welsch(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis) {

    std::size_t basis_elements = 0;
    std::size_t eigenvector_elements = 0;
    if (!valid_spectral_dimensions(
            quad_order, basis_order, basis_elements)
        || !checked_size_mul(
            static_cast<std::size_t>(quad_order),
            static_cast<std::size_t>(quad_order),
            eigenvector_elements)) {
        return false;
    }

    std::vector<double> d(static_cast<std::size_t>(quad_order), 0.0);
    std::vector<double> e(static_cast<std::size_t>(quad_order), 0.0);
    std::vector<double> eigenvectors(eigenvector_elements, 0.0);

    for (int i = 0; i < quad_order - 1; ++i) {
        e[static_cast<std::size_t>(i)] = std::sqrt(static_cast<double>(i + 1));
    }
    for (int i = 0; i < quad_order; ++i) {
        eigenvectors[
            static_cast<std::size_t>(i)
                * static_cast<std::size_t>(quad_order)
            + static_cast<std::size_t>(i)] = 1.0;
    }

    if (!tridiagonal_ql(d, e, eigenvectors, quad_order)) {
        return false;
    }

    std::vector<int> order(static_cast<std::size_t>(quad_order), 0);
    for (int i = 0; i < quad_order; ++i) {
        order[static_cast<std::size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return d[static_cast<std::size_t>(lhs)] < d[static_cast<std::size_t>(rhs)];
    });

    for (int out = 0; out < quad_order; ++out) {
        const int col = order[static_cast<std::size_t>(out)];
        const double node = d[static_cast<std::size_t>(col)];
        const double first_component =
            eigenvectors[static_cast<std::size_t>(col)];
        z[static_cast<std::size_t>(out)] = node;
        weights[static_cast<std::size_t>(out)] = first_component * first_component;
        if (!std::isfinite(node)
            || !std::isfinite(weights[static_cast<std::size_t>(out)])) {
            return false;
        }
    }

    basis.assign(basis_elements, 0.0);
    for (int q = 0; q < quad_order; ++q) {
        const std::size_t base =
            static_cast<std::size_t>(q)
            * static_cast<std::size_t>(basis_order);
        basis[base] = 1.0;
        if (basis_order > 1) {
            basis[base + 1] =
                z[static_cast<std::size_t>(q)];
        }
        for (int n = 1; n < basis_order - 1; ++n) {
            basis[base + static_cast<std::size_t>(n + 1)] =
                (z[static_cast<std::size_t>(q)]
                    * basis[base + static_cast<std::size_t>(n)]
                 - std::sqrt(static_cast<double>(n))
                    * basis[base + static_cast<std::size_t>(n - 1)])
                / std::sqrt(static_cast<double>(n + 1));
        }
    }
    return hermite_rule_invariants_hold(
        z, weights, basis, quad_order, basis_order);
}

struct CachedHermiteRule {
    std::vector<double> z;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> weighted_basis;
};

struct HermiteRuleCacheEntry {
    std::shared_ptr<const CachedHermiteRule> rule;
    std::size_t bytes = 0;
    std::list<std::uint64_t>::iterator lru_position;
};

struct HermiteRuleCache {
    std::unordered_map<std::uint64_t, HermiteRuleCacheEntry> entries;
    std::list<std::uint64_t> lru;
    std::size_t bytes = 0;
    std::size_t max_entries = kHermiteRuleCacheMaxEntries;
    std::size_t max_bytes = kHermiteRuleCacheMaxBytes;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t insertions = 0;
    std::uint64_t evictions = 0;
    std::uint64_t oversized_skips = 0;
    std::uint64_t duplicate_builds = 0;
};

std::uint64_t hermite_rule_cache_key(int quad_order, int basis_order) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(quad_order)) << 32)
        | static_cast<std::uint32_t>(basis_order);
}

HermiteRuleCache& hermite_rule_cache() {
    static HermiteRuleCache cache;
    return cache;
}

std::mutex& hermite_rule_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::size_t vector_storage_bytes(const std::vector<double>& values) {
    std::size_t bytes = 0;
    if (!checked_size_mul(values.capacity(), sizeof(double), bytes)) {
        return std::numeric_limits<std::size_t>::max();
    }
    return bytes;
}

std::size_t cached_rule_bytes(const CachedHermiteRule& rule) {
    std::size_t total = 0;
    for (const std::vector<double>* values : {
             &rule.z, &rule.weights, &rule.basis, &rule.weighted_basis}) {
        const std::size_t bytes = vector_storage_bytes(*values);
        if (!checked_size_add(total, bytes, total)) {
            return std::numeric_limits<std::size_t>::max();
        }
    }
    return total;
}

void touch_cache_entry(
    HermiteRuleCache& cache,
    HermiteRuleCacheEntry& entry) {

    cache.lru.splice(
        cache.lru.begin(), cache.lru, entry.lru_position);
    entry.lru_position = cache.lru.begin();
}

void evict_lru_entry(HermiteRuleCache& cache) {
    if (cache.lru.empty()) {
        return;
    }
    const std::uint64_t key = cache.lru.back();
    const auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
        cache.bytes -= it->second.bytes;
        cache.entries.erase(it);
        ++cache.evictions;
    }
    cache.lru.pop_back();
}

void reset_cache_statistics(HermiteRuleCache& cache) {
    cache.hits = 0;
    cache.misses = 0;
    cache.insertions = 0;
    cache.evictions = 0;
    cache.oversized_skips = 0;
    cache.duplicate_builds = 0;
}

void clear_cache_contents(HermiteRuleCache& cache) {
    cache.entries.clear();
    cache.lru.clear();
    cache.bytes = 0;
    reset_cache_statistics(cache);
}

std::shared_ptr<const CachedHermiteRule> find_cached_hermite_rule(
    int quad_order,
    int basis_order) {

    const auto key = hermite_rule_cache_key(quad_order, basis_order);
    std::lock_guard<std::mutex> lock(hermite_rule_cache_mutex());
    auto& cache = hermite_rule_cache();
    const auto it = cache.entries.find(key);
    if (it == cache.entries.end()) {
        ++cache.misses;
        return {};
    }
    ++cache.hits;
    touch_cache_entry(cache, it->second);
    return it->second.rule;
}

bool load_cached_hermite_rule(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis) {

    const auto rule = find_cached_hermite_rule(quad_order, basis_order);
    if (!rule) {
        return false;
    }
    z = rule->z;
    weights = rule->weights;
    basis = rule->basis;
    return true;
}

bool load_cached_hermite_rule(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis,
    std::vector<double>& weighted_basis) {

    const auto rule = find_cached_hermite_rule(quad_order, basis_order);
    if (!rule) {
        return false;
    }
    z = rule->z;
    weights = rule->weights;
    basis = rule->basis;
    weighted_basis = rule->weighted_basis;
    return true;
}

void store_cached_hermite_rule(
    int quad_order,
    int basis_order,
    const std::vector<double>& z,
    const std::vector<double>& weights,
    const std::vector<double>& basis,
    const std::vector<double>* precomputed_weighted_basis = nullptr) {

    const auto key = hermite_rule_cache_key(quad_order, basis_order);
    std::vector<double> weighted_basis;
    if (precomputed_weighted_basis != nullptr) {
        weighted_basis = *precomputed_weighted_basis;
    } else {
        weighted_basis.assign(basis.size(), 0.0);
        for (int q = 0; q < quad_order; ++q) {
            const double weight = weights[static_cast<std::size_t>(q)];
            const std::size_t base =
                static_cast<std::size_t>(q)
                * static_cast<std::size_t>(basis_order);
            for (int n = 0; n < basis_order; ++n) {
                const std::size_t idx = base + static_cast<std::size_t>(n);
                weighted_basis[idx] = weight * basis[idx];
            }
        }
    }
    auto rule = std::make_shared<CachedHermiteRule>(
        CachedHermiteRule{z, weights, basis, std::move(weighted_basis)});
    const std::size_t entry_bytes = cached_rule_bytes(*rule);

    std::lock_guard<std::mutex> lock(hermite_rule_cache_mutex());
    auto& cache = hermite_rule_cache();
    const auto existing = cache.entries.find(key);
    if (existing != cache.entries.end()) {
        ++cache.duplicate_builds;
        touch_cache_entry(cache, existing->second);
        return;
    }
    if (cache.max_entries == 0 || entry_bytes > cache.max_bytes) {
        ++cache.oversized_skips;
        return;
    }
    while (!cache.entries.empty()
           && (cache.entries.size() >= cache.max_entries
               || cache.bytes > cache.max_bytes - entry_bytes)) {
        evict_lru_entry(cache);
    }

    cache.lru.push_front(key);
    try {
        cache.entries.emplace(
            key,
            HermiteRuleCacheEntry{
                std::move(rule), entry_bytes, cache.lru.begin()});
    } catch (...) {
        cache.lru.pop_front();
        throw;
    }
    cache.bytes += entry_bytes;
    ++cache.insertions;
}

bool standard_normal_hermite_rule_uncached(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis) {

    std::size_t basis_elements = 0;
    if (!valid_spectral_dimensions(
            quad_order, basis_order, basis_elements)) {
        return false;
    }

    z.assign(static_cast<std::size_t>(quad_order), 0.0);
    weights.assign(static_cast<std::size_t>(quad_order), 0.0);
    basis.assign(basis_elements, 0.0);

    if (quad_order >= 200) {
        return standard_normal_hermite_rule_golub_welsch(
            quad_order, basis_order, z, weights, basis);
    }

    const int m = (quad_order + 1) / 2;
    double root = 0.0;
    std::vector<double> positive_roots;
    positive_roots.reserve(static_cast<std::size_t>(m));

    for (int i = 0; i < m; ++i) {
        if (i == 0) {
            root = std::sqrt(2.0 * quad_order + 1.0)
                - 1.85575 * std::pow(2.0 * quad_order + 1.0, -1.0 / 6.0);
        } else if (i == 1) {
            root -= 1.14 * std::pow(static_cast<double>(quad_order), 0.426) / root;
        } else if (i == 2) {
            root = 1.86 * positive_roots[1] - 0.86 * positive_roots[0];
        } else if (i == 3) {
            root = 1.91 * positive_roots[2] - 0.91 * positive_roots[1];
        } else {
            root = 2.0 * positive_roots[static_cast<std::size_t>(i - 1)]
                - positive_roots[static_cast<std::size_t>(i - 2)];
        }

        double p1 = 0.0;
        double p2 = 0.0;
        double derivative = 0.0;
        bool converged = false;
        for (int iter = 0; iter < 20; ++iter) {
            p1 = std::pow(kPi, -0.25);
            p2 = 0.0;
            for (int j = 1; j <= quad_order; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = root * std::sqrt(2.0 / static_cast<double>(j)) * p2
                    - std::sqrt(static_cast<double>(j - 1) / static_cast<double>(j)) * p3;
            }
            derivative = std::sqrt(2.0 * quad_order) * p2;
            if (derivative == 0.0) {
                return false;
            }
            const double delta = p1 / derivative;
            root -= delta;
            if (std::abs(delta) <= 1e-14) {
                converged = true;
                break;
            }
        }
        if (!converged || !std::isfinite(root) || derivative == 0.0) {
            return false;
        }

        positive_roots.push_back(root);

        const int left = i;
        const int right = quad_order - 1 - i;
        const double prob_node = std::sqrt(2.0) * root;
        const double normal_weight = 2.0 / (derivative * derivative) / std::sqrt(kPi);
        z[static_cast<std::size_t>(left)] = -prob_node;
        z[static_cast<std::size_t>(right)] = prob_node;
        weights[static_cast<std::size_t>(left)] = normal_weight;
        weights[static_cast<std::size_t>(right)] = normal_weight;
    }

    for (int q = 0; q < quad_order; ++q) {
        const std::size_t base =
            static_cast<std::size_t>(q)
            * static_cast<std::size_t>(basis_order);
        basis[base] = 1.0;
        if (basis_order > 1) {
            basis[base + 1] = z[static_cast<std::size_t>(q)];
        }
        for (int n = 1; n < basis_order - 1; ++n) {
            basis[base + static_cast<std::size_t>(n + 1)] =
                (z[static_cast<std::size_t>(q)]
                    * basis[base + static_cast<std::size_t>(n)]
                 - std::sqrt(static_cast<double>(n))
                    * basis[base + static_cast<std::size_t>(n - 1)])
                / std::sqrt(static_cast<double>(n + 1));
        }
    }

    return hermite_rule_invariants_hold(
        z, weights, basis, quad_order, basis_order);
}

}  // namespace

HermiteRuleCacheInfo hermite_rule_cache_info() {
    std::lock_guard<std::mutex> lock(hermite_rule_cache_mutex());
    const auto& cache = hermite_rule_cache();
    return {
        cache.entries.size(),
        cache.bytes,
        cache.max_entries,
        cache.max_bytes,
        cache.hits,
        cache.misses,
        cache.insertions,
        cache.evictions,
        cache.oversized_skips,
        cache.duplicate_builds,
    };
}

void clear_hermite_rule_cache() {
    std::lock_guard<std::mutex> lock(hermite_rule_cache_mutex());
    clear_cache_contents(hermite_rule_cache());
}

void set_hermite_rule_cache_limits_for_testing(
    std::size_t max_entries,
    std::size_t max_bytes) {

    std::lock_guard<std::mutex> lock(hermite_rule_cache_mutex());
    auto& cache = hermite_rule_cache();
    clear_cache_contents(cache);
    cache.max_entries = max_entries;
    cache.max_bytes = max_bytes;
}

void reset_hermite_rule_cache_limits_for_testing() {
    std::lock_guard<std::mutex> lock(hermite_rule_cache_mutex());
    auto& cache = hermite_rule_cache();
    clear_cache_contents(cache);
    cache.max_entries = kHermiteRuleCacheMaxEntries;
    cache.max_bytes = kHermiteRuleCacheMaxBytes;
}

bool standard_normal_hermite_rule(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis) {

    std::size_t basis_elements = 0;
    if (!valid_spectral_dimensions(
            quad_order, basis_order, basis_elements)) {
        return false;
    }
    if (load_cached_hermite_rule(quad_order, basis_order, z, weights, basis)) {
        return true;
    }
    if (!standard_normal_hermite_rule_uncached(
            quad_order, basis_order, z, weights, basis)) {
        return false;
    }
    store_cached_hermite_rule(quad_order, basis_order, z, weights, basis);
    return true;
}

bool standard_normal_hermite_rule_with_weighted_basis(
    int quad_order,
    int basis_order,
    std::vector<double>& z,
    std::vector<double>& weights,
    std::vector<double>& basis,
    std::vector<double>& weighted_basis) {

    std::size_t basis_elements = 0;
    if (!valid_spectral_dimensions(
            quad_order, basis_order, basis_elements)) {
        return false;
    }
    if (load_cached_hermite_rule(
            quad_order, basis_order, z, weights, basis, weighted_basis)) {
        return true;
    }
    if (!standard_normal_hermite_rule_uncached(
            quad_order, basis_order, z, weights, basis)) {
        return false;
    }
    weighted_basis.assign(basis.size(), 0.0);
    for (int q = 0; q < quad_order; ++q) {
        const double weight = weights[static_cast<std::size_t>(q)];
        const std::size_t base =
            static_cast<std::size_t>(q)
            * static_cast<std::size_t>(basis_order);
        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = base + static_cast<std::size_t>(n);
            weighted_basis[idx] = weight * basis[idx];
        }
    }
    store_cached_hermite_rule(
        quad_order, basis_order, z, weights, basis, &weighted_basis);
    return true;
}

bool physicists_hermite_normal_rule(
    int order,
    std::vector<double>& nodes,
    std::vector<double>& weights) {

    std::vector<double> unused_basis;
    if (!standard_normal_hermite_rule(order, 1, nodes, weights, unused_basis)) {
        return false;
    }
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (double& node : nodes) {
        node *= inv_sqrt2;
    }
    return true;
}

void project_multiply(
    const std::vector<double>& coeff,
    const std::vector<double>& fi_row,
    const std::vector<double>& basis,
    const std::vector<double>& weighted_basis,
    int quad_order,
    int basis_order,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    const double* coeff_ptr = coeff.data();
    double* out_ptr = out.data();
    for (int q = 0; q < quad_order; ++q) {
        double value = 0.0;
        const std::size_t base =
            static_cast<std::size_t>(q)
            * static_cast<std::size_t>(basis_order);
        const double* basis_row = basis.data() + base;
        const double* weighted_row = weighted_basis.data() + base;
        for (int n = 0; n < basis_order; ++n) {
            value += basis_row[n] * coeff_ptr[n];
        }
        const double factor = fi_row[static_cast<std::size_t>(q)] * value;
        for (int n = 0; n < basis_order; ++n) {
            out_ptr[n] += weighted_row[n] * factor;
        }
    }
}

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
    std::vector<double>& dout) {

    std::fill(out.begin(), out.end(), 0.0);
    std::fill(dout.begin(), dout.end(), 0.0);
    const double* coeff_ptr = coeff.data();
    const double* dcoeff0 = dcoeff.data();
    const double* dcoeff1 =
        dcoeff.data() + static_cast<std::size_t>(basis_order);
    const double* dcoeff2 =
        dcoeff.data() + 2 * static_cast<std::size_t>(basis_order);
    double* out_ptr = out.data();
    double* dout0 = dout.data();
    double* dout1 = dout.data() + static_cast<std::size_t>(basis_order);
    double* dout2 =
        dout.data() + 2 * static_cast<std::size_t>(basis_order);
    const double* dx0_ptr = dx_dalpha.data();
    const double* dx1_ptr =
        dx_dalpha.data() + static_cast<std::size_t>(quad_order);
    const double* dx2_ptr =
        dx_dalpha.data() + 2 * static_cast<std::size_t>(quad_order);

    for (int q = 0; q < quad_order; ++q) {
        const std::size_t basis_base =
            static_cast<std::size_t>(q)
            * static_cast<std::size_t>(basis_order);
        const double* basis_row = basis.data() + basis_base;
        const double* weighted_row = weighted_basis.data() + basis_base;
        double value = 0.0;
        double dvalue0 = 0.0;
        double dvalue1 = 0.0;
        double dvalue2 = 0.0;

        for (int n = 0; n < basis_order; ++n) {
            const double basis_value = basis_row[n];
            value += basis_value * coeff_ptr[n];
            dvalue0 += basis_value * dcoeff0[n];
            dvalue1 += basis_value * dcoeff1[n];
            dvalue2 += basis_value * dcoeff2[n];
        }

        const double fi = fi_row[static_cast<std::size_t>(q)];
        const double dfi = dfi_dx_row[static_cast<std::size_t>(q)];
        const double out_factor = fi * value;
        const double dout0_factor = dfi * dx0_ptr[q] * value + fi * dvalue0;
        const double dout1_factor = dfi * dx1_ptr[q] * value + fi * dvalue1;
        const double dout2_factor = dfi * dx2_ptr[q] * value + fi * dvalue2;

        for (int n = 0; n < basis_order; ++n) {
            const double weighted_basis_value = weighted_row[n];
            out_ptr[n] += weighted_basis_value * out_factor;
            dout0[n] += weighted_basis_value * dout0_factor;
            dout1[n] += weighted_basis_value * dout1_factor;
            dout2[n] += weighted_basis_value * dout2_factor;
        }
    }
}

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
    std::vector<double>& dout) {

    std::fill(out.begin(), out.end(), 0.0);
    std::fill(dout.begin(), dout.end(), 0.0);
    for (int q = 0; q < quad_order; ++q) {
        const std::size_t basis_base =
            static_cast<std::size_t>(q)
            * static_cast<std::size_t>(basis_order);
        const double* basis_row = basis.data() + basis_base;
        const double* weighted_row = weighted_basis.data() + basis_base;
        double value = 0.0;
        for (int n = 0; n < basis_order; ++n) {
            value += basis_row[n] * coeff[static_cast<std::size_t>(n)];
        }

        const double fi = fi_row[static_cast<std::size_t>(q)];
        for (int n = 0; n < basis_order; ++n) {
            out[static_cast<std::size_t>(n)] +=
                weighted_row[n] * fi * value;
        }
        for (int p = 0; p < n_params; ++p) {
            const std::size_t param_base =
                static_cast<std::size_t>(p)
                * static_cast<std::size_t>(basis_order);
            double dvalue = 0.0;
            for (int n = 0; n < basis_order; ++n) {
                dvalue += basis_row[n] * dcoeff[
                    param_base + static_cast<std::size_t>(n)];
            }
            const double factor = fi * (
                scores[
                    static_cast<std::size_t>(q)
                        * static_cast<std::size_t>(n_params)
                    + static_cast<std::size_t>(p)]
                * value
                + dvalue);
            for (int n = 0; n < basis_order; ++n) {
                dout[param_base + static_cast<std::size_t>(n)] +=
                    weighted_row[n] * factor;
            }
        }
    }
}

void local_gh_matvec(
    const std::vector<double>& z,
    double rho,
    double sigma_cond,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& v,
    std::vector<double>& out) {

    const int K = static_cast<int>(z.size());
    const double z0 = z.front();
    const double z_last = z.back();
    const double dz = z[1] - z[0];

    std::fill(out.begin(), out.end(), 0.0);
    for (int i = 0; i < K; ++i) {
        const double center = rho * z[static_cast<std::size_t>(i)];
        double acc = 0.0;
        for (std::size_t q = 0; q < gh_nodes.size(); ++q) {
            const double y = center + std::sqrt(2.0) * sigma_cond * gh_nodes[q];
            const double weight = gh_weights[q];

            if (y <= z0) {
                acc += weight * v[0];
            } else if (y >= z_last) {
                acc += weight * v[static_cast<std::size_t>(K - 1)];
            } else {
                int left = static_cast<int>(std::floor((y - z0) / dz));
                if (left >= K - 1) {
                    acc += weight * v[static_cast<std::size_t>(K - 1)];
                } else {
                    const double lam = (y - z[static_cast<std::size_t>(left)]) / dz;
                    acc += weight * (
                        (1.0 - lam) * v[static_cast<std::size_t>(left)]
                        + lam * v[static_cast<std::size_t>(left + 1)]);
                }
            }
        }
        out[static_cast<std::size_t>(i)] = acc;
    }
}

void local_gh_predict_matvec(
    const std::vector<double>& z,
    const std::vector<double>& trap_w,
    double rho,
    double sigma_cond,
    const std::vector<double>& gh_nodes,
    const std::vector<double>& gh_weights,
    const std::vector<double>& source,
    std::vector<double>& out_density) {

    const int K = static_cast<int>(z.size());
    const double z0 = z.front();
    const double z_last = z.back();
    const double dz = z[1] - z[0];

    std::fill(out_density.begin(), out_density.end(), 0.0);
    for (int i = 0; i < K; ++i) {
        const double center = rho * z[static_cast<std::size_t>(i)];
        for (std::size_t q = 0; q < gh_nodes.size(); ++q) {
            const double y = center + std::sqrt(2.0) * sigma_cond * gh_nodes[q];
            const double weighted_source = gh_weights[q] * source[static_cast<std::size_t>(i)];

            if (y <= z0) {
                out_density[0] += weighted_source;
            } else if (y >= z_last) {
                out_density[static_cast<std::size_t>(K - 1)] += weighted_source;
            } else {
                int left = static_cast<int>(std::floor((y - z0) / dz));
                if (left >= K - 1) {
                    out_density[static_cast<std::size_t>(K - 1)] += weighted_source;
                } else {
                    const double lam = (y - z[static_cast<std::size_t>(left)]) / dz;
                    out_density[static_cast<std::size_t>(left)] +=
                        weighted_source * (1.0 - lam);
                    out_density[static_cast<std::size_t>(left + 1)] +=
                        weighted_source * lam;
                }
            }
        }
    }

    for (int j = 0; j < K; ++j) {
        out_density[static_cast<std::size_t>(j)] /= trap_w[static_cast<std::size_t>(j)];
    }
}

}  // namespace scar_internal
