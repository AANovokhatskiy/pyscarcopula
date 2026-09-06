#include "scar/jacobi.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

bool close(double lhs, double rhs, double tolerance = 5e-12) {
    return std::abs(lhs - rhs)
        <= tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

}  // namespace

int run_jacobi_domain_tests() {
    const scar::JacobiParamsResult transformed =
        scar::jacobi_raw_to_physical({-3.0, -1.25, 0.8});
    if (!transformed.is_ok()
        || !close(transformed.value.kappa, 0.049787068367863944)
        || !close(transformed.value.m, 0.22270013882530884)
        || !close(transformed.value.xi, 2.225540928492468)) {
        return 1;
    }
    const scar::JacobiRawParamsResult raw =
        scar::jacobi_physical_to_raw({1.2, 0.4, 0.25}, 1e-6);
    if (!raw.is_ok()
        || !close(raw.value[0], std::log(1.2))
        || !close(raw.value[1], std::log(2.0 / 3.0))
        || !close(raw.value[2], std::log(0.25))) {
        return 2;
    }
    const scar::JacobiRawParamsResult raw_gradient =
        scar::jacobi_gradient_to_raw(
            {1.2, 0.4, 0.25}, {2.0, 3.0, 4.0});
    if (!raw_gradient.is_ok()
        || !close(raw_gradient.value[0], 2.4)
        || !close(raw_gradient.value[1], 0.72)
        || !close(raw_gradient.value[2], 1.0)
        || scar::jacobi_gradient_to_raw(
            {1.2, 1.0, 0.25}, {2.0, 3.0, 4.0}).is_ok()) {
        return 20;
    }
    const scar::JacobiParamsResult clipped_low =
        scar::jacobi_raw_to_physical({
            -scar::kJacobiRawClip - 1.0,
            -scar::kJacobiRawClip - 1.0,
            -scar::kJacobiRawClip - 1.0});
    const scar::JacobiParamsResult limit_low =
        scar::jacobi_raw_to_physical({
            -scar::kJacobiRawClip,
            -scar::kJacobiRawClip,
            -scar::kJacobiRawClip});
    const scar::JacobiParamsResult clipped_high =
        scar::jacobi_raw_to_physical({
            scar::kJacobiRawClip + 1.0,
            scar::kJacobiRawClip + 1.0,
            scar::kJacobiRawClip + 1.0});
    const scar::JacobiParamsResult limit_high =
        scar::jacobi_raw_to_physical({
            scar::kJacobiRawClip,
            scar::kJacobiRawClip,
            scar::kJacobiRawClip});
    if (!clipped_low.is_ok() || !limit_low.is_ok()
        || !clipped_high.is_ok() || !limit_high.is_ok()
        || clipped_low.value.kappa != limit_low.value.kappa
        || clipped_low.value.m != limit_low.value.m
        || clipped_low.value.xi != limit_low.value.xi
        || clipped_high.value.kappa != limit_high.value.kappa
        || clipped_high.value.m != limit_high.value.m
        || clipped_high.value.xi != limit_high.value.xi
        || scar::jacobi_raw_to_physical({
            std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}).is_ok()) {
        return 16;
    }
    const scar::JacobiShapeResult shape =
        scar::jacobi_stationary_shape({1.2, 0.4, 0.25});
    if (!shape.is_ok()
        || !close(shape.value.alpha, 15.36)
        || !close(shape.value.beta, 23.04)
        || scar::ok(scar::validate_jacobi_params({1.0, 0.0, 0.2}))) {
        return 3;
    }
    const std::vector<double> stationary_uniforms{0.0, 0.5};
    const scar::JacobiVectorResult stationary = scar::sample_jacobi_stationary(
        {1.0, 0.5, std::sqrt(0.5)}, stationary_uniforms);
    if (!stationary.is_ok()
        || stationary.value.size() != stationary_uniforms.size()
        || stationary.value[0] != 0.0
        || !close(stationary.value[1], 0.5, 2e-12)) {
        return 21;
    }
    const scar::JacobiVectorResult invalid_stationary =
        scar::sample_jacobi_stationary(
            {1.0, 0.5, std::sqrt(0.5)}, {1.0});
    if (invalid_stationary.status != scar::Status::InvalidParameter) {
        return 22;
    }

    scar::JacobiNumericalConfig memory_config;
    memory_config.quad_order = 16;
    memory_config.basis_order = 4;
    memory_config.n_obs = 8;
    memory_config.matrix = true;
    memory_config.gradient = true;
    const scar::JacobiMemoryResult memory =
        scar::estimate_jacobi_workspace(memory_config);
    memory_config.memory_budget_bytes = 1024;
    const scar::JacobiMemoryResult limited =
        scar::estimate_jacobi_workspace(memory_config);
    if (!memory.is_ok() || memory.value.bytes != 26368
        || limited.is_ok() || limited.value.within_budget
        || limited.value.bytes != memory.value.bytes) {
        return 4;
    }

    scar::JacobiNumericalConfig eigensolver_config;
    eigensolver_config.quad_order = 128;
    eigensolver_config.basis_order = 1;
    eigensolver_config.gh_order = 1;
    eigensolver_config.matrix = false;
    eigensolver_config.memory_budget_bytes = 100000;
    const scar::JacobiMemoryResult eigensolver_memory =
        scar::estimate_jacobi_workspace(eigensolver_config);
    if (eigensolver_memory.is_ok()
        || eigensolver_memory.value.within_budget
        || eigensolver_memory.value.bytes <= 128ULL * 128ULL * 8ULL
        || scar::build_jacobi_rule(
            2.5, 3.5, 128, 1, 100000).is_ok()
        || scar::gauss_jacobi_probability_rule(
            2.5, 3.5, 128, 100000).is_ok()
        || scar::gauss_hermite_probability_rule(128, 100000).is_ok()) {
        return 17;
    }

    scar::JacobiNumericalConfig limit_config;
    limit_config.quad_order = scar::kMaxJacobiOrder;
    limit_config.basis_order = scar::kMaxJacobiOrder;
    limit_config.gh_order = scar::kMaxJacobiOrder;
    if (!scar::ok(scar::validate_jacobi_config(limit_config))) {
        return 18;
    }
    limit_config.quad_order = scar::kMaxJacobiOrder + 1;
    if (scar::ok(scar::validate_jacobi_config(limit_config))
        || scar::gauss_jacobi_probability_rule(
            2.0, 3.0, scar::kMaxJacobiOrder + 1).is_ok()
        || scar::gauss_hermite_probability_rule(
            scar::kMaxJacobiOrder + 1).is_ok()) {
        return 19;
    }

    const double tau_values[] = {0.0, 0.2, 0.6, 1.0};
    for (double tau : tau_values) {
        const scar::JacobiScalarResult value = scar::jacobi_lamperti(tau, 0.25);
        const scar::JacobiScalarResult recovered =
            scar::jacobi_inverse_lamperti(value.value, 0.25);
        if (!value.is_ok() || !recovered.is_ok()
            || !close(recovered.value, tau)) {
            return 5;
        }
    }
    const scar::JacobiBoundaryResult reflected = scar::apply_jacobi_boundary(
        -0.25, 1.0, scar::JacobiBoundaryPolicy::Reflect);
    const scar::JacobiBoundaryResult clipped = scar::apply_jacobi_boundary(
        1.25, 1.0, scar::JacobiBoundaryPolicy::Clip);
    if (!reflected.is_ok() || !reflected.value.intervened
        || !close(reflected.value.value, 0.25)
        || !clipped.is_ok() || !clipped.value.intervened
        || !close(clipped.value.value, 1.0)) {
        return 6;
    }

    const double pi = std::acos(-1.0);
    if (!close(scar::jacobi_digamma(1.0).value,
               -0.5772156649015329, 2e-12)
        || !close(scar::jacobi_trigamma(1.0).value,
                  pi * pi / 6.0, 2e-12)
        || !close(scar::jacobi_log_beta(2.0, 3.0).value,
                  std::log(1.0 / 12.0), 2e-12)) {
        return 7;
    }

    const scar::JacobiQuadratureResult hermite =
        scar::gauss_hermite_probability_rule(7);
    if (!hermite.is_ok() || hermite.value.nodes.size() != 7) {
        return 8;
    }
    double h0 = 0.0;
    double h1 = 0.0;
    double h2 = 0.0;
    double h4 = 0.0;
    for (std::size_t index = 0; index < 7; ++index) {
        const double node = hermite.value.nodes[index];
        const double weight = hermite.value.weights[index];
        h0 += weight;
        h1 += weight * node;
        h2 += weight * node * node;
        h4 += weight * node * node * node * node;
    }
    if (!close(h0, 1.0) || !close(h1, 0.0)
        || !close(h2, 0.5) || !close(h4, 0.75)) {
        return 9;
    }

    const scar::JacobiBasisResult rule =
        scar::build_jacobi_rule(15.36, 23.04, 16, 4);
    if (!rule.is_ok() || rule.value.tau.size() != 16
        || rule.value.basis.size() != 64) {
        return 10;
    }
    double weight_sum = 0.0;
    double mean = 0.0;
    std::array<double, 16> gram{};
    for (std::size_t row = 0; row < 16; ++row) {
        const double weight = rule.value.weights[row];
        weight_sum += weight;
        mean += weight * rule.value.tau[row];
        for (std::size_t left = 0; left < 4; ++left) {
            for (std::size_t right = 0; right < 4; ++right) {
                gram[left * 4 + right] += weight
                    * rule.value.basis[row * 4 + left]
                    * rule.value.basis[row * 4 + right];
            }
        }
    }
    if (!close(weight_sum, 1.0) || !close(mean, 0.4)) {
        return 11;
    }
    for (std::size_t left = 0; left < 4; ++left) {
        for (std::size_t right = 0; right < 4; ++right) {
            const double expected = left == right ? 1.0 : 0.0;
            if (!close(gram[left * 4 + right], expected, 2e-11)) {
                return 12;
            }
        }
    }

    const scar::JacobiFixedRuleResult fixed =
        scar::build_fixed_jacobi_rule({1.2, 0.4, 0.25}, 16);
    if (!fixed.is_ok() || fixed.value.weight_derivatives.size() != 48) {
        return 13;
    }
    for (std::size_t parameter = 0; parameter < 3; ++parameter) {
        double derivative_sum = 0.0;
        for (std::size_t index = 0; index < 16; ++index) {
            derivative_sum +=
                fixed.value.weight_derivatives[parameter * 16 + index];
        }
        if (!close(derivative_sum, 0.0, 2e-11)) {
            return 14;
        }
    }

    if (scar::gauss_jacobi_probability_rule(-1.0, 2.0, 8).is_ok()
        || scar::build_jacobi_rule(2.0, 3.0, 4, 5).is_ok()
        || scar::jacobi_lamperti_drift(
            {1.0, 0.5, 0.2}, -0.1).is_ok()) {
        return 15;
    }
    return 0;
}
