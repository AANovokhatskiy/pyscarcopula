#pragma once

#include "scar/core/result.hpp"

#include <vector>

namespace scar {

using OuParameterVectorResult = Result<std::vector<double>>;

/// Map (kappa, mu, nu, ...) to (log kappa, mu, log sigma_x, ...).
OuParameterVectorResult ou_to_log_stationary(
    const std::vector<double>& physical) noexcept;

/// Map (log kappa, mu, log sigma_x, ...) to (kappa, mu, nu, ...).
OuParameterVectorResult ou_from_log_stationary(
    const std::vector<double>& optimizer) noexcept;

/// Pull a physical-coordinate gradient back to log-stationary coordinates.
OuParameterVectorResult ou_gradient_to_log_stationary(
    const std::vector<double>& physical,
    const std::vector<double>& gradient) noexcept;

/// Map a log-stationary gradient back to physical OU coordinates.
OuParameterVectorResult ou_gradient_from_log_stationary(
    const std::vector<double>& physical,
    const std::vector<double>& gradient) noexcept;

/// Project the first three coordinates onto an optimizer box.
OuParameterVectorResult ou_project_optimizer_block(
    const std::vector<double>& values,
    const std::vector<double>& lower,
    const std::vector<double>& upper) noexcept;

/// Convert scaled optimizer coordinates to physical coordinates.
OuParameterVectorResult optimizer_scaled_to_physical(
    const std::vector<double>& values,
    const std::vector<double>& scale) noexcept;

/// Convert physical coordinates to scaled optimizer coordinates.
OuParameterVectorResult physical_to_optimizer_scaled(
    const std::vector<double>& values,
    const std::vector<double>& scale) noexcept;

/// Pull a physical-coordinate gradient back through x = scale * z.
OuParameterVectorResult gradient_to_optimizer_scaled(
    const std::vector<double>& gradient,
    const std::vector<double>& scale) noexcept;

/// Map a scaled optimizer gradient back to physical coordinates.
OuParameterVectorResult gradient_from_optimizer_scaled(
    const std::vector<double>& gradient,
    const std::vector<double>& scale) noexcept;

}  // namespace scar
