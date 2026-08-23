#pragma once

#include "scar/copula.hpp"
#include "scar/detail/safety.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace scar_internal {

using scar::SCAR_INVALID_PARAMETER;
using scar::SCAR_INVALID_SIZE;
using scar::SCAR_OK;

inline int validate_multivariate_observations(
    const scar::CopulaSpec& spec,
    const scar::Observations& observations) {

    for (const auto& row : observations) {
        if (row.size() != static_cast<std::size_t>(spec.dim)) {
            return SCAR_INVALID_SIZE;
        }
        if (!std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            return SCAR_INVALID_PARAMETER;
        }
    }
    return SCAR_OK;
}

inline double parameter_at(
    const std::vector<double>& parameters,
    std::size_t row) {

    return parameters.size() == 1 ? parameters[0] : parameters[row];
}

inline void initialize_multivariate_grid(
    scar::MultivariateGridResult& out,
    std::size_t n_obs,
    std::size_t n_grid) {

    out.pdf.n_obs = static_cast<std::int64_t>(n_obs);
    out.pdf.n_grid = static_cast<std::int64_t>(n_grid);
    out.d_pdf_dx.n_obs = out.pdf.n_obs;
    out.d_pdf_dx.n_grid = out.pdf.n_grid;
    std::size_t elements = 0;
    if (!checked_size_mul(n_obs, n_grid, elements)) {
        out.status = SCAR_INVALID_SIZE;
        return;
    }
    out.pdf.values.assign(elements, 0.0);
    out.d_pdf_dx.values.assign(elements, 0.0);
}

}  // namespace scar_internal
