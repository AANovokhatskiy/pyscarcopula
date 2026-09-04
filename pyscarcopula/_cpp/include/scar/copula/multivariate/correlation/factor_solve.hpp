#pragma once

#include <cstddef>

namespace scar {
class FactorCorrelationOperator;
}

namespace scar_internal {

// Internal row solve. Workspace must not overlap values/output.
// Workspace contains at least rank doubles.
// Values and output may alias. Workspace is reset on every call.
void factor_solve_row_with_workspace(
    const scar::FactorCorrelationOperator& correlation,
    const double* values,
    double* output,
    double* workspace,
    std::size_t workspace_size);

}  // namespace scar_internal
