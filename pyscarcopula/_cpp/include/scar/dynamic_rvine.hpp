#pragma once

#include "scar/gas.hpp"
#include "scar/jacobi.hpp"
#include "scar/ou.hpp"
#include "scar/rvine.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace scar {

/// Dynamics attached to one edge in the R-vine Rosenblatt composition.
enum class DynamicRvineKind : int {
    Static = 0,
    Gas = 1,
    ScarOu = 2,
    ScarJacobi = 3,
};

/// Typed edge descriptor for the application-level dynamic-vine composition.
///
/// The generic runtime in `scar/rvine.hpp` remains independent of GAS and
/// latent-state models.  This descriptor belongs to the composition layer and
/// lets it invoke the existing prepared dynamic evaluators while reusing the
/// common topology and pair kernels.
struct DynamicRvineEdge {
    rvine::EdgeSpec edge;
    DynamicRvineKind dynamics = DynamicRvineKind::Static;
    GasParams gas_params{};
    GasConfig gas_config{};
    OuParams ou_params{};
    OuNumericalConfig ou_config{};
    std::string ou_method = "auto";
    JacobiParams jacobi_params{};
    JacobiEvaluatorConfig jacobi_config{};
};

/// Execute static and dynamic edge transforms through one C++ traversal.
///
/// Static edges read scalar/row parameters from `parameters`.  Dynamic edges
/// run their native filter/evaluator on the canonical pair path produced by
/// the traversal.  No caller callback participates in the numerical loop.
rvine::RosenblattResult dynamic_rvine_rosenblatt_transform(
    const RVineDensityPlan& plan,
    const std::vector<DynamicRvineEdge>& edges,
    const rvine::ParameterPack& parameters,
    DoubleView observations,
    std::int64_t observation_rows,
    std::int64_t observation_columns,
    int n_threads = 1,
    bool capture_node_values = false);

}  // namespace scar
