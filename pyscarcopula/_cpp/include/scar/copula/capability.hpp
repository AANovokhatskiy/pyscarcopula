#pragma once

#include "scar/copula/model_descriptor.hpp"

#include <string>

namespace scar {

enum class NativeOperation : int {
    ParameterTransformBoundsInitialization = 0,
    PointDensityDerivatives = 1,
    RowGridDensityGradient = 2,
    LikelihoodObjectiveGradient = 3,
    StateFilterSmoother = 4,
    RosenblattResidual = 5,
    RadialGofSummary = 6,
    UnconditionalSamplingTransform = 7,
    ConditionalSamplingTransform = 8,
    ArbitraryConditionalMcmc = 9,
    EdgeStructureSelectionScore = 10,
};

enum class DynamicsKind : int {
    Mle = 0,
    Gas = 1,
    ScarTmOu = 2,
    ScarTmJacobi = 3,
    ScarPOu = 4,
    ScarMOu = 5,
};

struct CapabilityRequest {
    TypedModelDescriptor descriptor;
    NativeOperation operation;
    DynamicsKind dynamics;
};

struct CapabilityInfo {
    bool supported = false;
    std::string constraints;
    std::string reason;
};

/// Query the authoritative native capability matrix for one concrete model.
CapabilityInfo query_capability(const CapabilityRequest& request);

inline CapabilityInfo query_capability(
    const TypedModelDescriptor& descriptor,
    NativeOperation operation,
    DynamicsKind dynamics) {

    return query_capability(CapabilityRequest{
        descriptor, operation, dynamics});
}

}  // namespace scar
