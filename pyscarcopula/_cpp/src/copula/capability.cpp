#include "scar/copula/capability.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace scar {
namespace {

bool is_pair_model(NativeModelId model_id) noexcept {
    return model_id >= NativeModelId::Independent
        && model_id <= NativeModelId::BivariateGaussian;
}

bool is_static_multivariate(NativeModelId model_id) noexcept {
    return model_id == NativeModelId::Gaussian
        || model_id == NativeModelId::Student;
}

bool is_common_static_operation(NativeOperation operation) noexcept {
    switch (operation) {
        case NativeOperation::ParameterTransformBoundsInitialization:
        case NativeOperation::RowGridDensityGradient:
        case NativeOperation::LikelihoodObjectiveGradient:
        case NativeOperation::RosenblattResidual:
        case NativeOperation::RadialGofSummary:
        case NativeOperation::UnconditionalSamplingTransform:
        case NativeOperation::ConditionalSamplingTransform:
            return true;
        default:
            return false;
    }
}

bool valid_pair_rotation(
    NativeModelId model_id,
    int rotation) noexcept {

    if (model_id == NativeModelId::Clayton
        || model_id == NativeModelId::Gumbel
        || model_id == NativeModelId::Joe) {
        return rotation == 0 || rotation == 90
            || rotation == 180 || rotation == 270;
    }
    return rotation == 0;
}

bool correlation_matches(const TypedModelDescriptor& descriptor) noexcept {
    const CorrelationKind correlation = descriptor.correlation_kind();
    switch (descriptor.model_id()) {
        case NativeModelId::Independent:
        case NativeModelId::Clayton:
        case NativeModelId::Frank:
        case NativeModelId::Gumbel:
        case NativeModelId::Joe:
        case NativeModelId::BivariateGaussian:
            return correlation == CorrelationKind::NotApplicable;
        case NativeModelId::Gaussian:
        case NativeModelId::Student:
            return correlation == CorrelationKind::DenseCholesky
                || correlation == CorrelationKind::Fixed
                || correlation == CorrelationKind::Shrinkage
                || correlation == CorrelationKind::Factor;
        case NativeModelId::EquicorrGaussian:
            return correlation == CorrelationKind::Equicorrelation;
        case NativeModelId::StochasticStudent:
            return correlation == CorrelationKind::DenseCholesky
                || correlation == CorrelationKind::Fixed
                || correlation == CorrelationKind::Shrinkage
                || correlation == CorrelationKind::Factor
                || correlation
                    == CorrelationKind::FactorJointDynamicEstimation;
        case NativeModelId::Vine:
            return correlation == CorrelationKind::MixedPairEdges;
    }
    return false;
}

CapabilityInfo unsupported(std::string reason) {
    return CapabilityInfo{false, "native execution required; no fallback", std::move(reason)};
}

CapabilityInfo supported() {
    return CapabilityInfo{
        true,
        "exact built-in descriptor; native execution required",
        {}};
}

const char* dynamics_name(DynamicsKind dynamics) noexcept {
    switch (dynamics) {
        case DynamicsKind::Mle:
            return "MLE";
        case DynamicsKind::Gas:
            return "GAS";
        case DynamicsKind::ScarTmOu:
            return "SCAR-TM-OU";
        case DynamicsKind::ScarTmJacobi:
            return "SCAR-TM-JACOBI";
        case DynamicsKind::ScarPOu:
            return "SCAR-P-OU";
        case DynamicsKind::ScarMOu:
            return "SCAR-M-OU";
    }
    return "unknown dynamics";
}

}  // namespace

TypedModelDescriptor make_typed_model_descriptor(
    NativeModelId model_id,
    int dimension,
    CorrelationKind correlation_kind,
    int rotation,
    FactorEstimationKind factor_estimation) {

    if (is_pair_model(model_id)) {
        if (dimension != 2) {
            throw std::invalid_argument(
                "pair native descriptors require dimension=2");
        }
        if (correlation_kind != CorrelationKind::NotApplicable) {
            throw std::invalid_argument(
                "pair native descriptors require no correlation mode");
        }
        return TypedModelDescriptor(
            PairCopulaDescriptor{}, model_id, rotation);
    }
    if (dimension < 2) {
        throw std::invalid_argument(
            "multivariate native descriptors require dimension >= 2");
    }
    if (model_id == NativeModelId::Gaussian) {
        if (correlation_kind == CorrelationKind::Factor) {
            return TypedModelDescriptor(
                FactorGaussianDescriptor{{dimension}}, factor_estimation);
        }
        if (correlation_kind == CorrelationKind::DenseCholesky
            || correlation_kind == CorrelationKind::Fixed
            || correlation_kind == CorrelationKind::Shrinkage) {
            return TypedModelDescriptor(
                DenseGaussianDescriptor{{dimension}}, correlation_kind);
        }
    } else if (model_id == NativeModelId::Student
               || model_id == NativeModelId::StochasticStudent) {
        if (correlation_kind == CorrelationKind::Factor
            || correlation_kind
                == CorrelationKind::FactorJointDynamicEstimation) {
            return TypedModelDescriptor(
                FactorStudentDescriptor{{dimension}},
                model_id,
                correlation_kind
                        == CorrelationKind::FactorJointDynamicEstimation
                    ? FactorEstimationKind::Joint
                    : factor_estimation);
        }
        if (correlation_kind == CorrelationKind::DenseCholesky
            || correlation_kind == CorrelationKind::Fixed
            || correlation_kind == CorrelationKind::Shrinkage) {
            return TypedModelDescriptor(
                DenseStudentDescriptor{{dimension}},
                model_id,
                correlation_kind);
        }
    } else if (model_id == NativeModelId::EquicorrGaussian
               && correlation_kind == CorrelationKind::Equicorrelation) {
        return TypedModelDescriptor(
            EquicorrGaussianDescriptor{{dimension}});
    } else if (model_id == NativeModelId::Vine
               && correlation_kind == CorrelationKind::MixedPairEdges) {
        return TypedModelDescriptor(VineDescriptor{{dimension}});
    }
    throw std::invalid_argument(
        "native model and correlation alternatives are incompatible");
}

CapabilityInfo query_capability(const CapabilityRequest& request) {
    const TypedModelDescriptor& descriptor = request.descriptor;
    const NativeModelId model_id = descriptor.model_id();
    const bool dynamic = request.dynamics != DynamicsKind::Mle;

    if ((is_pair_model(model_id) && descriptor.expected_dimension() != 2)
        || (!is_pair_model(model_id)
            && descriptor.expected_dimension() < 2)) {
        return unsupported("descriptor dimension is invalid");
    }
    if (!correlation_matches(descriptor)) {
        return unsupported(
            "model and correlation alternatives are incompatible");
    }
    if (is_pair_model(model_id)) {
        if (!valid_pair_rotation(model_id, descriptor.rotation())) {
            return unsupported("rotation is not supported by this pair model");
        }
        if (model_id == NativeModelId::Independent && dynamic) {
            return unsupported(
                "IndependentCopula has no dynamic state capability");
        }
        if (request.operation
            == NativeOperation::ArbitraryConditionalMcmc) {
            return unsupported(
                "arbitrary conditional MCMC is a vine-only capability");
        }
        if (request.operation == NativeOperation::StateFilterSmoother
            && !dynamic) {
            return unsupported(
                "state filtering is not applicable to static MLE");
        }
        return supported();
    }

    if (is_static_multivariate(model_id)) {
        if (dynamic) {
            return unsupported(
                "static multivariate model does not support "
                + std::string(dynamics_name(request.dynamics)));
        }
        return is_common_static_operation(request.operation)
            ? supported()
            : unsupported(
                "operation is not supported by static multivariate models");
    }

    if (model_id == NativeModelId::EquicorrGaussian) {
        const bool supported_dynamics =
            request.dynamics == DynamicsKind::Mle
            || request.dynamics == DynamicsKind::Gas
            || request.dynamics == DynamicsKind::ScarTmOu;
        if (!supported_dynamics) {
            return unsupported(
                "EquicorrGaussianCopula does not support "
                + std::string(dynamics_name(request.dynamics)));
        }
        if (request.operation == NativeOperation::StateFilterSmoother) {
            return dynamic ? supported() : unsupported(
                "state filtering is not applicable to static MLE");
        }
        return is_common_static_operation(request.operation)
            ? supported()
            : unsupported(
                "operation is not supported by equicorrelation models");
    }

    if (model_id == NativeModelId::StochasticStudent) {
        if (descriptor.correlation_kind()
                == CorrelationKind::FactorJointDynamicEstimation
            && dynamic) {
            return unsupported(
                "dynamic joint factor estimation is not implemented");
        }
        const bool supported_dynamics =
            request.dynamics == DynamicsKind::Mle
            || request.dynamics == DynamicsKind::Gas
            || request.dynamics == DynamicsKind::ScarTmOu
            || request.dynamics == DynamicsKind::ScarPOu
            || request.dynamics == DynamicsKind::ScarMOu;
        if (!supported_dynamics) {
            return unsupported(
                "StochasticStudentCopula does not support "
                + std::string(dynamics_name(request.dynamics)));
        }
        if (request.operation == NativeOperation::StateFilterSmoother) {
            return dynamic ? supported() : unsupported(
                "state filtering is not applicable to static MLE");
        }
        return is_common_static_operation(request.operation)
            ? supported()
            : unsupported(
                "operation is not supported by stochastic Student models");
    }

    if (model_id == NativeModelId::Vine) {
        if (request.operation == NativeOperation::PointDensityDerivatives) {
            return unsupported(
                "point density derivatives belong to pair edges");
        }
        if (request.operation == NativeOperation::StateFilterSmoother) {
            return dynamic ? supported() : unsupported(
                "state filtering is not applicable to static vine edges");
        }
        return supported();
    }

    return unsupported("native model identifier is not registered");
}

}  // namespace scar
