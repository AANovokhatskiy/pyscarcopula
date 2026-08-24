#include "module.hpp"

#include "scar/copula/capability.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_capability(py::module_& m) {
    py::enum_<scar::NativeModelId>(
        m, "NativeModelId", "Exact built-in native model identifier.")
        .value("Independent", scar::NativeModelId::Independent)
        .value("Clayton", scar::NativeModelId::Clayton)
        .value("Frank", scar::NativeModelId::Frank)
        .value("Gumbel", scar::NativeModelId::Gumbel)
        .value("Joe", scar::NativeModelId::Joe)
        .value(
            "BivariateGaussian",
            scar::NativeModelId::BivariateGaussian)
        .value("Gaussian", scar::NativeModelId::Gaussian)
        .value("Student", scar::NativeModelId::Student)
        .value(
            "EquicorrGaussian",
            scar::NativeModelId::EquicorrGaussian)
        .value(
            "StochasticStudent",
            scar::NativeModelId::StochasticStudent)
        .value("Vine", scar::NativeModelId::Vine);

    py::enum_<scar::FactorEstimationKind>(
        m, "FactorEstimationKind", "Native factor-estimation policy.")
        .value("TwoStage", scar::FactorEstimationKind::TwoStage)
        .value("Joint", scar::FactorEstimationKind::Joint);

    py::enum_<scar::NativeOperation>(
        m, "NativeOperation", "Numerical operation capability identifier.")
        .value(
            "ParameterTransformBoundsInitialization",
            scar::NativeOperation::ParameterTransformBoundsInitialization)
        .value(
            "PointDensityDerivatives",
            scar::NativeOperation::PointDensityDerivatives)
        .value(
            "RowGridDensityGradient",
            scar::NativeOperation::RowGridDensityGradient)
        .value(
            "LikelihoodObjectiveGradient",
            scar::NativeOperation::LikelihoodObjectiveGradient)
        .value(
            "StateFilterSmoother",
            scar::NativeOperation::StateFilterSmoother)
        .value(
            "RosenblattResidual",
            scar::NativeOperation::RosenblattResidual)
        .value(
            "RadialGofSummary",
            scar::NativeOperation::RadialGofSummary)
        .value(
            "UnconditionalSamplingTransform",
            scar::NativeOperation::UnconditionalSamplingTransform)
        .value(
            "ConditionalSamplingTransform",
            scar::NativeOperation::ConditionalSamplingTransform)
        .value(
            "ArbitraryConditionalMcmc",
            scar::NativeOperation::ArbitraryConditionalMcmc)
        .value(
            "EdgeStructureSelectionScore",
            scar::NativeOperation::EdgeStructureSelectionScore);

    py::enum_<scar::DynamicsKind>(
        m, "DynamicsKind", "Native model dynamics identifier.")
        .value("Mle", scar::DynamicsKind::Mle)
        .value("Gas", scar::DynamicsKind::Gas)
        .value("ScarTmOu", scar::DynamicsKind::ScarTmOu)
        .value("ScarTmJacobi", scar::DynamicsKind::ScarTmJacobi)
        .value("ScarPOu", scar::DynamicsKind::ScarPOu)
        .value("ScarMOu", scar::DynamicsKind::ScarMOu);

    py::class_<scar::TypedModelDescriptor>(
        m,
        "TypedModelDescriptor",
        "Opaque typed descriptor used by native capability queries.")
        .def_property_readonly(
            "model_id", &scar::TypedModelDescriptor::model_id)
        .def_property_readonly(
            "dimension", &scar::TypedModelDescriptor::expected_dimension)
        .def_property_readonly(
            "correlation_kind",
            &scar::TypedModelDescriptor::correlation_kind)
        .def_property_readonly(
            "factor_estimation",
            &scar::TypedModelDescriptor::factor_estimation)
        .def_property_readonly(
            "rotation", &scar::TypedModelDescriptor::rotation)
        .def_property_readonly(
            "alternative_index",
            [](const scar::TypedModelDescriptor& descriptor) {
                return descriptor.alternative().index();
            });

    py::class_<scar::CapabilityInfo>(
        m,
        "CapabilityInfo",
        "Native support decision with constraints and diagnostic reason.")
        .def_readonly("supported", &scar::CapabilityInfo::supported)
        .def_readonly("constraints", &scar::CapabilityInfo::constraints)
        .def_readonly("reason", &scar::CapabilityInfo::reason);

    m.def(
        "make_typed_model_descriptor",
        &scar::make_typed_model_descriptor,
        py::arg("model_id"),
        py::arg("dimension"),
        py::arg("correlation_kind"),
        py::arg("rotation") = 0,
        py::arg("factor_estimation") =
            scar::FactorEstimationKind::TwoStage);
    m.def(
        "query_capability",
        py::overload_cast<
            const scar::TypedModelDescriptor&,
            scar::NativeOperation,
            scar::DynamicsKind>(&scar::query_capability),
        py::arg("descriptor"),
        py::arg("operation"),
        py::arg("dynamics"));
}

}  // namespace pyscarcopula::bindings
