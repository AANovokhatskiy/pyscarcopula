#pragma once

#include <variant>

namespace scar {

/// Stable identifiers exposed by the native model registry.
enum class NativeModelId : int {
    Independent = 0,
    Clayton = 1,
    Frank = 2,
    Gumbel = 3,
    Joe = 4,
    BivariateGaussian = 5,
    Gaussian = 6,
    Student = 7,
    EquicorrGaussian = 8,
    StochasticStudent = 9,
    Vine = 10,
};

/// Correlation/configuration alternative used by capability queries.
///
/// DenseCholesky and Factor retain their original values because CopulaSpec
/// uses them as computational storage alternatives.  The remaining values
/// describe higher-level configuration policies without changing numerical
/// state.
enum class CorrelationKind : int {
    DenseCholesky = 0,
    Factor = 1,
    Fixed = 2,
    Shrinkage = 3,
    Equicorrelation = 4,
    FactorJointDynamicEstimation = 5,
    NotApplicable = 6,
    MixedPairEdges = 7,
};

enum class FactorEstimationKind : int {
    TwoStage = 0,
    Joint = 1,
};

namespace model_descriptor_detail {

struct PairDimension {
    constexpr int expected_dimension() const noexcept {
        return 2;
    }
};

struct RuntimeDimension {
    int dimension = 2;

    constexpr int expected_dimension() const noexcept {
        return dimension;
    }
};

}  // namespace model_descriptor_detail

/// Typed shape descriptor for bivariate copula families.
struct PairCopulaDescriptor : model_descriptor_detail::PairDimension {};

/// Typed shape descriptors for the current multivariate alternatives.
struct DenseGaussianDescriptor : model_descriptor_detail::RuntimeDimension {};
struct FactorGaussianDescriptor : model_descriptor_detail::RuntimeDimension {};
struct EquicorrGaussianDescriptor : model_descriptor_detail::RuntimeDimension {};
struct DenseStudentDescriptor : model_descriptor_detail::RuntimeDimension {};
struct FactorStudentDescriptor : model_descriptor_detail::RuntimeDimension {};
struct VineDescriptor : model_descriptor_detail::RuntimeDimension {};

/// Type-safe model alternative produced by the temporary CopulaSpec adapter.
class TypedModelDescriptor {
public:
    using Alternative = std::variant<
        PairCopulaDescriptor,
        DenseGaussianDescriptor,
        FactorGaussianDescriptor,
        EquicorrGaussianDescriptor,
        DenseStudentDescriptor,
        FactorStudentDescriptor,
        VineDescriptor>;

    TypedModelDescriptor(
        PairCopulaDescriptor descriptor,
        NativeModelId model_id = NativeModelId::Clayton,
        int rotation = 0) noexcept
        : descriptor_(descriptor),
          model_id_(model_id),
          correlation_kind_(CorrelationKind::NotApplicable),
          rotation_(rotation) {}

    TypedModelDescriptor(
        DenseGaussianDescriptor descriptor,
        CorrelationKind correlation_kind =
            CorrelationKind::DenseCholesky) noexcept
        : descriptor_(descriptor),
          model_id_(NativeModelId::Gaussian),
          correlation_kind_(correlation_kind) {}

    TypedModelDescriptor(
        FactorGaussianDescriptor descriptor,
        FactorEstimationKind factor_estimation =
            FactorEstimationKind::TwoStage) noexcept
        : descriptor_(descriptor),
          model_id_(NativeModelId::Gaussian),
          correlation_kind_(CorrelationKind::Factor),
          factor_estimation_(factor_estimation) {}

    TypedModelDescriptor(EquicorrGaussianDescriptor descriptor) noexcept
        : descriptor_(descriptor),
          model_id_(NativeModelId::EquicorrGaussian),
          correlation_kind_(CorrelationKind::Equicorrelation) {}

    TypedModelDescriptor(
        DenseStudentDescriptor descriptor,
        NativeModelId model_id = NativeModelId::Student,
        CorrelationKind correlation_kind =
            CorrelationKind::DenseCholesky) noexcept
        : descriptor_(descriptor),
          model_id_(model_id),
          correlation_kind_(correlation_kind) {}

    TypedModelDescriptor(
        FactorStudentDescriptor descriptor,
        NativeModelId model_id = NativeModelId::Student,
        FactorEstimationKind factor_estimation =
            FactorEstimationKind::TwoStage) noexcept
        : descriptor_(descriptor),
          model_id_(model_id),
          correlation_kind_(
              model_id == NativeModelId::StochasticStudent
                  && factor_estimation == FactorEstimationKind::Joint
                  ? CorrelationKind::FactorJointDynamicEstimation
                  : CorrelationKind::Factor),
          factor_estimation_(factor_estimation) {}

    TypedModelDescriptor(VineDescriptor descriptor) noexcept
        : descriptor_(descriptor),
          model_id_(NativeModelId::Vine),
          correlation_kind_(CorrelationKind::MixedPairEdges) {}

    int expected_dimension() const noexcept {
        return std::visit(
            [](const auto& descriptor) {
                return descriptor.expected_dimension();
            },
            descriptor_);
    }

    const Alternative& alternative() const noexcept {
        return descriptor_;
    }

    NativeModelId model_id() const noexcept {
        return model_id_;
    }

    CorrelationKind correlation_kind() const noexcept {
        return correlation_kind_;
    }

    FactorEstimationKind factor_estimation() const noexcept {
        return factor_estimation_;
    }

    int rotation() const noexcept {
        return rotation_;
    }

private:
    Alternative descriptor_;
    NativeModelId model_id_ = NativeModelId::Clayton;
    CorrelationKind correlation_kind_ = CorrelationKind::NotApplicable;
    FactorEstimationKind factor_estimation_ = FactorEstimationKind::TwoStage;
    int rotation_ = 0;
};

/// Validate metadata and construct the matching concrete descriptor.
TypedModelDescriptor make_typed_model_descriptor(
    NativeModelId model_id,
    int dimension,
    CorrelationKind correlation_kind,
    int rotation = 0,
    FactorEstimationKind factor_estimation = FactorEstimationKind::TwoStage);

}  // namespace scar
