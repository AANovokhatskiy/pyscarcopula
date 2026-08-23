#pragma once

#include <variant>

namespace scar {
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

/// Type-safe model alternative produced by the temporary CopulaSpec adapter.
class TypedModelDescriptor {
public:
    using Alternative = std::variant<
        PairCopulaDescriptor,
        DenseGaussianDescriptor,
        FactorGaussianDescriptor,
        EquicorrGaussianDescriptor,
        DenseStudentDescriptor,
        FactorStudentDescriptor>;

    TypedModelDescriptor(PairCopulaDescriptor descriptor) noexcept
        : descriptor_(descriptor) {}

    TypedModelDescriptor(DenseGaussianDescriptor descriptor) noexcept
        : descriptor_(descriptor) {}

    TypedModelDescriptor(FactorGaussianDescriptor descriptor) noexcept
        : descriptor_(descriptor) {}

    TypedModelDescriptor(EquicorrGaussianDescriptor descriptor) noexcept
        : descriptor_(descriptor) {}

    TypedModelDescriptor(DenseStudentDescriptor descriptor) noexcept
        : descriptor_(descriptor) {}

    TypedModelDescriptor(FactorStudentDescriptor descriptor) noexcept
        : descriptor_(descriptor) {}

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

private:
    Alternative descriptor_;
};

}  // namespace scar
