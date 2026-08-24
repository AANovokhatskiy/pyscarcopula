#pragma once

#include "scar/copula/model_descriptor.hpp"
#include "scar/copula/rotation.hpp"
#include "scar/copula/transforms.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace scar {

class FactorCorrelationOperator;

/// Copula kernels supported by the native dispatch layer.
enum class CopulaFamily : int {
#define SCAR_PAIR_FAMILY(                                                \
    enum_name, package_name, enum_value, transform_policy, rotation_policy, \
    default_transform, default_offset)                                  \
    enum_name = enum_value,
#include "scar/copula/pair/families.def"
#undef SCAR_PAIR_FAMILY
    Student = 6,
    EquicorrGaussian = 7,
    MultivariateGaussian = 8,
};

enum class CorrelationKind : int {
    DenseCholesky = 0,
    Factor = 1,
};

struct TypedModelStorage;

/// Compatibility descriptor accepted by the generic runtime boundary.
///
/// Family-specific numerical state is owned by a typed storage alternative,
/// rather than exposed as an ever-growing collection of unrelated fields.
/// The compatibility adapter remains available while callers migrate to
/// prepared typed models.
struct CopulaSpec {
    CopulaSpec();
    ~CopulaSpec();
    CopulaSpec(const CopulaSpec& other);
    CopulaSpec(CopulaSpec&& other) noexcept;
    CopulaSpec& operator=(const CopulaSpec& other);
    CopulaSpec& operator=(CopulaSpec&& other) noexcept;

    CopulaFamily family = CopulaFamily::Clayton;
    Rotation rotation = Rotation::R0;
    Transform transform = Transform::Softplus;
    double offset = 0.0001;
    int dim = 2;
    CorrelationKind correlation_kind = CorrelationKind::DenseCholesky;

    TypedModelDescriptor model_descriptor() const noexcept;

    // Transitional accessors used by the Stage 4 CopulaSpec adapter. New
    // model code consumes the typed package contracts directly.
    std::vector<double>& dense_inverse_cholesky();
    const std::vector<double>& dense_inverse_cholesky() const;
    double& dense_log_determinant();
    double dense_log_determinant() const;
    std::shared_ptr<const FactorCorrelationOperator>& factor_operator();
    const std::shared_ptr<const FactorCorrelationOperator>&
        factor_operator() const;
    std::size_t& factor_dimension_tile();
    std::size_t factor_dimension_tile() const;
    std::int64_t& student_ppf_observation_count();
    std::int64_t student_ppf_observation_count() const;
    std::vector<double>& student_ppf_nodes();
    const std::vector<double>& student_ppf_nodes() const;
    std::vector<double>& student_ppf_table();
    const std::vector<double>& student_ppf_table() const;
    std::vector<double>& pair_gaussian_first_scores();
    const std::vector<double>& pair_gaussian_first_scores() const;
    std::vector<double>& pair_gaussian_second_scores();
    const std::vector<double>& pair_gaussian_second_scores() const;
    std::vector<double>& equicorr_sum_scores();
    const std::vector<double>& equicorr_sum_scores() const;
    std::vector<double>& equicorr_sum_squares();
    const std::vector<double>& equicorr_sum_squares() const;

    /// Rebuild the typed alternative after compatibility metadata changes.
    void reset_model_storage();
    TypedModelStorage& model_storage();
    const TypedModelStorage& model_storage() const;

private:
    void synchronize_model_storage() const;
    mutable std::unique_ptr<TypedModelStorage> model_storage_;
};

}  // namespace scar
