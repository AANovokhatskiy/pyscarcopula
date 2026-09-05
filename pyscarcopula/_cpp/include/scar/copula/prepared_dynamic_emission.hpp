#pragma once

#include "scar/copula/spec.hpp"
#include "scar/core/result.hpp"
#include "scar/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace scar {

enum class DynamicEmissionKind : int {
    Unsupported = 0,
    Pair = 1,
    Student = 2,
    Equicorrelation = 3,
};

struct DynamicEmissionRowResult {
    double parameter = 0.0;
    double log_pdf = 0.0;
    double dlog_dparameter = 0.0;
    Status status = Status::Ok;
    FailureContext failure{};

    bool is_ok() const noexcept {
        return ok(status);
    }
};

/// Build a dense Student compatibility spec from a base correlation and a
/// raw shrinkage coordinate without exposing concrete correlation kernels to
/// application modules.
Result<CopulaSpec> prepare_shrinkage_dynamic_spec(
    const CopulaSpec& template_spec,
    DoubleView base_correlation,
    double raw_shrinkage);

/// Refresh only the correlation of a call-owned Student spec. Preserve its
/// dense PPF cache; leave the spec unchanged if correlation preparation fails.
Result<bool> update_shrinkage_dynamic_spec(
    CopulaSpec& spec,
    DoubleView base_correlation,
    double raw_shrinkage);

/// Per-call scratch storage for a prepared dynamic emission.
class PreparedDynamicEmissionWorkspace {
public:
    PreparedDynamicEmissionWorkspace();
    ~PreparedDynamicEmissionWorkspace();
    PreparedDynamicEmissionWorkspace(
        PreparedDynamicEmissionWorkspace&&) noexcept;
    PreparedDynamicEmissionWorkspace& operator=(
        PreparedDynamicEmissionWorkspace&&) noexcept;

    PreparedDynamicEmissionWorkspace(
        const PreparedDynamicEmissionWorkspace&) = delete;
    PreparedDynamicEmissionWorkspace& operator=(
        const PreparedDynamicEmissionWorkspace&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend class PreparedDynamicEmission;
};

/// Model-neutral scalar dynamic-emission interface used by GAS and SCAR-OU.
///
/// The compatibility CopulaSpec is copied and resolved once by the owning
/// constructor. Call-scoped adapters may explicitly borrow a spec to avoid
/// copying large immutable caches; the borrowed spec must outlive the returned
/// emission. Application modules call this interface without including
/// concrete pair, Student, or equicorrelation headers.
class PreparedDynamicEmission {
public:
    explicit PreparedDynamicEmission(const CopulaSpec& spec);
    static PreparedDynamicEmission borrow(const CopulaSpec& spec);
    ~PreparedDynamicEmission();
    PreparedDynamicEmission(PreparedDynamicEmission&&) noexcept;
    PreparedDynamicEmission& operator=(PreparedDynamicEmission&&) noexcept;

    PreparedDynamicEmission(const PreparedDynamicEmission&) = delete;
    PreparedDynamicEmission& operator=(const PreparedDynamicEmission&) = delete;

    void refresh(const CopulaSpec& spec);

    DynamicEmissionKind kind() const noexcept;
    CopulaFamily family() const noexcept;
    CorrelationKind correlation_kind() const noexcept;
    int expected_dimension() const noexcept;
    bool is_supported() const noexcept;
    bool is_supported_for_ou() const noexcept;
    bool is_independent() const noexcept;
    bool is_unrotated_gaussian_pair() const noexcept;
    bool has_cached_observations(std::size_t rows) const noexcept;
    bool observation_cache_compatible(std::size_t rows) const noexcept;
    double h_from_cached_observation(
        std::size_t row,
        bool reverse,
        double parameter) const;

    const CopulaSpec& compatibility_spec() const noexcept;

    Status validate_observations(
        ObservationView observations,
        bool require_nonempty = true) const;

    PreparedDynamicEmissionWorkspace make_workspace(
        bool derivative) const;

    double transform_state(double state) const;
    double dtransform_state(double state) const;
    DynamicEmissionRowResult evaluate_parameter(
        const double* row,
        std::int64_t row_index,
        double parameter,
        bool derivative,
        PreparedDynamicEmissionWorkspace& workspace) const;
    DynamicEmissionRowResult evaluate_state(
        const double* row,
        std::int64_t row_index,
        double state,
        bool derivative,
        PreparedDynamicEmissionWorkspace& workspace) const;
    double log_pdf_at_state(
        const double* row,
        std::int64_t row_index,
        double state,
        PreparedDynamicEmissionWorkspace& workspace) const;

    void prepare_grid_transform(
        const std::vector<double>& states,
        std::vector<double>& parameters,
        std::vector<double>& derivatives) const;
    void fill_density_row(
        const double* observations,
        std::int64_t row_index,
        const std::vector<double>& parameters,
        double* densities,
        double* log_scale = nullptr) const;
    void fill_density_and_gradient_row(
        const double* observations,
        std::int64_t row_index,
        const std::vector<double>& parameters,
        const std::vector<double>& derivatives,
        double* densities,
        double* gradients,
        double* log_scale = nullptr) const;
    void fill_density_and_gradient_grid(
        const double* observations,
        std::int64_t rows,
        const std::vector<double>& parameters,
        const std::vector<double>& derivatives,
        std::vector<double>& densities,
        std::vector<double>& gradients,
        int n_threads,
        double* log_scale_sum = nullptr) const;
    /// Fill a bounded row range while retaining global prepared-cache indices.
    bool fill_density_and_gradient_block(
        const double* observations,
        std::int64_t first_row,
        std::int64_t rows,
        const std::vector<double>& parameters,
        const std::vector<double>& derivatives,
        std::vector<double>& densities,
        std::vector<double>& gradients,
        std::vector<double>& row_log_scales,
        int n_threads) const;
    void fill_density_row_on_state_grid(
        const double* observations,
        std::int64_t row_index,
        const std::vector<double>& states,
        std::vector<double>& densities) const;

    double h(
        double first,
        double second,
        double parameter) const;
    void h_pair(
        double first,
        double second,
        double parameter,
        double& first_next,
        double& second_next) const;
    double inverse_h(
        double quantile,
        double given,
        double parameter) const;

private:
    struct BorrowedSpecTag {};
    PreparedDynamicEmission(const CopulaSpec& spec, BorrowedSpecTag);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace scar
