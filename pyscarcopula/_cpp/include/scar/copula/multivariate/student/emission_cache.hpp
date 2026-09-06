#pragma once

#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/copula/multivariate/student/density.hpp"

#include <vector>
#include <atomic>

namespace scar_internal {

/// Opt-in, fixed-observation/fixed-correlation log-density reconstruction.
/// Endpoint values and slopes use the direct Student kernel, never PPF tables.
/// Residuals at three interior points are empirical diagnostics, not bounds.
class StudentEmissionCache {
public:
    StudentEmissionCache(const PreparedStudentDensity& model,
                         scar::ObservationView observations, double offset,
                         const scar::StudentEmissionCacheConfig& config);
    bool evaluate(std::int64_t row, double df, double& value,
                  double& derivative) const;
    scar::StudentEmissionCacheDiagnostics diagnostics() const noexcept {
        auto out = diagnostics_;
        out.interpolation_hits = hits_.load(std::memory_order_relaxed);
        out.exact_endpoint_hits = endpoint_hits_.load(std::memory_order_relaxed);
        out.exact_fallbacks = fallbacks_.load(std::memory_order_relaxed);
        return out;
    }

private:
    std::size_t rows_ = 0;
    double offset_ = 0.0;
    std::vector<double> knots_, values_, slopes_;
    std::vector<double> endpoint_values_, endpoint_slopes_;
    scar::StudentEmissionCacheDiagnostics diagnostics_;
    mutable std::atomic<std::uint64_t> hits_{0}, fallbacks_{0}, endpoint_hits_{0};
};

}  // namespace scar_internal
