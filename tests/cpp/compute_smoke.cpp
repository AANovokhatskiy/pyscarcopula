#include "scar/copula.hpp"
#include "scar/core/checked_arithmetic.hpp"
#include "scar/core/result.hpp"
#include "scar/core/threading.hpp"
#include "scar/math/normal.hpp"
#include "scar/status.hpp"

#include <cmath>
#include <variant>
#include <vector>

int main() {
    const double span_values[] = {0.25, 0.75};
    const scar::DoubleView span{span_values, 2};
    const scar::DoubleMatrixView matrix{span_values, 1, 2};
    if (span.size() != 2 || span[1] != 0.75
        || matrix.size() != 1 || matrix.row(0)[0] != 0.25) {
        return 1;
    }

    std::size_t shape_size = 0;
    std::uint64_t byte_count = 0;
    if (!scar::core::checked_shape_size(3, 4, shape_size)
        || shape_size != 12
        || !scar::core::checked_byte_count<double>(
            shape_size, byte_count)
        || byte_count != 12 * sizeof(double)
        || scar::core::limit_worker_count(8, 3) != 3
        || scar::core::worker_count_for_items(8, 31, 4) != 1
        || scar::core::worker_count_for_items(8, 32, 4) != 8) {
        return 2;
    }

    const scar::Result<double> foundation_result = scar::success(1.0);
    if (!foundation_result.is_ok()
        || std::abs(scar::math::normal_cdf(0.0) - 0.5) > 1e-15
        || std::abs(scar::math::normal_quantile(0.5)) > 1e-15) {
        return 3;
    }

    scar::CopulaSpec spec;
    spec.family = scar::CopulaFamily::Independent;

    const scar::TypedModelDescriptor pair_descriptor =
        spec.model_descriptor();
    scar::CopulaSpec student_spec;
    student_spec.family = scar::CopulaFamily::Student;
    student_spec.correlation_kind = scar::CorrelationKind::Factor;
    student_spec.dim = 5;
    const scar::TypedModelDescriptor student_descriptor =
        student_spec.model_descriptor();
    if (pair_descriptor.expected_dimension() != 2
        || student_descriptor.expected_dimension() != 5
        || !std::holds_alternative<scar::FactorStudentDescriptor>(
            student_descriptor.alternative())) {
        return 4;
    }

    if (!scar::is_supported(spec)) {
        return 5;
    }

    const scar::Observations observations{{0.25, 0.75}};
    const std::vector<double> parameters{0.0};
    const auto density = scar::copula_pdf(spec, observations, parameters);
    if (density.size() != 1 || std::abs(density.front() - 1.0) > 1e-15) {
        return 6;
    }
    return scar::SCAR_OK;
}
