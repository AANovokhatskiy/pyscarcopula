#include "scar/copula.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
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

    scar::CopulaSpec clayton_spec;
    clayton_spec.family = scar::CopulaFamily::Clayton;
    clayton_spec.rotation = scar::Rotation::R180;
    clayton_spec.transform = scar::Transform::Softplus;
    const scar::PreparedPairKernel clayton(clayton_spec);
    const scar::PreparedPairKernel student(student_spec);
    if (!clayton.is_supported()
        || student.is_supported()
        || std::abs(clayton.tau_to_parameter(0.5) - 2.0) > 1e-15
        || std::abs(clayton.parameter_to_tau(2.0) - 0.5) > 1e-15) {
        return 7;
    }

    const std::vector<double> value_grid{-1.0, 0.0, 1.0};
    std::vector<double> parameter_grid;
    std::vector<double> derivative_grid;
    std::vector<double> densities(value_grid.size(), 0.0);
    std::vector<double> gradients(value_grid.size(), 0.0);
    clayton.prepare_parameter_grid(
        value_grid, parameter_grid, derivative_grid);
    clayton.fill_grid_row_with_gradient(
        0.25,
        0.75,
        parameter_grid,
        derivative_grid,
        densities.data(),
        gradients.data());
    if (parameter_grid.size() != value_grid.size()
        || derivative_grid.size() != value_grid.size()) {
        return 8;
    }
    for (std::size_t index = 0; index < value_grid.size(); ++index) {
        double scalar_density = 0.0;
        double scalar_gradient = 0.0;
        clayton.pdf_and_gradient(
            0.25,
            0.75,
            value_grid[index],
            scalar_density,
            scalar_gradient);
        if (densities[index] != scalar_density
            || gradients[index] != scalar_gradient) {
            return 9;
        }
    }
    return scar::SCAR_OK;
}
