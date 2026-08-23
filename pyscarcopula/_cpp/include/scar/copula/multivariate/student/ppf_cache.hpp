#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {
struct CopulaSpec;
}

namespace scar::copula::multivariate::student {

struct PpfCache {
    std::int64_t observation_count = 0;
    std::vector<double> nodes;
    std::vector<double> table;
};

PpfCache& ppf_cache(CopulaSpec& spec);
const PpfCache& ppf_cache(const CopulaSpec& spec);

}  // namespace scar::copula::multivariate::student

namespace scar_internal {

struct PpfInterpolation {
    std::array<std::size_t, 4> node{};
    std::array<double, 4> value_weight{};
    std::array<double, 4> derivative_weight{};
    int count = 0;
};

bool student_ppf_cache_available(
    const scar::copula::multivariate::student::PpfCache& cache,
    int dimension,
    std::int64_t row_index);
PpfInterpolation make_ppf_interpolation(
    const std::vector<double>& nodes,
    double df);
double interpolate_ppf_value(
    const scar::copula::multivariate::student::PpfCache& cache,
    int dimension,
    const PpfInterpolation& interpolation,
    std::int64_t row_index,
    int column,
    double* derivative);
void interpolate_ppf_row(
    const scar::copula::multivariate::student::PpfCache& cache,
    int dimension,
    const PpfInterpolation& interpolation,
    std::int64_t row_index,
    double* values,
    double* derivatives);
void interpolate_bivariate_ppf(
    const scar::copula::multivariate::student::PpfCache& cache,
    const PpfInterpolation& interpolation,
    std::size_t observation,
    double& first,
    double& second,
    double& first_derivative,
    double& second_derivative);

}  // namespace scar_internal
