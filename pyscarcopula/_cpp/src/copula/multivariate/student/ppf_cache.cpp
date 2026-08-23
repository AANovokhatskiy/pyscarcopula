#include "scar/copula/multivariate/student/ppf_cache.hpp"

#include "scar/copula/model_storage.hpp"
#include "scar/copula/spec.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <stdexcept>

namespace scar::copula::multivariate::student {

PpfCache& ppf_cache(CopulaSpec& spec) {
    TypedModelStorage& storage = spec.model_storage();
    if (auto* dense = std::get_if<DenseModelStorage>(&storage.value)) {
        return dense->ppf;
    }
    if (auto* factor = std::get_if<FactorModelStorage>(&storage.value)) {
        return factor->ppf;
    }
    throw std::logic_error("copula does not own a Student PPF cache");
}

const PpfCache& ppf_cache(const CopulaSpec& spec) {
    const TypedModelStorage& storage = spec.model_storage();
    if (const auto* dense = std::get_if<DenseModelStorage>(&storage.value)) {
        return dense->ppf;
    }
    if (const auto* factor = std::get_if<FactorModelStorage>(&storage.value)) {
        return factor->ppf;
    }
    throw std::logic_error("copula does not own a Student PPF cache");
}

}  // namespace scar::copula::multivariate::student

namespace scar_internal {
namespace {

void add_interpolation_weight(
    PpfInterpolation& interpolation,
    std::size_t node,
    double value_weight,
    double derivative_weight) {

    for (int i = 0; i < interpolation.count; ++i) {
        if (interpolation.node[static_cast<std::size_t>(i)] == node) {
            interpolation.value_weight[static_cast<std::size_t>(i)]
                += value_weight;
            interpolation.derivative_weight[static_cast<std::size_t>(i)]
                += derivative_weight;
            return;
        }
    }
    const std::size_t index =
        static_cast<std::size_t>(interpolation.count);
    interpolation.node[index] = node;
    interpolation.value_weight[index] = value_weight;
    interpolation.derivative_weight[index] = derivative_weight;
    ++interpolation.count;
}

}  // namespace

bool student_ppf_cache_available(
    const scar::copula::multivariate::student::PpfCache& cache,
    int dimension,
    std::int64_t row_index) {

    const std::size_t n_nodes = cache.nodes.size();
    std::size_t n_obs = 0;
    std::size_t dim = 0;
    std::size_t rows = 0;
    std::size_t expected = 0;
    if (dimension <= 0) {
        return false;
    }
    dim = static_cast<std::size_t>(dimension);
    if (!checked_nonnegative_size(cache.observation_count, n_obs)
        || !checked_size_mul(n_nodes, n_obs, rows)
        || !checked_size_mul(rows, dim, expected)) {
        return false;
    }
    return row_index >= 0
        && row_index < cache.observation_count
        && n_nodes >= 2
        && cache.table.size() == expected;
}

PpfInterpolation make_ppf_interpolation(
    const std::vector<double>& nodes,
    double df) {

    const auto upper = std::upper_bound(nodes.begin(), nodes.end(), df);
    std::size_t idx = upper == nodes.begin()
        ? 0
        : static_cast<std::size_t>(upper - nodes.begin() - 1);
    idx = std::min(idx, nodes.size() - 2);

    const double lo = nodes[idx];
    const double hi = nodes[idx + 1];
    const double interval = hi - lo;
    const double alpha = (df - lo) / interval;
    PpfInterpolation interpolation;
    if (df <= nodes.front() || df >= nodes.back()) {
        add_interpolation_weight(
            interpolation, idx, 1.0 - alpha, -1.0 / interval);
        add_interpolation_weight(
            interpolation, idx + 1, alpha, 1.0 / interval);
        return interpolation;
    }

    const double alpha2 = alpha * alpha;
    const double alpha3 = alpha2 * alpha;
    const double h00 = 2.0 * alpha3 - 3.0 * alpha2 + 1.0;
    const double h10 = alpha3 - 2.0 * alpha2 + alpha;
    const double h01 = -2.0 * alpha3 + 3.0 * alpha2;
    const double h11 = alpha3 - alpha2;
    const double dh00 = (6.0 * alpha2 - 6.0 * alpha) / interval;
    const double dh10 = 3.0 * alpha2 - 4.0 * alpha + 1.0;
    const double dh01 = (-6.0 * alpha2 + 6.0 * alpha) / interval;
    const double dh11 = 3.0 * alpha2 - 2.0 * alpha;

    const std::size_t lo_slope_node = idx == 0 ? idx : idx - 1;
    const double lo_slope_interval =
        nodes[idx + 1] - nodes[lo_slope_node];
    const std::size_t hi_slope_node =
        idx + 1 == nodes.size() - 1 ? idx + 1 : idx + 2;
    const double hi_slope_interval =
        nodes[hi_slope_node] - nodes[idx];

    add_interpolation_weight(interpolation, idx, h00, dh00);
    add_interpolation_weight(interpolation, idx + 1, h01, dh01);
    add_interpolation_weight(
        interpolation,
        lo_slope_node,
        -h10 * interval / lo_slope_interval,
        -dh10 / lo_slope_interval);
    add_interpolation_weight(
        interpolation,
        idx + 1,
        h10 * interval / lo_slope_interval,
        dh10 / lo_slope_interval);
    add_interpolation_weight(
        interpolation,
        idx,
        -h11 * interval / hi_slope_interval,
        -dh11 / hi_slope_interval);
    add_interpolation_weight(
        interpolation,
        hi_slope_node,
        h11 * interval / hi_slope_interval,
        dh11 / hi_slope_interval);
    return interpolation;
}

double interpolate_ppf_value(
    const scar::copula::multivariate::student::PpfCache& cache,
    int dimension,
    const PpfInterpolation& interpolation,
    std::int64_t row_index,
    int column,
    double* derivative) {

    const std::size_t node_stride =
        static_cast<std::size_t>(cache.observation_count)
        * static_cast<std::size_t>(dimension);
    const std::size_t row_offset =
        static_cast<std::size_t>(row_index)
        * static_cast<std::size_t>(dimension)
        + static_cast<std::size_t>(column);
    double value = 0.0;
    double derivative_value = 0.0;
    for (int i = 0; i < interpolation.count; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const double table_value = cache.table[
            interpolation.node[index] * node_stride + row_offset];
        value += interpolation.value_weight[index] * table_value;
        derivative_value += (
            interpolation.derivative_weight[index] * table_value);
    }
    if (derivative != nullptr) {
        *derivative = derivative_value;
    }
    return value;
}

void interpolate_ppf_row(
    const scar::copula::multivariate::student::PpfCache& cache,
    int dimension,
    const PpfInterpolation& interpolation,
    std::int64_t row_index,
    double* values,
    double* derivatives) {

    const std::size_t dimension_size =
        static_cast<std::size_t>(dimension);
    const std::size_t node_stride =
        static_cast<std::size_t>(cache.observation_count) * dimension_size;
    const std::size_t row_offset =
        static_cast<std::size_t>(row_index) * dimension_size;
    for (std::size_t column = 0; column < dimension_size; ++column) {
        double value = 0.0;
        double derivative = 0.0;
        for (int i = 0; i < interpolation.count; ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            const double table_value = cache.table[
                interpolation.node[index] * node_stride
                + row_offset
                + column];
            value += interpolation.value_weight[index] * table_value;
            derivative +=
                interpolation.derivative_weight[index] * table_value;
        }
        values[column] = value;
        if (derivatives != nullptr) {
            derivatives[column] = derivative;
        }
    }
}

void interpolate_bivariate_ppf(
    const scar::copula::multivariate::student::PpfCache& cache,
    const PpfInterpolation& interpolation,
    std::size_t observation,
    double& first,
    double& second,
    double& first_derivative,
    double& second_derivative) {

    const std::size_t row_offset = observation * 2;
    const std::size_t node_stride =
        static_cast<std::size_t>(cache.observation_count) * 2;
    first = 0.0;
    second = 0.0;
    first_derivative = 0.0;
    second_derivative = 0.0;
    for (int i = 0; i < interpolation.count; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const std::size_t offset =
            interpolation.node[index] * node_stride + row_offset;
        const double value_weight = interpolation.value_weight[index];
        const double derivative_weight =
            interpolation.derivative_weight[index];
        const double table_first = cache.table[offset];
        const double table_second = cache.table[offset + 1];
        first += value_weight * table_first;
        second += value_weight * table_second;
        first_derivative += derivative_weight * table_first;
        second_derivative += derivative_weight * table_second;
    }
}

}  // namespace scar_internal
