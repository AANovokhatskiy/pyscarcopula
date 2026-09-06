#include "scar/copula/multivariate/student/ppf_cache.hpp"

#include "scar/copula/model_storage.hpp"
#include "scar/copula/spec.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

namespace {

void append_nodes(std::vector<double>& nodes, double lo, double hi,
                  int count, bool geometric, double shift = 0.0) {
    if (count == 0) {
        return;
    }
    const double start = geometric ? std::log10(lo) : lo;
    const double stop = geometric ? std::log10(hi) : hi;
    const double step = count > 1 ? (stop - start) / (count - 1) : 0.0;
    for (int i = 0; i < count; ++i) {
        double value = start + static_cast<double>(i) * step;
        if (geometric) {
            value = std::pow(10.0, value);
        }
        if (i == 0) value = lo;
        if (i == count - 1 && count > 1) value = hi;
        nodes.push_back(value + shift);
    }
}

bool valid_nodes(DoubleView nodes) {
    if (nodes.size() < 2 || nodes.data() == nullptr) return false;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (!std::isfinite(nodes[i]) || nodes[i] <= 0.0
            || (i > 0 && nodes[i] <= nodes[i - 1])) return false;
    }
    return true;
}

}  // namespace

Result<PreparedPpfTable> prepare_ppf_table(
    DoubleView observations, const PpfTableConfig& config) {

    Result<PreparedPpfTable> result;
    if (!(config.df_lo > 2.0 && config.df_lo < 2.005)
        || !std::isfinite(config.df_hi) || config.df_hi <= 5.0
        || config.n_boundary < 0 || config.n_lo < 0 || config.n_hi < 0) {
        result.status = Status::InvalidParameter;
        return result;
    }
    const std::uint64_t node_count = 1ULL + config.n_boundary + config.n_lo + config.n_hi;
    if (node_count > 1000000 || (!observations.empty() && observations.data() == nullptr)) {
        result.status = Status::InvalidSize;
        return result;
    }
    auto& out = result.value;
    out.nodes.reserve(static_cast<std::size_t>(node_count));
    out.nodes.push_back(config.df_lo);
    append_nodes(out.nodes, 1e-10, 2.005 - config.df_lo, config.n_boundary, true, config.df_lo);
    append_nodes(out.nodes, 2.005, 5.0, config.n_lo, false);
    append_nodes(out.nodes, 5.0, config.df_hi, config.n_hi, true);
    std::sort(out.nodes.begin(), out.nodes.end());
    out.nodes.erase(std::unique(out.nodes.begin(), out.nodes.end()), out.nodes.end());
    if (out.nodes.size() < 2) {
        result.status = Status::InvalidParameter;
        return result;
    }
    out.observations.resize(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
        if (!std::isfinite(observations[i])) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(i);
            result.value = {};
            return result;
        }
        out.observations[i] = scar_internal::clip_pseudo_observation(observations[i]);
    }
    std::size_t elements = 0;
    std::uint64_t bytes = 0;
    if (!core::checked_size_mul(out.nodes.size(), observations.size(), elements)
        || !core::checked_byte_count<double>(elements, bytes)
        || bytes > config.max_table_bytes) {
        return result;
    }
    out.table.resize(elements);
    out.has_table = true;
    for (std::size_t node = 0; node < out.nodes.size(); ++node) {
        const auto distribution = scar_internal::student_distribution_parameters(out.nodes[node]);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            // Adjacent df nodes provide a close bracket without changing
            // the exact solver's convergence criterion or table values.
            const double initial = node == 0
                ? std::numeric_limits<double>::quiet_NaN()
                : out.table[(node - 1) * observations.size() + i];
            const double value = scar_internal::student_quantile_refined_value(
                out.observations[i], distribution, initial);
            if (!std::isfinite(value)) {
                result.status = Status::NumericalFailure;
                result.failure.index = static_cast<std::int64_t>(i);
                result.value = {};
                return result;
            }
            out.table[node * observations.size() + i] = value;
        }
    }
    return result;
}

Result<std::vector<double>> evaluate_ppf_table(
    DoubleView observations, DoubleView nodes, DoubleView table,
    double df, std::size_t offset, std::size_t count) {

    Result<std::vector<double>> result;
    std::size_t elements = 0;
    if (!std::isfinite(df) || df <= 0.0 || !valid_nodes(nodes)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    if (offset > observations.size() || count > observations.size() - offset
        || (!observations.empty() && observations.data() == nullptr)
        || !core::checked_size_mul(nodes.size(), observations.size(), elements)
        || (!table.empty() && (table.size() != elements || table.data() == nullptr))) {
        result.status = Status::InvalidSize;
        return result;
    }
    const bool interpolate = !table.empty() && df >= nodes[0] && df <= nodes[nodes.size() - 1];
    const auto interpolation = scar_internal::make_ppf_interpolation(nodes, df);
    const auto distribution = scar_internal::student_distribution_parameters(df);
    result.value.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(observations[offset + i])) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(offset + i);
            result.value.clear();
            return result;
        }
        const double value = interpolate
            ? scar_internal::interpolate_ppf_value(
                table, observations.size(), interpolation, offset + i, nullptr)
            : scar_internal::student_quantile_refined_value(observations[offset + i], distribution);
        if (!std::isfinite(value)) {
            result.status = Status::NumericalFailure;
            result.failure.index = static_cast<std::int64_t>(offset + i);
            result.value.clear();
            return result;
        }
        result.value[i] = value;
    }
    return result;
}

Result<std::vector<double>> interpolate_ppf_table(
    DoubleView nodes, DoubleView table, double df, std::size_t width) {

    Result<std::vector<double>> result;
    if (!valid_nodes(nodes) || !std::isfinite(df)
        || df < nodes[0] || df > nodes[nodes.size() - 1]) {
        result.status = Status::InvalidParameter;
        return result;
    }
    std::size_t elements = 0;
    if (!core::checked_size_mul(nodes.size(), width, elements)
        || elements != table.size() || (elements > 0 && table.data() == nullptr)) {
        result.status = Status::InvalidSize;
        return result;
    }
    const auto interpolation = scar_internal::make_ppf_interpolation(nodes, df);
    result.value.resize(width);
    for (std::size_t i = 0; i < width; ++i) {
        result.value[i] = scar_internal::interpolate_ppf_value(
            table, width, interpolation, i, nullptr);
    }
    return result;
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

    return make_ppf_interpolation(scar::DoubleView{nodes.data(), nodes.size()}, df);
}

PpfInterpolation make_ppf_interpolation(scar::DoubleView nodes, double df) {

    const auto upper = std::upper_bound(nodes.data(), nodes.data() + nodes.size(), df);
    std::size_t idx = upper == nodes.data()
        ? 0
        : static_cast<std::size_t>(upper - nodes.data() - 1);
    idx = std::min(idx, nodes.size() - 2);

    const double lo = nodes[idx];
    const double hi = nodes[idx + 1];
    const double interval = hi - lo;
    const double alpha = (df - lo) / interval;
    PpfInterpolation interpolation;
    if (df <= nodes[0] || df >= nodes[nodes.size() - 1]) {
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
    return interpolate_ppf_value(
        scar::DoubleView{cache.table.data(), cache.table.size()},
        node_stride, interpolation, row_offset, derivative);
}

double interpolate_ppf_value(
    scar::DoubleView table, std::size_t node_stride,
    const PpfInterpolation& interpolation, std::size_t offset, double* derivative) {

    double value = 0.0;
    double derivative_value = 0.0;
    for (int i = 0; i < interpolation.count; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const double table_value = table[
            interpolation.node[index] * node_stride + offset];
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
