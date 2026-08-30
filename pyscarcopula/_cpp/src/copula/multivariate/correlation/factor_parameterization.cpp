#include "scar/copula/multivariate/correlation/factor_parameterization.hpp"

#include "scar/copula/transforms.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace scar {
namespace {

constexpr std::size_t kFactorScoreWorkBytes = 32 * 1024 * 1024;
constexpr double kJointDiagonalRawFloor = 1e-10;

bool valid_factor_shape(
    DoubleView values,
    std::size_t dimension,
    std::size_t rank) {
    return dimension >= 2
        && rank >= 1
        && rank < dimension
        && values.size() == dimension * rank
        && values.data() != nullptr
        && std::all_of(
            values.data(),
            values.data() + values.size(),
            [](double value) { return std::isfinite(value); });
}

bool orthonormalize_columns(
    std::vector<double>& matrix,
    std::size_t rows,
    std::size_t columns) {

    const double threshold =
        std::sqrt(std::numeric_limits<double>::epsilon());
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t previous = 0;
             previous < column;
             ++previous) {
            long double projection = 0.0L;
            for (std::size_t row = 0; row < rows; ++row) {
                projection += static_cast<long double>(
                    matrix[row * columns + previous])
                    * static_cast<long double>(
                        matrix[row * columns + column]);
            }
            const double coefficient = static_cast<double>(projection);
            for (std::size_t row = 0; row < rows; ++row) {
                matrix[row * columns + column] -= coefficient
                    * matrix[row * columns + previous];
            }
        }
        long double norm_squared = 0.0L;
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = matrix[row * columns + column];
            norm_squared += static_cast<long double>(value) * value;
        }
        const double norm = std::sqrt(static_cast<double>(norm_squared));
        if (!(norm > threshold) || !std::isfinite(norm)) {
            return false;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            matrix[row * columns + column] /= norm;
        }
    }
    return true;
}

bool symmetric_eigen(
    std::vector<double> matrix,
    std::size_t dimension,
    std::vector<double>& values,
    std::vector<double>& vectors) {

    vectors.assign(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        vectors[index * dimension + index] = 1.0;
    }
    const std::size_t maximum_iterations =
        std::max<std::size_t>(64, 64 * dimension * dimension);
    const double tolerance =
        8.0 * std::numeric_limits<double>::epsilon();
    for (std::size_t iteration = 0;
         iteration < maximum_iterations;
         ++iteration) {
        double maximum = 0.0;
        double diagonal_scale = 1.0;
        std::size_t pivot_row = 0;
        std::size_t pivot_column = dimension > 1 ? 1 : 0;
        for (std::size_t row = 0; row < dimension; ++row) {
            diagonal_scale = std::max(
                diagonal_scale,
                std::abs(matrix[row * dimension + row]));
            for (std::size_t column = row + 1;
                 column < dimension;
                 ++column) {
                const double value =
                    std::abs(matrix[row * dimension + column]);
                if (value > maximum) {
                    maximum = value;
                    pivot_row = row;
                    pivot_column = column;
                }
            }
        }
        if (maximum <= tolerance * diagonal_scale) {
            values.resize(dimension);
            for (std::size_t index = 0; index < dimension; ++index) {
                values[index] = matrix[index * dimension + index];
            }
            return true;
        }
        const double app = matrix[pivot_row * dimension + pivot_row];
        const double aqq = matrix[pivot_column * dimension + pivot_column];
        const double apq = matrix[pivot_row * dimension + pivot_column];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t index = 0; index < dimension; ++index) {
            if (index == pivot_row || index == pivot_column) {
                continue;
            }
            const double aip = matrix[index * dimension + pivot_row];
            const double aiq = matrix[index * dimension + pivot_column];
            const double rotated_p = cosine * aip - sine * aiq;
            const double rotated_q = sine * aip + cosine * aiq;
            matrix[index * dimension + pivot_row] = rotated_p;
            matrix[pivot_row * dimension + index] = rotated_p;
            matrix[index * dimension + pivot_column] = rotated_q;
            matrix[pivot_column * dimension + index] = rotated_q;
        }
        matrix[pivot_row * dimension + pivot_row] =
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq;
        matrix[pivot_column * dimension + pivot_column] =
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq;
        matrix[pivot_row * dimension + pivot_column] = 0.0;
        matrix[pivot_column * dimension + pivot_row] = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            const double first = vectors[row * dimension + pivot_row];
            const double second = vectors[row * dimension + pivot_column];
            vectors[row * dimension + pivot_row] =
                cosine * first - sine * second;
            vectors[row * dimension + pivot_column] =
                sine * first + cosine * second;
        }
    }
    return false;
}

std::vector<std::size_t> pivot_anchor_rows(
    DoubleView loadings,
    std::size_t dimension,
    std::size_t rank) {

    std::vector<double> residual(loadings.data(), loadings.data() + loadings.size());
    std::vector<std::size_t> pivots(dimension);
    std::iota(pivots.begin(), pivots.end(), 0);
    std::vector<double> basis(rank * rank, 0.0);
    for (std::size_t order = 0; order < rank; ++order) {
        std::size_t best = order;
        double best_norm = -1.0;
        for (std::size_t candidate = order;
             candidate < dimension;
             ++candidate) {
            const std::size_t row = pivots[candidate];
            double norm = 0.0;
            for (std::size_t column = 0; column < rank; ++column) {
                const double value = residual[row * rank + column];
                norm += value * value;
            }
            if (norm > best_norm) {
                best_norm = norm;
                best = candidate;
            }
        }
        std::swap(pivots[order], pivots[best]);
        const std::size_t anchor = pivots[order];
        const double norm = std::sqrt(std::max(best_norm, 0.0));
        if (!(norm > 0.0)) {
            continue;
        }
        for (std::size_t column = 0; column < rank; ++column) {
            basis[column * rank + order] =
                residual[anchor * rank + column] / norm;
        }
        for (std::size_t candidate = order + 1;
             candidate < dimension;
             ++candidate) {
            const std::size_t row = pivots[candidate];
            double projection = 0.0;
            for (std::size_t column = 0; column < rank; ++column) {
                projection += residual[row * rank + column]
                    * basis[column * rank + order];
            }
            for (std::size_t column = 0; column < rank; ++column) {
                residual[row * rank + column] -= projection
                    * basis[column * rank + order];
            }
        }
    }
    pivots.resize(rank);
    return pivots;
}

bool descriptor_indices(
    DoubleView free_rows,
    DoubleView free_columns,
    DoubleView diagonal_entries,
    std::size_t parameter_count,
    std::size_t dimension,
    std::size_t rank,
    std::vector<std::size_t>& rows,
    std::vector<std::size_t>& columns,
    std::vector<bool>& diagonal) {

    if (free_rows.size() != parameter_count
        || free_columns.size() != parameter_count
        || diagonal_entries.size() != parameter_count) {
        return false;
    }
    rows.resize(parameter_count);
    columns.resize(parameter_count);
    diagonal.resize(parameter_count);
    for (std::size_t index = 0; index < parameter_count; ++index) {
        const double row_value = free_rows[index];
        const double column_value = free_columns[index];
        if (!std::isfinite(row_value)
            || !std::isfinite(column_value)
            || row_value < 0.0
            || column_value < 0.0
            || row_value != std::floor(row_value)
            || column_value != std::floor(column_value)
            || row_value >= static_cast<double>(dimension)
            || column_value >= static_cast<double>(rank)) {
            return false;
        }
        rows[index] = static_cast<std::size_t>(row_value);
        columns[index] = static_cast<std::size_t>(column_value);
        diagonal[index] = diagonal_entries[index] != 0.0;
    }
    return true;
}

bool raw_factor_matrix(
    DoubleView parameters,
    DoubleView free_rows,
    DoubleView free_columns,
    DoubleView diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    std::vector<double>& raw,
    std::vector<std::size_t>& rows,
    std::vector<std::size_t>& columns,
    std::vector<bool>& diagonal) {

    if (parameters.data() == nullptr
        || !std::all_of(
            parameters.data(),
            parameters.data() + parameters.size(),
            [](double value) { return std::isfinite(value); })
        || !descriptor_indices(
            free_rows,
            free_columns,
            diagonal_entries,
            parameters.size(),
            dimension,
            rank,
            rows,
            columns,
            diagonal)) {
        return false;
    }
    raw.assign(dimension * rank, 0.0);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        double value = parameters[index];
        if (diagonal[index]) {
            value = copula::softplus(value) + kJointDiagonalRawFloor;
        }
        raw[rows[index] * rank + columns[index]] = value;
    }
    return true;
}

std::vector<double> standardized_scores(
    ObservationView observations,
    std::size_t coordinate) {

    const std::size_t rows = observations.n_obs;
    std::vector<double> scores(rows);
    long double mean = 0.0L;
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (std::size_t row = 0; row < rows; ++row) {
        const double probability = std::clamp(
            observations.row(row)[coordinate],
            epsilon,
            1.0 - epsilon);
        scores[row] = scar::math::normal_quantile_refined(probability);
        mean += scores[row];
    }
    const double center = static_cast<double>(mean / rows);
    long double variance = 0.0L;
    for (double& value : scores) {
        value -= center;
        variance += static_cast<long double>(value) * value;
    }
    const double scale = std::sqrt(static_cast<double>(variance / rows));
    if (!(scale > std::sqrt(epsilon))) {
        std::fill(scores.begin(), scores.end(), 0.0);
    } else {
        for (double& value : scores) {
            value /= scale;
        }
    }
    return scores;
}

std::vector<double> standardized_score_block(
    ObservationView observations,
    std::size_t start,
    std::size_t stop) {

    const std::size_t rows = observations.n_obs;
    std::vector<double> block((stop - start) * rows, 0.0);
    for (std::size_t coordinate = start;
         coordinate < stop;
         ++coordinate) {
        const std::vector<double> scores = standardized_scores(
            observations, coordinate);
        std::copy(
            scores.begin(),
            scores.end(),
            block.begin() + (coordinate - start) * rows);
    }
    return block;
}

}  // namespace

FactorCorrelationTransformResult factor_correlation_from_loadings(
    DoubleView loadings,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min) {

    FactorCorrelationTransformResult result;
    result.dimension = dimension;
    result.rank = rank;
    if (!valid_factor_shape(loadings, dimension, rank)
        || !std::isfinite(uniqueness_min)
        || !(uniqueness_min > 0.0 && uniqueness_min < 1.0)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    result.loadings.assign(loadings.data(), loadings.data() + loadings.size());
    result.uniqueness.resize(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        long double norm_squared = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            const double value = loadings[row * rank + column];
            norm_squared += static_cast<long double>(value) * value;
        }
        result.uniqueness[row] = 1.0 - static_cast<double>(norm_squared);
        if (result.uniqueness[row] < uniqueness_min) {
            result.status = Status::InvalidParameter;
            result.failure.coordinate = static_cast<int>(row);
            return result;
        }
    }
    return result;
}

FactorCorrelationTransformResult factor_correlation_from_unconstrained(
    DoubleView values,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min) {

    FactorCorrelationTransformResult result;
    result.dimension = dimension;
    result.rank = rank;
    if (!valid_factor_shape(values, dimension, rank)
        || !std::isfinite(uniqueness_min)
        || !(uniqueness_min > 0.0 && uniqueness_min < 1.0)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    const double max_norm = std::sqrt(std::nextafter(
        1.0 - uniqueness_min, 0.0));
    result.loadings.assign(dimension * rank, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        double row_scale = 0.0;
        for (std::size_t column = 0; column < rank; ++column) {
            row_scale = std::max(
                row_scale, std::abs(values[row * rank + column]));
        }
        long double scaled_norm_squared = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            const double scaled = row_scale > 0.0
                ? values[row * rank + column] / row_scale
                : 0.0;
            result.loadings[row * rank + column] = scaled;
            scaled_norm_squared += static_cast<long double>(scaled) * scaled;
        }
        const double inverse_scale = row_scale > 0.0 ? 1.0 / row_scale : 0.0;
        const double denominator = std::sqrt(
            static_cast<double>(scaled_norm_squared)
            + inverse_scale * inverse_scale);
        long double norm_squared = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            double& loading = result.loadings[row * rank + column];
            loading = denominator > 0.0 ? loading / denominator : 0.0;
            norm_squared += static_cast<long double>(loading) * loading;
        }
        const double norm = std::sqrt(static_cast<double>(norm_squared));
        const double scale = norm > max_norm ? max_norm / norm : 1.0;
        for (std::size_t column = 0; column < rank; ++column) {
            result.loadings[row * rank + column] *= scale;
        }
    }
    FactorCorrelationTransformResult validated = factor_correlation_from_loadings(
        {result.loadings.data(), result.loadings.size()},
        dimension,
        rank,
        uniqueness_min);
    return validated;
}

Result<std::vector<double>> factor_correlation_to_dense(
    DoubleView loadings,
    DoubleView uniqueness,
    std::size_t dimension,
    std::size_t rank) {

    if (!valid_factor_shape(loadings, dimension, rank)
        || uniqueness.size() != dimension
        || uniqueness.data() == nullptr) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> dense(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            long double value = 0.0L;
            for (std::size_t factor = 0; factor < rank; ++factor) {
                value += static_cast<long double>(
                    loadings[row * rank + factor])
                    * static_cast<long double>(
                        loadings[column * rank + factor]);
            }
            if (row == column) {
                value += uniqueness[row];
            }
            dense[row * dimension + column] = static_cast<double>(value);
            dense[column * dimension + row] = static_cast<double>(value);
        }
    }
    return success(std::move(dense));
}

FactorLoadingParameterizationResult factor_parameterization_from_loadings(
    DoubleView loadings,
    std::size_t dimension,
    std::size_t rank,
    double uniqueness_min) {

    FactorLoadingParameterizationResult result;
    result.dimension = dimension;
    result.rank = rank;
    if (!valid_factor_shape(loadings, dimension, rank)
        || rank > (dimension - 1) / 2
        || !std::isfinite(uniqueness_min)
        || !(uniqueness_min > 0.0 && uniqueness_min < 1.0)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    result.max_norm = std::sqrt(std::nextafter(
        1.0 - uniqueness_min, 0.0));
    const std::vector<std::size_t> anchors = pivot_anchor_rows(
        loadings, dimension, rank);
    result.anchors.reserve(rank);
    for (std::size_t value : anchors) {
        result.anchors.push_back(static_cast<std::int64_t>(value));
    }

    std::vector<double> rotation(rank * rank, 0.0);
    for (std::size_t row = 0; row < rank; ++row) {
        for (std::size_t column = 0; column < rank; ++column) {
            rotation[row * rank + column] =
                loadings[anchors[column] * rank + row];
        }
    }
    if (!orthonormalize_columns(rotation, rank, rank)) {
        result.status = Status::NumericalFailure;
        return result;
    }
    std::vector<double> canonical(dimension * rank, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < rank; ++column) {
            for (std::size_t component = 0;
                 component < rank;
                 ++component) {
                canonical[row * rank + column] +=
                    loadings[row * rank + component]
                    * rotation[component * rank + column];
            }
        }
    }
    for (std::size_t column = 0; column < rank; ++column) {
        if (canonical[anchors[column] * rank + column] < 0.0) {
            for (std::size_t row = 0; row < dimension; ++row) {
                canonical[row * rank + column] *= -1.0;
            }
        }
    }
    const double anchor_floor = std::min(
        1e-6, result.max_norm * 1e-3);
    for (std::size_t order = 0; order < rank; ++order) {
        const std::size_t row = anchors[order];
        for (std::size_t column = order + 1;
             column < rank;
             ++column) {
            canonical[row * rank + column] = 0.0;
        }
        canonical[row * rank + order] = std::max(
            canonical[row * rank + order], anchor_floor);
        long double norm_squared = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            const double value = canonical[row * rank + column];
            norm_squared += static_cast<long double>(value) * value;
        }
        if (norm_squared >= result.max_norm * result.max_norm) {
            const double scale = result.max_norm * (1.0 - 1e-8)
                / std::sqrt(static_cast<double>(norm_squared));
            for (std::size_t column = 0; column < rank; ++column) {
                canonical[row * rank + column] *= scale;
            }
        }
    }

    std::vector<double> raw(dimension * rank, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        long double norm_squared = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            const double value = canonical[row * rank + column];
            norm_squared += static_cast<long double>(value) * value;
        }
        const double denominator = std::sqrt(std::max(
            result.max_norm * result.max_norm
                - static_cast<double>(norm_squared),
            std::numeric_limits<double>::min()));
        for (std::size_t column = 0; column < rank; ++column) {
            raw[row * rank + column] =
                canonical[row * rank + column] / denominator;
        }
    }
    std::vector<std::int64_t> anchor_order(dimension, -1);
    for (std::size_t order = 0; order < rank; ++order) {
        anchor_order[anchors[order]] = static_cast<std::int64_t>(order);
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        const std::int64_t order = anchor_order[row];
        const std::size_t stop = order >= 0
            ? static_cast<std::size_t>(order) + 1
            : rank;
        for (std::size_t column = 0; column < stop; ++column) {
            const bool diagonal = order >= 0
                && column == static_cast<std::size_t>(order);
            result.free_rows.push_back(static_cast<std::int64_t>(row));
            result.free_columns.push_back(static_cast<std::int64_t>(column));
            result.diagonal_entries.push_back(diagonal ? 1 : 0);
            double value = raw[row * rank + column];
            if (diagonal) {
                value = copula::inverse_softplus(std::max(
                    value - kJointDiagonalRawFloor,
                    std::numeric_limits<double>::epsilon()));
            }
            result.parameters.push_back(value);
        }
    }
    return result;
}

Result<std::vector<double>> factor_parameterization_loadings(
    DoubleView parameters,
    DoubleView free_rows,
    DoubleView free_columns,
    DoubleView diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    double max_norm) {

    std::vector<double> raw;
    std::vector<std::size_t> rows;
    std::vector<std::size_t> columns;
    std::vector<bool> diagonal;
    if (!std::isfinite(max_norm)
        || !(max_norm > 0.0)
        || !raw_factor_matrix(
            parameters,
            free_rows,
            free_columns,
            diagonal_entries,
            dimension,
            rank,
            raw,
            rows,
            columns,
            diagonal)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> loadings(dimension * rank, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        long double norm_squared = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            const double value = raw[row * rank + column];
            norm_squared += static_cast<long double>(value) * value;
        }
        const double denominator = std::sqrt(
            1.0 + static_cast<double>(norm_squared));
        for (std::size_t column = 0; column < rank; ++column) {
            loadings[row * rank + column] =
                max_norm * raw[row * rank + column] / denominator;
        }
    }
    return success(std::move(loadings));
}

Result<std::vector<double>> factor_parameterization_pullback(
    DoubleView parameters,
    DoubleView loading_gradient,
    DoubleView free_rows,
    DoubleView free_columns,
    DoubleView diagonal_entries,
    std::size_t dimension,
    std::size_t rank,
    double max_norm) {

    if (!valid_factor_shape(
            loading_gradient, dimension, rank)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> raw;
    std::vector<std::size_t> rows;
    std::vector<std::size_t> columns;
    std::vector<bool> diagonal;
    if (!std::isfinite(max_norm)
        || !(max_norm > 0.0)
        || !raw_factor_matrix(
            parameters,
            free_rows,
            free_columns,
            diagonal_entries,
            dimension,
            rank,
            raw,
            rows,
            columns,
            diagonal)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> raw_gradient(dimension * rank, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        long double norm_squared = 0.0L;
        long double projection = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            const double raw_value = raw[row * rank + column];
            norm_squared += static_cast<long double>(raw_value) * raw_value;
            projection += static_cast<long double>(raw_value)
                * loading_gradient[row * rank + column];
        }
        const double denominator = std::sqrt(
            1.0 + static_cast<double>(norm_squared));
        const double projection_scale = static_cast<double>(projection)
            / (denominator * denominator * denominator);
        for (std::size_t column = 0; column < rank; ++column) {
            raw_gradient[row * rank + column] = max_norm * (
                loading_gradient[row * rank + column] / denominator
                - raw[row * rank + column] * projection_scale);
        }
    }
    std::vector<double> result(parameters.size());
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        result[index] = raw_gradient[rows[index] * rank + columns[index]];
        if (diagonal[index]) {
            result[index] *= copula::logistic_unit(parameters[index]);
        }
    }
    return success(std::move(result));
}

FactorInitializationResult estimate_factor_loadings(
    ObservationView observations,
    std::size_t rank,
    double uniqueness_min,
    std::size_t configured_dimension_tile,
    DoubleMatrixView random_projection) {

    FactorInitializationResult result;
    const std::size_t rows = observations.n_obs;
    const std::size_t dimension = observations.dim > 0
        ? static_cast<std::size_t>(observations.dim)
        : 0;
    const std::size_t subspace = random_projection.dim > 0
        ? static_cast<std::size_t>(random_projection.dim)
        : 0;
    result.dimension = dimension;
    result.rank = rank;
    result.subspace_size = subspace;
    if (observations.data() == nullptr
        || rows == 0
        || dimension < 2
        || rank < 1
        || rank >= dimension
        || rows <= rank
        || subspace < rank
        || subspace > dimension
        || subspace > rows
        || random_projection.data() == nullptr
        || random_projection.n_obs != dimension
        || configured_dimension_tile < 1
        || !std::isfinite(uniqueness_min)
        || !(uniqueness_min > 0.0 && uniqueness_min < 1.0)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    for (std::size_t index = 0; index < rows * dimension; ++index) {
        const double value = observations.data()[index];
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(index);
            return result;
        }
    }
    for (std::size_t index = 0; index < dimension * subspace; ++index) {
        if (!std::isfinite(random_projection.data()[index])) {
            result.status = Status::InvalidParameter;
            result.failure.index = static_cast<std::int64_t>(index);
            return result;
        }
    }
    result.score_tile = std::min(
        configured_dimension_tile,
        std::max<std::size_t>(
            1,
            kFactorScoreWorkBytes
                / (rows * sizeof(double))));

    std::vector<double> sample_subspace(rows * subspace, 0.0);
    for (std::size_t start = 0;
         start < dimension;
         start += result.score_tile) {
        const std::size_t stop = std::min(
            dimension, start + result.score_tile);
        const std::vector<double> score_block = standardized_score_block(
            observations, start, stop);
        for (std::size_t coordinate = start;
             coordinate < stop;
             ++coordinate) {
            const double* scores = score_block.data()
                + (coordinate - start) * rows;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t component = 0;
                     component < subspace;
                     ++component) {
                    sample_subspace[row * subspace + component] +=
                        scores[row]
                        * random_projection.row(coordinate)[component];
                }
            }
        }
    }
    if (!orthonormalize_columns(sample_subspace, rows, subspace)) {
        result.status = Status::NumericalFailure;
        return result;
    }

    std::vector<double> variable_subspace(dimension * subspace, 0.0);
    for (std::size_t start = 0;
         start < dimension;
         start += result.score_tile) {
        const std::size_t stop = std::min(
            dimension, start + result.score_tile);
        const std::vector<double> score_block = standardized_score_block(
            observations, start, stop);
        for (std::size_t coordinate = start;
             coordinate < stop;
             ++coordinate) {
            const double* scores = score_block.data()
                + (coordinate - start) * rows;
            for (std::size_t component = 0;
                 component < subspace;
                 ++component) {
                long double value = 0.0L;
                for (std::size_t row = 0; row < rows; ++row) {
                    value += static_cast<long double>(scores[row])
                        * sample_subspace[row * subspace + component];
                }
                variable_subspace[coordinate * subspace + component] =
                    static_cast<double>(value);
            }
        }
    }
    std::fill(sample_subspace.begin(), sample_subspace.end(), 0.0);
    for (std::size_t start = 0;
         start < dimension;
         start += result.score_tile) {
        const std::size_t stop = std::min(
            dimension, start + result.score_tile);
        const std::vector<double> score_block = standardized_score_block(
            observations, start, stop);
        for (std::size_t coordinate = start;
             coordinate < stop;
             ++coordinate) {
            const double* scores = score_block.data()
                + (coordinate - start) * rows;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t component = 0;
                     component < subspace;
                     ++component) {
                    sample_subspace[row * subspace + component] +=
                        scores[row]
                        * variable_subspace[
                            coordinate * subspace + component];
                }
            }
        }
    }
    if (!orthonormalize_columns(sample_subspace, rows, subspace)) {
        result.status = Status::NumericalFailure;
        return result;
    }

    std::vector<double> compressed(subspace * dimension, 0.0);
    for (std::size_t start = 0;
         start < dimension;
         start += result.score_tile) {
        const std::size_t stop = std::min(
            dimension, start + result.score_tile);
        const std::vector<double> score_block = standardized_score_block(
            observations, start, stop);
        for (std::size_t coordinate = start;
             coordinate < stop;
             ++coordinate) {
            const double* scores = score_block.data()
                + (coordinate - start) * rows;
            for (std::size_t component = 0;
                 component < subspace;
                 ++component) {
                long double value = 0.0L;
                for (std::size_t row = 0; row < rows; ++row) {
                    value += static_cast<long double>(
                        sample_subspace[row * subspace + component])
                        * scores[row];
                }
                compressed[component * dimension + coordinate] =
                    static_cast<double>(value);
            }
        }
    }
    std::vector<double> gram(subspace * subspace, 0.0);
    for (std::size_t row = 0; row < subspace; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            long double value = 0.0L;
            for (std::size_t coordinate = 0;
                 coordinate < dimension;
                 ++coordinate) {
                value += static_cast<long double>(
                    compressed[row * dimension + coordinate])
                    * compressed[column * dimension + coordinate];
            }
            gram[row * subspace + column] = static_cast<double>(value);
            gram[column * subspace + row] = static_cast<double>(value);
        }
    }
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    if (!symmetric_eigen(
            std::move(gram), subspace, eigenvalues, eigenvectors)) {
        result.status = Status::NumericalFailure;
        return result;
    }
    std::vector<std::size_t> order(subspace);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](std::size_t left, std::size_t right) {
            return eigenvalues[left] > eigenvalues[right];
        });

    result.loadings.assign(dimension * rank, 0.0);
    result.leading_eigenvalues.resize(rank);
    for (std::size_t factor = 0; factor < rank; ++factor) {
        const std::size_t component = order[factor];
        const double singular = std::sqrt(std::max(
            eigenvalues[component], 0.0));
        const double covariance_eigenvalue =
            singular * singular / static_cast<double>(rows);
        result.leading_eigenvalues[factor] = covariance_eigenvalue;
        const double loading_scale = std::sqrt(std::max(
            covariance_eigenvalue - 1.0, 0.0));
        if (!(singular > 0.0) || loading_scale == 0.0) {
            continue;
        }
        for (std::size_t coordinate = 0;
             coordinate < dimension;
             ++coordinate) {
            long double right_value = 0.0L;
            for (std::size_t row = 0; row < subspace; ++row) {
                right_value += static_cast<long double>(
                    compressed[row * dimension + coordinate])
                    * eigenvectors[row * subspace + component];
            }
            result.loadings[coordinate * rank + factor] =
                static_cast<double>(right_value) / singular * loading_scale;
        }
    }
    const double max_norm = std::sqrt(std::nextafter(
        1.0 - uniqueness_min, 0.0));
    for (std::size_t row = 0; row < dimension; ++row) {
        long double norm_squared = 0.0L;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            const double value = result.loadings[row * rank + factor];
            norm_squared += static_cast<long double>(value) * value;
        }
        const double norm = std::sqrt(static_cast<double>(norm_squared));
        const double scale = norm > max_norm ? max_norm / norm : 1.0;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            result.loadings[row * rank + factor] *= scale;
        }
    }
    return result;
}

}  // namespace scar
